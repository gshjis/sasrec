"""Обучение SASRec для next-item prediction с in-batch negatives и опциональной logQ коррекцией.

Режимы:
  --logQ False  : обычные in-batch negative sampling (cross-entropy по диагонали)
  --logQ True   : in-batch logQ negative sampling (вычитаем log(q_item) из логитов целевых колонок)

Валидация: полное скорование всех айтемов (full-sort).
Метрики снимаются по K ∈ {10, 50, 100} с стратификацией по популярности (head / mid / tail).
logQ не применяется при валидации.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader

from data_extractor.config import HfConfig
from data_extractor.hf_loader import HfYambdaLoader
from data_preprocessing.tokenizer import Tokenizer
from models.sasrec.batch_formation import NextItemBatchFormation
from models.sasrec.metrics import (
    compute_strata,
    entropy_from_logits,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    targets_to_strata_tensor,
)
from models.sasrec.model import SASRec


def _detect_kaggle() -> bool:
    """Автоопределение окружения Kaggle kernel."""
    return Path("/kaggle/working").exists()


def _default_data_dir() -> str:
    """Путь по умолчанию для данных: /kaggle/working/data в Kaggle, ./data локально."""
    return "/kaggle/working/data" if _detect_kaggle() else "./data"


def _default_log_dir() -> str:
    """Путь по умолчанию для логов: /kaggle/working/logs в Kaggle, ./logs локально."""
    return "/kaggle/working/logs" if _detect_kaggle() else "./logs"


LOG_DIR: Path | None = None  # будет инициализировано в main()


def setup_logging(log_dir: Path) -> None:
    """Настроить логирование в stdout и в файл log_dir/train.log."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    # Убираем старые хендлеры, если переинициализируем
    for h in list(logger.handlers):
        logger.removeHandler(h)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


logger = logging.getLogger("train_model")
logger.setLevel(logging.INFO)


def train_epoch(
    model: SASRec,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    logQ: bool = False,
    item_freq: dict[int, float] | None = None,
) -> float:
    """Одна эпоха обучения. Возвращает средний loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)

        valid_mask = targets >= 0
        if not valid_mask.any():
            continue

        input_ids = input_ids[valid_mask]
        attention_mask = attention_mask[valid_mask]
        targets = targets[valid_mask]

        batch_size = targets.size(0)

        # user_emb: [batch, embed_dim]
        user_emb = model(input_ids, attention_mask)

        # item_emb для всех таргетов в батче: [batch, embed_dim]
        target_embs = model.item_emb(targets)

        # logits: [batch, batch] — diagonal = positives
        logits = user_emb @ target_embs.T

        if logQ:
            # logQ коррекция: вычитаем log(q_item) из каждой колонки j
            target_items = targets.detach().cpu().tolist()
            q_values = []
            for item_id in target_items:
                q = item_freq.get(int(item_id), 1e-6) if item_freq else 1e-6
                q_values.append(max(q, 1e-6))
            q_tensor = torch.tensor(q_values, device=device, dtype=torch.float)
            log_q = torch.log(q_tensor)  # [batch]
            logits = logits - log_q.unsqueeze(0)  # вычитаем из каждой колонки

        labels = torch.arange(batch_size, device=device)
        loss = torch.nn.functional.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(
    model: SASRec,
    dataloader: DataLoader,
    device: torch.device,
    num_items: int,
    item_freq: dict[int, float] | None = None,
    tail_quantile: float = 0.5,
    ks: tuple[int, ...] = (10, 50, 100),
) -> dict[str, float]:
    """Валидация на полном скоровании всех айтемов. logQ не применяется.

    Снимает метрики по каждому K из ks с стратификацией по популярности
    (head / mid / tail) для Recall@K и NDCG@K, а также покрытие (в т.ч. head/tail).
    """
    model.eval()
    total_examples = 0

    # Агрегаторы по каждому K
    total_recall: dict[int, float] = {k: 0.0 for k in ks}
    total_ndcg: dict[int, float] = {k: 0.0 for k in ks}
    total_mrr: dict[int, float] = {k: 0.0 for k in ks}
    total_entropy = 0.0

    # Страты по популярности
    strata = compute_strata(item_freq) if item_freq else {}
    tail_items: set[int] = set()
    head_items: set[int] = set()
    if item_freq:
        freq_values = sorted(item_freq.values())
        n = len(freq_values)
        tail_threshold = freq_values[int(n * tail_quantile)]
        head_threshold = freq_values[int(n * (1 - tail_quantile))]
        tail_items = {
            item_id for item_id, f in item_freq.items() if f <= tail_threshold
        }
        head_items = {
            item_id for item_id, f in item_freq.items() if f >= head_threshold
        }

    covered_items_by_k: dict[int, set[int]] = {k: set() for k in ks}
    tail_covered_by_k: dict[int, set[int]] = {k: set() for k in ks}
    head_covered_by_k: dict[int, set[int]] = {k: set() for k in ks}

    # Стратифицированные агрегаторы
    stratum_to_idx = {"head": 0, "mid": 1, "tail": 2}
    total_hits_by_stratum: dict[int, dict[str, float]] = {
        k: {"head": 0.0, "mid": 0.0, "tail": 0.0} for k in ks
    }
    total_ndcg_sum_by_stratum: dict[int, dict[str, float]] = {
        k: {"head": 0.0, "mid": 0.0, "tail": 0.0} for k in ks
    }
    stratum_counts: dict[str, int] = {"head": 0, "mid": 0, "tail": 0}

    max_k = max(ks)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            valid_mask = targets >= 0
            if not valid_mask.any():
                continue

            input_ids = input_ids[valid_mask]
            attention_mask = attention_mask[valid_mask]
            targets = targets[valid_mask]

            batch_size = targets.size(0)

            # user_emb: [batch, embed_dim]
            user_emb = model(input_ids, attention_mask)

            # Полные логиты по всем айтемам: [batch, num_items]
            logits = model.score_all_items(user_emb)

            # Исключаем PAD (0) из кандидатов
            logits[:, 0] = float("-inf")

            # Топ-K для максимального K (срезаем для меньших K)
            _, topk_max = torch.topk(logits, k=min(max_k, logits.size(1)), dim=1)

            # Страты таргетов в этом батче
            target_strata = (
                targets_to_strata_tensor(targets, strata, device) if strata else None
            )

            total_examples += batch_size
            total_entropy += entropy_from_logits(logits) * batch_size

            for k in ks:
                # Общие метрики
                total_recall[k] += recall_at_k(logits, targets, k) * batch_size
                total_ndcg[k] += ndcg_at_k(logits, targets, k) * batch_size
                total_mrr[k] += mrr_at_k(logits, targets, k) * batch_size

                # Стратифицированные метрики
                if target_strata is not None:
                    for s in ("head", "mid", "tail"):
                        mask = target_strata == stratum_to_idx[s]
                        if mask.any():
                            stratum_counts[s] += int(mask.sum().item())
                            total_hits_by_stratum[k][s] += (
                                recall_at_k(logits[mask], targets[mask], k)
                                * mask.sum().item()
                            )
                            total_ndcg_sum_by_stratum[k][s] += (
                                ndcg_at_k(logits[mask], targets[mask], k)
                                * mask.sum().item()
                            )

                # Покрытие
                topk_k = topk_max[:, :k]
                for items in topk_k.tolist():
                    covered_items_by_k[k].update(int(x) for x in items)
                    if item_freq:
                        tail_hits = {int(x) for x in items if int(x) in tail_items}
                        tail_covered_by_k[k].update(tail_hits)
                        head_hits = {int(x) for x in items if int(x) in head_items}
                        head_covered_by_k[k].update(head_hits)

    denom = max(total_examples, 1)
    num_items_denom = max(num_items - 1, 1)  # исключаем PAD

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"recall@{k}"] = total_recall[k] / denom
        metrics[f"ndcg@{k}"] = total_ndcg[k] / denom
        metrics[f"mrr@{k}"] = total_mrr[k] / denom
        metrics[f"coverage@{k}"] = len(covered_items_by_k[k]) / num_items_denom
        if item_freq:
            metrics[f"tail_coverage@{k}"] = len(tail_covered_by_k[k]) / max(
                len(tail_items), 1
            )
            metrics[f"head_coverage@{k}"] = len(head_covered_by_k[k]) / max(
                len(head_items), 1
            )
            for s in ("head", "mid", "tail"):
                cnt = stratum_counts[s]
                metrics[f"recall@{k}_{s}"] = total_hits_by_stratum[k][s] / max(cnt, 1)
                metrics[f"ndcg@{k}_{s}"] = total_ndcg_sum_by_stratum[k][s] / max(cnt, 1)

    metrics["entropy"] = total_entropy / denom

    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Обучить SASRec next-item с in-batch negatives и опциональной logQ"
    )
    parser.add_argument(
        "--train-parquet",
        type=str,
        required=True,
        help="Путь к parquet с обучающими сессиями",
    )
    parser.add_argument(
        "--val-parquet",
        type=str,
        required=True,
        help="Путь к parquet с валидационными сессиями",
    )
    parser.add_argument(
        "--logQ",
        action="store_true",
        help="Если задан — использовать logQ коррекцию при обучении",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Число эпох")
    parser.add_argument("--batch-size", type=int, default=256, help="Размер батча")
    parser.add_argument(
        "--max-seq-len", type=int, default=50, help="Макс длина последовательности"
    )
    parser.add_argument("--embed-dim", type=int, default=64, help="Размер эмбеддинга")
    parser.add_argument(
        "--num-blocks", type=int, default=2, help="Число блоков трансформера"
    )
    parser.add_argument("--num-heads", type=int, default=1, help="Число голов внимания")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Устройство (cpu или cuda)"
    )
    parser.add_argument(
        "--tail-quantile", type=float, default=0.5, help="Квантиль для хвоста"
    )
    parser.add_argument("--seed", type=int, default=42, help="Случайное зерно")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=_default_data_dir(),
        help="Базовая директория с данными (по умолчанию ./data; в Kaggle — /kaggle/working/data)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=_default_log_dir(),
        help="Директория для логов (по умолчанию ./logs; в Kaggle — /kaggle/working/logs)",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default="yandex/yambda",
        help="Hugging Face репозиторий с данными (по умолчанию yandex/yambda)",
    )
    parser.add_argument(
        "--hf-filename",
        type=str,
        default="flat/50m/listens.parquet",
        help="Имя файла в HF репозитории (по умолчанию flat/50m/listens.parquet)",
    )
    parser.add_argument(
        "--hf-download-dir",
        type=str,
        default="./data/raw",
        help="Директория для скачивания данных HF (по умолчанию ./data/raw)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Настройка логирования в указанную директорию
    log_dir = Path(args.log_dir)
    setup_logging(log_dir)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(
        "Начало обучения: run_id=%s, logQ=%s, epochs=%d, batch_size=%d, "
        "max_seq_len=%d, embed_dim=%d, num_blocks=%d, num_heads=%d, dropout=%.2f, "
        "lr=%.6f, device=%s",
        run_id,
        args.logQ,
        args.epochs,
        args.batch_size,
        args.max_seq_len,
        args.embed_dim,
        args.num_blocks,
        args.num_heads,
        args.dropout,
        args.lr,
        args.device,
    )

    # Условное скачивание данных, если обучающая/валидационная выборка отсутствует
    train_path = Path(args.train_parquet)
    val_path = Path(args.val_parquet)
    if not (train_path.exists() and val_path.exists()):
        logger.info(
            "Обучающая или валидационная выборка не найдена, скачиваем данные из Hugging Face: %s/%s",
            args.hf_repo_id,
            args.hf_filename,
        )
        hf_config = HfConfig(
            repo_id=args.hf_repo_id,
            filename=args.hf_filename,
            download_dir=args.hf_download_dir,
        )
        loader = HfYambdaLoader(hf_config)
        loader.download()
        logger.info("Данные скачаны в %s", args.hf_download_dir)
        if not (train_path.exists() and val_path.exists()):
            raise FileNotFoundError(
                f"После скачивания файлы не найдены: {train_path}, {val_path}. "
                "Убедитесь, что после скачивания данные разделены на train и val."
            )

    # Загружаем данные
    logger.info("Загрузка обучающих данных из %s", args.train_parquet)
    train_df = pl.read_parquet(args.train_parquet)
    logger.info("Загрузка валидационных данных из %s", args.val_parquet)
    val_df = pl.read_parquet(args.val_parquet)

    # Токенизация: строим vocab на train, применяем к train и val
    logger.info("Токенизация айтемов...")
    tokenizer = Tokenizer(item_column="item_id")
    tokenizer.fit(train_df)
    train_tokenized = tokenizer.transform(train_df)
    val_tokenized = tokenizer.transform(val_df)

    vocab_size = tokenizer.vocab_size
    num_items = vocab_size

    logger.info("Vocab size: %d (PAD=0, UNK=1, реальные айтемы с 2)", vocab_size)

    # Сохраняем токенизированные данные
    processed_dir = Path(args.data_dir) / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_tokenized_path = processed_dir / "train_tokenized_latest.parquet"
    val_tokenized_path = processed_dir / "val_tokenized_latest.parquet"
    train_tokenized.write_parquet(str(train_tokenized_path))
    val_tokenized.write_parquet(str(val_tokenized_path))
    logger.info(
        "Токенизированные данные сохранены: %s (%d строк), %s (%d строк)",
        train_tokenized_path,
        train_tokenized.height,
        val_tokenized_path,
        val_tokenized.height,
    )

    # Частоты айтемов (в терминах token_id):
    # нужны для logQ коррекции при обучении и для стратификации метрик при валидации
    raw_item_freq = (
        train_df.group_by("item_id")
        .agg(pl.len().alias("cnt"))
        .with_columns((pl.col("cnt") / train_df.height).alias("freq"))
    )
    item_freq: dict[int, float] = {}
    for row in raw_item_freq.iter_rows(named=True):
        item_id = int(row["item_id"])
        token_id = tokenizer.item_to_token.get(item_id)
        if token_id is not None:
            item_freq[token_id] = float(row["freq"])

    # Даталоадеры
    bf = NextItemBatchFormation(
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_workers=0,
        shuffle=True,
    )
    train_loader = bf.create_dataloader(
        parquet_path=str(train_tokenized_path),
        session_column="session_id",
        user_column="uid",
        item_column="item_id",
        sort_column="timestamp",
    )
    val_loader = bf.create_dataloader(
        parquet_path=str(val_tokenized_path),
        session_column="session_id",
        user_column="uid",
        item_column="item_id",
        sort_column="timestamp",
        shuffle=False,
    )

    # Модель
    model = SASRec(
        num_items=num_items,
        max_seq_len=args.max_seq_len,
        embed_dim=args.embed_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Лог файл для метрик
    metrics_jsonl = log_dir / "metrics.jsonl"

    ks = (10, 50, 100)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            logQ=args.logQ,
            item_freq=item_freq,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            num_items=num_items,
            item_freq=item_freq,
            tail_quantile=args.tail_quantile,
            ks=ks,
        )

        # Формируем строку лога: основные метрики + по K + стратифицированные
        parts = [
            f"Epoch {epoch}/{args.epochs}",
            f"train_loss={train_loss:.4f}",
            f"logQ={args.logQ}",
        ]
        for k in ks:
            parts.append(f"val_recall@{k}={val_metrics[f'recall@{k}']:.4f}")
            parts.append(f"val_ndcg@{k}={val_metrics[f'ndcg@{k}']:.4f}")
            parts.append(f"val_mrr@{k}={val_metrics[f'mrr@{k}']:.4f}")
            parts.append(f"val_coverage@{k}={val_metrics[f'coverage@{k}']:.4f}")
            if item_freq:
                parts.append(
                    f"val_tail_coverage@{k}={val_metrics[f'tail_coverage@{k}']:.4f}"
                )
                parts.append(
                    f"val_head_coverage@{k}={val_metrics[f'head_coverage@{k}']:.4f}"
                )
                for s in ("head", "mid", "tail"):
                    parts.append(
                        f"val_recall@{k}_{s}={val_metrics[f'recall@{k}_{s}']:.4f}"
                    )
                    parts.append(f"val_ndcg@{k}_{s}={val_metrics[f'ndcg@{k}_{s}']:.4f}")
        parts.append(f"val_entropy={val_metrics['entropy']:.4f}")
        log_line = " | ".join(parts)
        logger.info(log_line)

        # Запись в metrics.jsonl
        record = {
            "epoch": epoch,
            "run_id": run_id,
            "logQ": args.logQ,
            "train_loss": train_loss,
            **val_metrics,
        }
        with open(metrics_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Обучение завершено: run_id=%s, logQ=%s", run_id, args.logQ)


if __name__ == "__main__":
    main()
