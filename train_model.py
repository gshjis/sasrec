"""Обучение SASRec: in-batch negatives, plain / standard logQ / improved logQ.

Рефакторинг (Этап Б):
  * Конфигурация через --config (YAML/JSON) с приоритетом CLI-аргументов.
  * seed_everything(seed): torch, numpy, random, PYTHONHASHSEED, DataLoader worker_init_fn.
  * Чекпоинты best.pt / last.pt через CheckpointManager (мониторинг val_loss, min).
  * run_meta.json после завершения (коммит, ветка, версии, сиды, аргументы, метрики, статус).
  * Единый log-dir: <LOG_DIR or RUNS_DIR>/<run_name>/ с train.log, metrics.jsonl, run_meta.json, чекпоинтами.
  * Единая температура для всех трёх режимов logQ (справедливое сравнение).
  * Аргументы --lr, --seed, --test-parquet; оценка на test внутри.
  * try/except вокруг основного блока с exit code 0/1.
  * Абсолютные пути через paths.py (независимость от CWD).
  * Скачивание HF вынесено из main() в отдельную функцию с retry.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- Этап А модули (абсолютные пути, независимость от CWD) ---
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from checkpoint import (
    CheckpointManager,
    load_checkpoint,
)
from config import (
    ExperimentConfig,
    LogQMode,
)
from data_extractor.config import HfConfig
from data_extractor.hf_loader import HfYambdaLoader
from data_preprocessing.tokenizer import Tokenizer
from models.sasrec.batch_formation import NextItemBatchFormation
from models.sasrec.metrics import MetricsAggregator
from models.sasrec.model import SASRec
from paths import BASE_DIR, DATA_DIR, ensure_dir
from run_meta import (
    RunMeta,
    new_run_meta,
    save_run_meta,
)

KS = (10, 50, 100)
DEFAULT_HF_REPO = "yandex/yambda"
DEFAULT_HF_FILE = "flat/50m/listens.parquet"

log = logging.getLogger("train_model")


# =====================================================================
# Утилиты воспроизводимости
# =====================================================================


def seed_everything(seed: int) -> None:
    """Применить сид ко всем источникам случайности."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """init_fn для DataLoader workers (распространяет базовый сид)."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =====================================================================
# Конфигурация
# =====================================================================


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    """Загрузить конфиг из YAML или JSON."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]

            return yaml.safe_load(text)
        except ImportError:
            raise ImportError(
                "PyYAML не установлен. Установите: pip install pyyaml "
                "или используйте JSON-конфиг."
            )
    return json.loads(text)


def load_config(
    config_path: Path | None, cli_overrides: dict[str, Any]
) -> ExperimentConfig:
    """Загрузить ExperimentConfig из --config, затем наложить CLI-переопределения.

    CLI-переопределения имеют приоритет над конфигурационным файлом.
    """
    cfg = ExperimentConfig.full()  # базовый preset

    if config_path is not None:
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Конфиг не найден: {cfg_path}")
        raw = _load_yaml_or_json(cfg_path)
        cfg = ExperimentConfig.from_dict(raw)

    # Точечное переопределение полей из CLI (ExperimentConfig frozen -> replace)
    if cli_overrides.get("lr") is not None:
        cfg = replace(
            cfg,
            model=replace(cfg.model, lr=cli_overrides["lr"]),
        )
    if cli_overrides.get("seed") is not None:
        cfg = replace(
            cfg,
            train=replace(cfg.train, seed=cli_overrides["seed"]),
            data=replace(cfg.data, seed=cli_overrides["seed"]),
        )
    if cli_overrides.get("epochs") is not None:
        cfg = replace(cfg, train=replace(cfg.train, epochs=cli_overrides["epochs"]))
    if cli_overrides.get("batch_size") is not None:
        cfg = replace(
            cfg, train=replace(cfg.train, batch_size=cli_overrides["batch_size"])
        )
    if cli_overrides.get("temperature") is not None:
        cfg = replace(
            cfg,
            train=replace(cfg.train, temperature=cli_overrides["temperature"]),
        )
    if cli_overrides.get("max_seq_len") is not None:
        cfg = replace(
            cfg,
            model=replace(cfg.model, max_seq_len=cli_overrides["max_seq_len"]),
        )
    if cli_overrides.get("device") is not None:
        cfg = replace(cfg, train=replace(cfg.train, device=cli_overrides["device"]))
    if cli_overrides.get("logq_mode") is not None:
        cfg = replace(
            cfg,
            train=replace(cfg.train, logq_mode=cli_overrides["logq_mode"]),
        )
    if cli_overrides.get("weight_decay") is not None:
        cfg = replace(
            cfg,
            model=replace(cfg.model, weight_decay=cli_overrides["weight_decay"]),
        )
    # data.* поля
    if cli_overrides.get("train_parquet") is not None:
        cfg = replace(
            cfg, data=replace(cfg.data, raw_path=cli_overrides["train_parquet"])
        )
    if cli_overrides.get("val_parquet") is not None:
        cfg = replace(
            cfg, data=replace(cfg.data, raw_path=cli_overrides["val_parquet"])
        )

    # run-level поля
    if cli_overrides.get("run_name") is not None:
        cfg = replace(cfg, run_name=cli_overrides["run_name"])
    if cli_overrides.get("mode") is not None:
        cfg = replace(cfg, mode=cli_overrides["mode"])
    if cli_overrides.get("output_dir") is not None:
        cfg = replace(cfg, output_dir=cli_overrides["output_dir"])
    if cli_overrides.get("log_dir") is not None:
        cfg = replace(cfg, log_dir=cli_overrides["log_dir"])
    if cli_overrides.get("with_test") is not None:
        cfg = replace(cfg, with_test=cli_overrides["with_test"])
    if cli_overrides.get("dry_run") is not None:
        cfg = replace(cfg, dry_run=cli_overrides["dry_run"])
    if cli_overrides.get("strict_device") is not None:
        cfg = replace(cfg, strict_device=cli_overrides["strict_device"])

    return cfg


# =====================================================================
# Скачивание данных (вынесено из main, с retry)
# =====================================================================


def download_hf_data_with_retry(
    repo_id: str,
    filename: str,
    download_dir: Path,
    max_retries: int = 3,
    backoff: float = 2.0,
) -> Path:
    """Этап pipeline: скачивание HF с exponential backoff retry."""
    download_dir = ensure_dir(download_dir)
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            log.info(
                "Скачивание %s/%s (попытка %d/%d)...",
                repo_id,
                filename,
                attempt,
                max_retries,
            )
            loader = HfYambdaLoader(
                HfConfig(
                    repo_id=repo_id,
                    filename=filename,
                    download_dir=str(download_dir),
                )
            )
            return Path(loader.download())
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            log.warning(
                "Скачивание не удалось (попытка %d/%d): %s", attempt, max_retries, exc
            )
            if attempt < max_retries:
                time.sleep(backoff**attempt)
    raise RuntimeError(
        f"Не удалось скачать {repo_id}/{filename} после {max_retries} попыток: {last_error}"
    )


# =====================================================================
# Обучение
# =====================================================================


def train_epoch(
    model: SASRec,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    item_freq: dict[int, float] | None,
    logq_mode: LogQMode = "plain",
    temperature: float = 1.0,
) -> float:
    """Одна эпоха обучения. Температура ЕДИНАЯ для всех трёх режимов logQ."""
    model.train()
    total, n_batches = 0.0, 0
    for batch in loader:
        mask = batch["targets"] >= 0
        if not mask.any():
            continue
        input_ids = batch["input_ids"][mask].to(device)
        attention_mask = batch["attention_mask"][mask].to(device)
        targets = batch["targets"][mask].to(device)
        b = len(targets)

        logits = model(input_ids, attention_mask) @ model.item_emb(targets).T
        logits = logits / temperature

        if logq_mode == "plain":
            loss = F.cross_entropy(logits, torch.arange(b, device=device))

        elif logq_mode == "standard":
            assert item_freq is not None
            q = torch.tensor(
                [max(item_freq.get(int(t), 1e-6), 1e-6) for t in targets.tolist()],
                device=device,
            )
            log_q = torch.log(q).unsqueeze(0) / temperature  # [1, B]
            logits = logits - log_q
            loss = F.cross_entropy(logits, torch.arange(b, device=device))

        elif logq_mode == "improved":
            assert item_freq is not None
            q = torch.tensor(
                [max(item_freq.get(int(t), 1e-6), 1e-6) for t in targets.tolist()],
                device=device,
            )
            log_q = torch.log(q)  # [B]

            pos_logits = torch.diag(logits)  # [B] — f(u_i, p_i) / T

            # Знаменатель: только негативы (j != i), с коррекцией −log Q'(d_j)
            neg_logits = logits - log_q.unsqueeze(0)  # [B, B]
            off_diag = ~torch.eye(b, dtype=torch.bool, device=device)
            neg_exp = torch.exp(neg_logits) * off_diag  # ноль на диагонали
            denom = neg_exp.sum(dim=1)  # [B]

            # Вес образца w_up = sg(1 − P_θ(p|u))
            pos_exp = torch.exp(pos_logits)  # [B]
            p_hat = pos_exp / (pos_exp + (1.0 / b) * denom + 1e-12)  # [B]
            w_up = (1.0 - p_hat).detach()  # stop-gradient

            log_prob = pos_logits - torch.log(denom + 1e-12)  # [B]
            loss_per_example = -w_up * log_prob  # [B]
            loss = loss_per_example.mean()

        else:
            raise ValueError(f"неизвестный logq_mode: {logq_mode}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        n_batches += 1
    return total / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: SASRec,
    loader: DataLoader,
    device: torch.device,
    num_items: int,
    item_freq: dict[int, float],
) -> dict[str, float]:
    """Оценка на любом сплите (val / test)."""
    model.eval()
    metrics = MetricsAggregator(num_items, item_freq, KS)
    for batch in loader:
        mask = batch["targets"] >= 0
        if not mask.any():
            continue
        targets = batch["targets"][mask].to(device)
        user_emb = model(
            batch["input_ids"][mask].to(device),
            batch["attention_mask"][mask].to(device),
        )
        logits = model.score_all_items(user_emb)
        logits[:, 0] = float("-inf")  # PAD не рекомендуем
        metrics.update(logits, targets)
    return metrics.compute()


def build_loaders(
    cfg: ExperimentConfig,
    tokenizer: Tokenizer,
    item_freq: dict[int, float],
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame | None,
) -> dict[str, DataLoader]:
    """Создать DataLoader'ы train/val (+ test при наличии) с worker_init_fn."""
    processed = ensure_dir(Path(cfg.data.processed_dir))
    train_path = processed / "train_tokenized.parquet"
    val_path = processed / "val_tokenized.parquet"
    tokenizer.transform(train_df).write_parquet(train_path)
    tokenizer.transform(val_df).write_parquet(val_path)

    test_path: Path | None = None
    if test_df is not None:
        test_path = processed / "test_tokenized.parquet"
        tokenizer.transform(test_df).write_parquet(test_path)

    bf = NextItemBatchFormation(
        batch_size=cfg.train.batch_size, max_seq_len=cfg.model.max_seq_len
    )
    loaders: dict[str, DataLoader] = {
        "train": bf.create_dataloader(
            str(train_path), shuffle=True, worker_init_fn=worker_init_fn
        ),
        "val": bf.create_dataloader(
            str(val_path), shuffle=False, worker_init_fn=worker_init_fn
        ),
    }
    if test_path is not None:
        loaders["test"] = bf.create_dataloader(
            str(test_path), shuffle=False, worker_init_fn=worker_init_fn
        )
    return loaders


# =====================================================================
# Точка входа
# =====================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SASRec next-item: plain / standard logQ / improved logQ"
    )
    p.add_argument("--config", type=str, default=None, help="Путь к YAML/JSON конфигу")
    p.add_argument(
        "--run-name", type=str, default=None, help="Имя запуска (имя папки в runs/)"
    )
    p.add_argument(
        "--mode",
        choices=["smoke", "full", "ablation"],
        default=None,
        help="Режим прогона (задаёт preset гиперпараметров)",
    )
    p.add_argument(
        "--train-parquet", type=str, default=None, help="Путь к train parquet"
    )
    p.add_argument(
        "--val-parquet", type=str, default=None, help="Путь к val parquet"
    )
    p.add_argument(
        "--test-parquet", type=str, default=None, help="Путь к test parquet (оценка)"
    )
    p.add_argument(
        "--data-dir", type=str, default=None, help="Базовая директория с данными"
    )
    p.add_argument(
        "--log-dir", type=str, default=None, help="Базовая директория логов"
    )
    p.add_argument(
        "--output-dir", type=str, default=None, help="Директория результатов (runs/)"
    )
    p.add_argument(
        "--epochs", type=int, default=None, help="Количество эпох"
    )
    p.add_argument(
        "--batch-size", type=int, default=None, help="Размер батча"
    )
    p.add_argument("--lr", type=float, default=None, help="Learning rate (Adam)")
    p.add_argument("--seed", type=int, default=None, help="Сид воспроизводимости")
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Температура scaled softmax (единая для всех режимов logQ)",
    )
    p.add_argument(
        "--logq-mode",
        choices=["plain", "standard", "improved"],
        default=None,
        help="Режим обучения",
    )
    p.add_argument(
        "--weight-decay", type=float, default=None, help="Weight decay оптимизатора"
    )
    p.add_argument(
        "--max-seq-len", type=int, default=None, help="Максимальная длина последовательности"
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Устройство: auto | cpu | cuda",
    )
    p.add_argument(
        "--with-test",
        action="store_true",
        default=None,
        help="Включить оценку на test",
    )
    p.add_argument(
        "--no-test",
        action="store_false",
        dest="with_test",
        default=None,
        help="Выключить оценку на test",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Прогон без обучения (валидация конвейера)",
    )
    p.add_argument(
        "--strict-device",
        action="store_true",
        default=None,
        help="Ошибка, если запрошено cuda но GPU недоступно",
    )
    return p.parse_args()


def _resolve_device(
    requested: str, strict: bool = False
) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            if strict:
                raise RuntimeError("Запрошен cuda, но GPU недоступен")
            log.warning("CUDA недоступен, fallback на cpu")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> int:
    """Точка входа с try/except, run_meta, чекпоинтами, оценкой на test."""
    meta: RunMeta | None = None
    log_dir: Path | None = None
    try:
        args = parse_args()
        cfg = load_config(Path(args.config) if args.config else None, vars(args))

        # Применяем preset режима, если указан
        if args.mode is not None:
            if args.mode == "smoke":
                cfg = ExperimentConfig.smoke()
            elif args.mode == "full":
                cfg = ExperimentConfig.full()
            elif args.mode == "ablation":
                cfg = ExperimentConfig.ablation()
            # CLI-переопределения уже наложены в load_config, но preset может их сбросить;
            # применим ещё раз после preset
            cfg = load_config(
                Path(args.config) if args.config else None, vars(args)
            )

        seed_everything(cfg.train.seed)

        run_name = args.run_name or cfg.run_name or f"run_{int(time.time())}"
        run_name = f"{run_name}_{cfg.mode}"
        log_dir = ensure_dir(Path(cfg.log_dir) / run_name)

        # Настройка логирования: файл + stdout
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_dir / "train.log", encoding="utf-8"),
            ],
        )

        device = _resolve_device(cfg.train.device, cfg.strict_device)
        log.info(
            "device=%s, logq_mode=%s, temperature=%.3f, seed=%d",
            device,
            cfg.train.logq_mode,
            cfg.train.temperature,
            cfg.train.seed,
        )

        # RunMeta
        meta = new_run_meta(
            run_name=run_name,
            mode=cfg.mode,
            seed=cfg.train.seed,
            args={k: v for k, v in vars(args).items() if v is not None},
            repo_dir=BASE_DIR,
        )

        # Данные
        train_parquet = Path(args.train_parquet) if args.train_parquet else None
        val_parquet = Path(args.val_parquet) if args.val_parquet else None
        test_parquet = Path(args.test_parquet) if args.test_parquet else None

        # Если паркеты не указаны — пробуем из processed dir
        processed = Path(cfg.data.processed_dir)
        if train_parquet is None:
            train_parquet = processed / "train_3m.parquet"
        if val_parquet is None:
            val_parquet = processed / "val_3m.parquet"
        if test_parquet is None and cfg.with_test:
            test_parquet = processed / "test_3m.parquet"

        for p in [train_parquet, val_parquet]:
            if p is not None and not p.exists():
                raise FileNotFoundError(f"Не найден parquet: {p}")

        if cfg.with_test and test_parquet is not None and not test_parquet.exists():
            log.warning("Test parquet не найден: %s, оценка на test пропущена", test_parquet)
            test_parquet = None

        # HF download только если нет train/val
        if not train_parquet.exists() or not val_parquet.exists():
            log.info(
                "Данных нет — скачиваем %s/%s", DEFAULT_HF_REPO, DEFAULT_HF_FILE
            )
            raw_path = download_hf_data_with_retry(
                DEFAULT_HF_REPO,
                DEFAULT_HF_FILE,
                Path(cfg.data.raw_path).parent if cfg.data.raw_path else DATA_DIR / "raw",
            )
            log.info("Скачано: %s", raw_path)
            # После скачивания нужно запустить prepare_3m.py — но это вынесено в pipeline.
            # Здесь просто сообщаем, что данные нужно подготовить отдельно.
            raise FileNotFoundError(
                f"Требуется запустить scripts/prepare_3m.py для подготовки train/val/test. "
                f"Сырой файл скачан: {raw_path}"
            )

        log.info("Чтение train: %s", train_parquet)
        train_df = pl.read_parquet(str(train_parquet))
        val_df = pl.read_parquet(str(val_parquet))
        test_df = pl.read_parquet(str(test_parquet)) if test_parquet else None

        log.info(
            "Размеры: train=%d, val=%d, test=%s",
            train_df.height,
            val_df.height,
            test_df.height if test_df is not None else "нет",
        )

        # Токенизация по train-словарю
        tokenizer = Tokenizer(item_column="item_id").fit(train_df)
        num_items = tokenizer.vocab_size
        log.info("vocab=%d", num_items)

        # Частоты айтемов: нужны для logQ и парето-метрик
        freqs = train_df.group_by("item_id").len()
        item_freq = {
            tokenizer.item_to_token[int(r["item_id"])]: r["len"] / train_df.height
            for r in freqs.iter_rows(named=True)
            if int(r["item_id"]) in tokenizer.item_to_token
        }

        # DataLoader'ы
        loaders = build_loaders(
            cfg, tokenizer, item_freq, train_df, val_df, test_df
        )

        # Модель + оптимизатор
        model = SASRec(
            num_items=num_items,
            max_seq_len=cfg.model.max_seq_len,
            embed_dim=cfg.model.embed_dim,
            num_blocks=cfg.model.num_blocks,
            num_heads=cfg.model.num_heads,
            dropout=cfg.model.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.model.lr, weight_decay=cfg.model.weight_decay
        )

        # CheckpointManager
        ckpt_dir = ensure_dir(log_dir / "checkpoints")
        ckpt_mgr = CheckpointManager(
            dir_path=ckpt_dir,
            keep_last=3,
            monitor="val_loss",
            mode="min",
        )

        # Цикл эпох
        for epoch in range(1, cfg.train.epochs + 1):
            loss = train_epoch(
                model,
                loaders["train"],
                optimizer,
                device,
                item_freq if cfg.train.logq_mode != "plain" else None,
                cfg.train.logq_mode,
                cfg.train.temperature,
            )
            val_metrics = evaluate(
                model, loaders["val"], device, num_items, item_freq
            )
            val_loss = val_metrics.get("loss", loss)

            log.info(
                "epoch=%d loss=%.4f val_loss=%.4f %s",
                epoch,
                loss,
                val_loss,
                " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items() if k != "loss"),
            )

            # Запись в metrics.jsonl
            metrics_row = {
                "epoch": epoch,
                "run_name": run_name,
                "logq_mode": cfg.train.logq_mode,
                "temperature": cfg.train.temperature,
                "loss": loss,
                "val_loss": val_loss,
                **val_metrics,
            }
            with open(log_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics_row) + "\n")

            # Чекпоинты
            ckpt_mgr.save(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                metrics={"val_loss": val_loss, **val_metrics},
                config=cfg.to_dict(),
            )

        # Оценка на test
        test_metrics: dict[str, float] = {}
        if cfg.with_test and "test" in loaders:
            # Загружаем best.pt для финальной оценки
            best_path = ckpt_dir / "best.pt"
            if best_path.exists():
                log.info("Загрузка best.pt для оценки на test")
                _, _ = load_checkpoint(best_path, model, optimizer, device=str(device))
            test_metrics = evaluate(
                model, loaders["test"], device, num_items, item_freq
            )
            log.info(
                "TEST %s",
                " ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()),
            )

        # Завершение run_meta
        meta.status = "success"
        meta.finished_at = datetime.now(UTC).isoformat()
        meta.duration_sec = (
            datetime.fromisoformat(meta.finished_at)
            - datetime.fromisoformat(meta.started_at)
        ).total_seconds()
        meta.metrics = {
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "best_checkpoint": str(ckpt_dir / "best.pt"),
            "last_checkpoint": str(ckpt_dir / "last.pt"),
        }
        meta.artifact_paths = [
            str(log_dir / "train.log"),
            str(log_dir / "metrics.jsonl"),
            str(ckpt_dir / "best.pt"),
            str(ckpt_dir / "last.pt"),
        ]
        save_run_meta(log_dir / "run_meta.json", meta)

        log.info(
            "[DONE] Запуск %s завершён за %.1f сек. Результаты: %s",
            run_name,
            meta.duration_sec,
            log_dir,
        )
        return 0

    except Exception as exc:
        log.exception("Обучение завершилось с ошибкой")
        if meta is not None:
            meta.status = "failed"
            meta.error = str(exc)
            meta.finished_at = datetime.now(UTC).isoformat()
            if log_dir is not None:
                save_run_meta(log_dir / "run_meta.json", meta)
        return 1


if __name__ == "__main__":
    sys.exit(main())
