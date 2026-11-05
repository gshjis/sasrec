"""Обучение SASRec: in-batch negatives, plain / standard logQ / improved logQ."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_extractor.config import HfConfig
from data_extractor.hf_loader import HfYambdaLoader
from data_preprocessing.tokenizer import Tokenizer
from models.sasrec.batch_formation import NextItemBatchFormation
from models.sasrec.metrics import MetricsAggregator
from models.sasrec.model import SASRec

KS = (10, 50, 100)
HF_REPO = "yandex/yambda"
HF_FILE = "flat/50m/listens.parquet"

log = logging.getLogger("train_model")


def train_epoch(
    model: SASRec,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    item_freq: dict[int, float] | None,
    logq_mode: str = "plain",
    temperature: float = 1.0,
) -> float:
    """Одна эпоха обучения.

    logq_mode:
      plain      — обычные in-batch negatives, cross-entropy по диагонали
      standard   — стандартный logQ: −log Q вычитается из логитов целей
      improved   — улучшенный logQ (Khrylchenko et al., RecSys'25):
                   позитив не в знаменателе, вес w_up = sg(1 − P(p|u)),
                   только негативы в знаменателе
    """
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


def main() -> None:
    p = argparse.ArgumentParser(
        description="SASRec next-item: plain / standard logQ / improved logQ"
    )
    p.add_argument("--train-parquet", required=True)
    p.add_argument("--val-parquet", required=True)
    p.add_argument(
        "--logq-mode",
        choices=["plain", "standard", "improved"],
        default="plain",
        help="Режим обучения: plain (без коррекции), standard (logQ), improved (улучшенный logQ)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Температура для scaled softmax",
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-seq-len", type=int, default=50)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--log-dir", default="./logs")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "train.log", encoding="utf-8"),
        ],
    )
    torch.manual_seed(42)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    log.info(
        "device=%s, logq_mode=%s, temperature=%.3f",
        device,
        args.logq_mode,
        args.temperature,
    )

    # Данные: при отсутствии траб/вал файлов скачиваем сырые прослушивания с HF
    if not (Path(args.train_parquet).exists() and Path(args.val_parquet).exists()):
        log.info("Данных нет — скачиваем %s/%s", HF_REPO, HF_FILE)
        HfYambdaLoader(
            HfConfig(
                repo_id=HF_REPO,
                filename=HF_FILE,
                download_dir=str(Path(args.data_dir) / "raw"),
            )
        ).download()
    train_df = pl.read_parquet(args.train_parquet)
    val_df = pl.read_parquet(args.val_parquet)

    # Токенизация по train-словарю
    tokenizer = Tokenizer(item_column="item_id").fit(train_df)
    processed = Path(args.data_dir) / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    train_path = processed / "train_tokenized.parquet"
    val_path = processed / "val_tokenized.parquet"
    tokenizer.transform(train_df).write_parquet(train_path)
    tokenizer.transform(val_df).write_parquet(val_path)
    num_items = tokenizer.vocab_size
    log.info("vocab=%d", num_items)

    # Частоты айтемов: нужны для logQ и для парето-метрик при валидации
    freqs = train_df.group_by("item_id").len()
    item_freq = {
        tokenizer.item_to_token[int(r["item_id"])]: r["len"] / train_df.height
        for r in freqs.iter_rows(named=True)
        if int(r["item_id"]) in tokenizer.item_to_token
    }

    bf = NextItemBatchFormation(
        batch_size=args.batch_size, max_seq_len=args.max_seq_len
    )
    loaders = {
        "train": bf.create_dataloader(str(train_path), shuffle=True),
        "val": bf.create_dataloader(str(val_path), shuffle=False),
    }

    model = SASRec(num_items=num_items, max_seq_len=args.max_seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            item_freq if args.logq_mode != "plain" else None,
            args.logq_mode,
            args.temperature,
        )
        metrics = evaluate(model, loaders["val"], device, num_items, item_freq)
        log.info(
            "epoch=%d loss=%.4f %s",
            epoch,
            loss,
            " ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
        )
        with open(log_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "logq_mode": args.logq_mode,
                        "temperature": args.temperature,
                        "loss": loss,
                        **metrics,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
