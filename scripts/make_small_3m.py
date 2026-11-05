"""Формирует тестовую выборку из первых 3 млн взаимодействий.

Разделяет на train (70%), val (15%), test (15%) без gap-буферов —
для быстрого тестового прогона на Kaggle ядре.
"""

from __future__ import annotations

import polars as pl


def main() -> None:
    src = "data/raw/yambda_50m_listens.parquet"
    df = pl.read_parquet(src)
    df3 = df.head(3_000_000)
    n = df3.height
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    df3[:n_train].write_parquet("data/processed/train_3m.parquet")
    df3[n_train : n_train + n_val].write_parquet("data/processed/val_3m.parquet")
    df3[n_train + n_val :].write_parquet("data/processed/test_3m.parquet")
    print(f"Train: {n_train}, Val: {n_val}, Test: {n - n_train - n_val}")


if __name__ == "__main__":
    main()
