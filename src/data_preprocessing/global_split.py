from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class GTSSplitConfig:
    train_end: int
    gap1_end: int
    val_end: int
    gap2_end: int


def gt_split(
    dataset: pl.DataFrame, config: GTSSplitConfig
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    ts = pl.col("timestamp")
    train = dataset.filter(ts < config.train_end)
    gap1 = dataset.filter((ts >= config.train_end) & (ts < config.gap1_end))
    val = dataset.filter((ts >= config.gap1_end) & (ts < config.val_end))
    gap2 = dataset.filter((ts >= config.val_end) & (ts < config.gap2_end))
    test = dataset.filter(ts >= config.gap2_end)
    return train, gap1, val, gap2, test
