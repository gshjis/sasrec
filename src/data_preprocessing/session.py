from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class BuildSessionsConfig:
    """Конфигурация построения сессий по правому хвосту логнормального распределения gap."""

    gap_std_threshold: float = 2.0  # число стандартных отклонений в лог-пространстве
    min_played_ratio_pct: float = (
        50.0  # фильтр прослушиваний (оставляем >= этого значения)
    )
    min_session_len: int = 10  # минимум событий в сессии
    remove_time_outliers: bool = (
        True  # удалять сессии длиннее верхнего уса боксплота по времени
    )
    time_outlier_multiplier: float = (
        1.5  # множитель IQR для верхнего уса (1.5 = стандартный боксплот)
    )
    uid_column: str = "uid"
    timestamp_column: str = "timestamp"
    played_ratio_column: str = "played_ratio_pct"
    session_column: str = "session_id"


def build_sessions(
    dataset: pl.DataFrame,
    listen_plus: bool = True,
    config: BuildSessionsConfig | None = None,
) -> pl.DataFrame:

    config = config or BuildSessionsConfig()
    uid = config.uid_column
    ts = config.timestamp_column

    df = dataset.clone()

    if listen_plus:
        df = df.filter(
            pl.col(config.played_ratio_column) >= config.min_played_ratio_pct
        )
    df = df.sort([uid, ts])
    df = df.with_columns(pl.col(ts).diff().over(uid).alias("gap_seconds"))
    df = df.with_columns(
        pl.when(pl.col("gap_seconds") > 0)
        .then(pl.col("gap_seconds").log())
        .otherwise(None)
        .alias("log_gap")
    )
    user_stats = df.group_by(uid).agg(
        [
            pl.col("log_gap").mean().alias("mean_log_gap"),
            pl.col("log_gap").std().alias("std_log_gap"),
        ]
    )
    df = df.join(user_stats, on=uid, how="left")
    df = df.with_columns(
        (pl.col("mean_log_gap") + config.gap_std_threshold * pl.col("std_log_gap"))
        .exp()
        .alias("gap_threshold")
    )
    df = df.with_columns(
        (
            pl.col("gap_seconds").is_null()
            | (pl.col("gap_seconds") > pl.col("gap_threshold"))
        )
        .fill_null(False)
        .alias("is_new_session")
    )
    df = df.with_columns(
        pl.col("is_new_session")
        .cast(pl.Int32)
        .cum_sum()
        .over(uid)
        .alias(config.session_column)
    )
    df = df.drop(
        [
            "gap_seconds",
            "log_gap",
            "mean_log_gap",
            "std_log_gap",
            "gap_threshold",
            "is_new_session",
        ]
    )
    if config.min_session_len > 1:
        session_lens = df.group_by([uid, config.session_column]).agg(
            pl.len().alias("session_len")
        )
        valid_sessions = session_lens.filter(
            pl.col("session_len") >= config.min_session_len
        )
        df = df.join(valid_sessions, on=[uid, config.session_column], how="inner")
        if "session_len" in df.columns:
            df = df.drop("session_len")

    if config.remove_time_outliers:
        session_durations = (
            df.group_by([uid, config.session_column])
            .agg(
                [
                    pl.col(ts).min().alias("t_start"),
                    pl.col(ts).max().alias("t_end"),
                ]
            )
            .with_columns(
                (pl.col("t_end") - pl.col("t_start")).alias("duration_sec")  # type: ignore[operator]
            )
        )

        q1 = session_durations["duration_sec"].quantile(0.25)
        q3 = session_durations["duration_sec"].quantile(0.75)
        iqr = q3 - q1  # type: ignore[operator]
        upper_whisker = q3 + config.time_outlier_multiplier * iqr  # type: ignore[operator]

        valid_durations = session_durations.filter(
            pl.col("duration_sec") <= upper_whisker
        ).select([uid, config.session_column])

        df = df.join(valid_durations, on=[uid, config.session_column], how="inner")

    return df.sort([uid, config.session_column, ts])
