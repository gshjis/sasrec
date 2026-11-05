"""Готовит тестовую выборку из первых N взаимодействий для пробного прогона.

Шаги:
  1. Берёт первые N строк сырых прослушиваний (по умолчанию 3 млн).
  2. Делит их по временным меткам через gt_split (train / gap1 / val / gap2 / test).
  3. Строит сессии отдельно внутри train и val (build_sessions).
  4. Сохраняет train_3m.parquet и val_3m.parquet с колонкой session_id.

Запуск:
    PYTHONPATH=src python scripts/prepare_3m.py \
        --raw data/raw/yambda_50m_listens.parquet \
        --out data/processed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from data_preprocessing.global_split import GTSSplitConfig, gt_split
from data_preprocessing.session import BuildSessionsConfig, build_sessions


def main() -> None:
    parser = argparse.ArgumentParser(description="Подготовка 3M выборки с сессиями")
    parser.add_argument(
        "--raw",
        type=str,
        default="data/raw/yambda_50m_listens.parquet",
        help="Путь к сырому parquet с прослушиваниями",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/processed",
        help="Директория для сохранения train_3m.parquet и val_3m.parquet",
    )
    parser.add_argument(
        "--interactions", type=int, default=3_000_000, help="Число взаимодействий"
    )
    parser.add_argument(
        "--gap-seconds", type=int, default=3600, help="Длина буфера между сплитами"
    )
    args = parser.parse_args()

    print(f"Чтение {args.raw} ...")
    # TODO: события в сыром файле НЕ отсортированы по времени — перед отсечением
    # первых N взаимодействий нужна сортировка по timestamp, иначе выборка
    # представляет весь период, а не начало истории.
    dataset = pl.read_parquet(args.raw).head(args.interactions)
    print(f"Взято взаимодействий: {dataset.height:,}")

    ts = dataset["timestamp"]
    train_end = int(ts.quantile(0.70))  # type: ignore[reportArgumentType]
    val_end = int(ts.quantile(0.85))  # type: ignore[reportArgumentType]

    split_cfg = GTSSplitConfig(
        train_end=train_end,
        gap1_end=train_end + args.gap_seconds,
        val_end=val_end,
        gap2_end=val_end + args.gap_seconds,
    )
    train, gap1, val, gap2, test = gt_split(dataset, split_cfg)
    print(
        "Сплит по времени: "
        f"train={train.height:,}, gap1={gap1.height:,}, val={val.height:,}, "
        f"gap2={gap2.height:,}, test={test.height:,}"
    )

    session_cfg = BuildSessionsConfig()
    train_sessions = build_sessions(train, session_cfg)
    val_sessions = build_sessions(val, session_cfg)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train_3m.parquet"
    val_path = out_dir / "val_3m.parquet"
    train_sessions.write_parquet(train_path)
    val_sessions.write_parquet(val_path)

    train_n_sessions = train_sessions.group_by(["uid", "session_id"]).len().height
    val_n_sessions = val_sessions.group_by(["uid", "session_id"]).len().height
    print(
        f"train: {train_sessions.height:,} строк, {train_n_sessions:,} сессий -> {train_path}"
    )
    print(
        f"val:   {val_sessions.height:,} строк, {val_n_sessions:,} сессий -> {val_path}"
    )


if __name__ == "__main__":
    main()
