"""Готовит выборку с сессиями для обучения и валидации.

Шаги:
  1. Берёт первые N взаимодействий по времени из сырых прослушиваний.
  2. Делит по временным меткам через gt_split (train / gap1 / val / gap2 / test).
  3. Строит сессии отдельно внутри train, val и test (build_sessions).
  4. Сохраняет train_3m.parquet, val_3m.parquet, test_3m.parquet с колонкой session_id.
  5. Пишет split_meta.json и fingerprint.json для воспроизводимости.

Запуск:
    PYTHONPATH=src python scripts/prepare_3m.py \
        --raw data/raw/yambda_50m_listens.parquet \
        --out data/processed \
        --interactions 3_000_000 \
        --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import polars as pl

# Абсолютные пути, независимые от CWD
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_preprocessing.global_split import GTSSplitConfig, gt_split
from data_preprocessing.session import BuildSessionsConfig, build_sessions
from paths import DATA_DIR, PROCESSED_DIR, ensure_dir

log = logging.getLogger("prepare_3m")


def _fingerprint(config: dict, raw_path: Path) -> str:
    """Считать SHA-256 fingerprint конфига + сырых данных для контроля целостности."""
    h = hashlib.sha256()
    h.update(json.dumps(config, sort_keys=True).encode())
    if raw_path.exists():
        with raw_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("prepare_3m.log", encoding="utf-8"),
        ],
    )

    parser = argparse.ArgumentParser(description="Подготовка выборки с сессиями")
    parser.add_argument(
        "--raw",
        type=str,
        default=str(DATA_DIR / "raw" / "yambda_50m_listens.parquet"),
        help="Путь к сырому parquet с прослушиваниями",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(PROCESSED_DIR),
        help="Директория для сохранения train/val/test parquet и meta",
    )
    parser.add_argument(
        "--interactions", type=int, default=3_000_000,
        help="Число взаимодействий (0 = все доступные)",
    )
    parser.add_argument(
        "--with-test", action="store_true", default=True,
        help="Сохранять test_3m.parquet и строить для него сессии",
    )
    parser.add_argument(
        "--no-test", action="store_false", dest="with_test",
        help="Не сохранять test_3m.parquet (быстрый smoke-прогон)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Сид для воспроизводимости",
    )
    parser.add_argument(
        "--train-share", type=float, default=0.70, help="Доля train (0..1)",
    )
    parser.add_argument(
        "--val-share", type=float, default=0.15, help="Доля val (0..1)",
    )
    parser.add_argument(
        "--test-share", type=float, default=0.15, help="Доля test (0..1)",
    )
    parser.add_argument(
        "--gap-seconds", type=int, default=3600,
        help="Длина буфера между сплитами (секунды)",
    )
    args = parser.parse_args()

    # Валидация долей
    total = args.train_share + args.val_share + args.test_share
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"train_share + val_share + test_share должно быть 1.0, получено {total}"
        )

    out_dir = ensure_dir(Path(args.out))
    raw_path = Path(args.raw)

    if not raw_path.exists():
        raise FileNotFoundError(f"Сырой файл не найден: {raw_path}")

    log.info("Чтение сырых данных: %s", raw_path)
    raw_df = pl.read_parquet(raw_path)
    log.info("Всего строк в сырых данных: %d", raw_df.height)

    # Отбираем первые N взаимодействий по времени (сортировка перед отбором — исправляет баг #57)
    ts_col = "timestamp"
    if args.interactions > 0 and args.interactions < raw_df.height:
        ts_sorted = raw_df[ts_col].sort()
        threshold = ts_sorted.gather(args.interactions - 1)
        dataset = (
            raw_df.lazy()
            .filter(pl.col(ts_col) <= threshold)
            .sort(ts_col)
            .collect()
        )
        if args.interactions > 0:
            dataset = dataset.head(args.interactions)
    else:
        dataset = raw_df.sort(ts_col)
    log.info("Взято взаимодействий по времени: %d", dataset.height)

    # Вычисляем границы сплита ПО ТОЛЬКО ОБУЧАЮЩЕЙ ЧАСТИ (без утечки будущего)
    # Сначала определяем временной порог train_end как квантиль от всех данных
    # (но без future — мы не знаем будущее на момент обучения, поэтому квантиль считается
    #  по временным меткам до предполагаемого конца train).
    # Для детерминированности используем долю train_share от общего временного диапазона.
    all_ts = dataset[ts_col]
    min_ts = int(all_ts.min())
    max_ts = int(all_ts.max())
    time_span = max_ts - min_ts

    train_end = min_ts + int(time_span * args.train_share)
    val_end = train_end + int(time_span * args.val_share)
    # test_end = max_ts (всё остальное)

    gap1_end = train_end + args.gap_seconds
    gap2_end = val_end + args.gap_seconds

    split_cfg = GTSSplitConfig(
        train_end=train_end,
        gap1_end=gap1_end,
        val_end=val_end,
        gap2_end=gap2_end,
    )
    train, gap1, val, gap2, test = gt_split(dataset, split_cfg)
    log.info(
        "Сплит по времени: train=%d, gap1=%d, val=%d, gap2=%d, test=%d",
        train.height, gap1.height, val.height, gap2.height, test.height,
    )

    # Сессии строятся отдельно внутри каждого сплита
    session_cfg = BuildSessionsConfig(seed=args.seed)

    log.info("Построение сессий train ...")
    train_sessions = build_sessions(train, config=session_cfg)
    log.info("Построение сессий val ...")
    val_sessions = build_sessions(val, config=session_cfg)

    test_sessions = None
    test_n = 0
    if args.with_test:
        log.info("Построение сессий test ...")
        test_sessions = build_sessions(test, config=session_cfg)
        test_n = test_sessions.group_by(["uid", "session_id"]).len().height

    train_n = train_sessions.group_by(["uid", "session_id"]).len().height
    val_n = val_sessions.group_by(["uid", "session_id"]).len().height

    train_path = out_dir / "train_3m.parquet"
    val_path = out_dir / "val_3m.parquet"
    test_path = out_dir / "test_3m.parquet"
    train_sessions.write_parquet(train_path)
    val_sessions.write_parquet(val_path)
    if args.with_test and test_sessions is not None:
        test_sessions.write_parquet(test_path)

    log.info(
        "train: %d строк, %d сессий -> %s",
        train_sessions.height, train_n, train_path,
    )
    log.info(
        "val:   %d строк, %d сессий -> %s",
        val_sessions.height, val_n, val_path,
    )
    if args.with_test:
        log.info(
            "test:  %d строк, %d сессий -> %s",
            test_sessions.height, test_n, test_path,
        )

    # split_meta.json — граница сплита, seed, хэш данных (только счётчики, без полных списков сессий)
    split_meta = {
        "train_end": train_end,
        "gap1_end": gap1_end,
        "val_end": val_end,
        "gap2_end": gap2_end,
        "min_ts": min_ts,
        "max_ts": max_ts,
        "train_share": args.train_share,
        "val_share": args.val_share,
        "test_share": args.test_share,
        "gap_seconds": args.gap_seconds,
        "seed": args.seed,
        "interactions": args.interactions,
        "raw_file": str(raw_path),
        "train_rows": int(train_sessions.height),
        "val_rows": int(val_sessions.height),
        "test_rows": int(test_sessions.height if test_sessions is not None else 0),
        "train_sessions": int(train_n),
        "val_sessions": int(val_n),
        "test_sessions": int(test_n),
    }
    split_meta_path = out_dir / "split_meta.json"
    with split_meta_path.open("w", encoding="utf-8") as f:
        json.dump(split_meta, f, ensure_ascii=False, indent=2)
    log.info("split_meta.json -> %s", split_meta_path)

    # fingerprint.json — хеш конфига + сырых данных
    config_dict = {
        "interactions": args.interactions,
        "train_share": args.train_share,
        "val_share": args.val_share,
        "test_share": args.test_share,
        "gap_seconds": args.gap_seconds,
        "seed": args.seed,
        "raw_file": str(raw_path),
    }
    fingerprint = _fingerprint(config_dict, raw_path)
    fingerprint_path = out_dir / "fingerprint.json"
    with fingerprint_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "fingerprint": fingerprint,
                "config": config_dict,
                "created_at": __import__("datetime").datetime.now().isoformat(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    log.info("fingerprint.json -> %s", fingerprint_path)

    log.info("Готово. Все файлы сохранены в %s", out_dir)


if __name__ == "__main__":
    main()
