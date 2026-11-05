"""Запуск матрицы абляционных исследований.

Единая точка входа для всех абляций: оборачивает prepare_3m.py и train_model.py.
Все комбинации обучаются на одних и тех же train/val/test данных.

Примеры:
    # Smoke-пробная матрица (12 комбинаций, 1 сид, 2 эпохи)
    PYTHONPATH=src python scripts/run_ablations.py \
        --matrix smoke --output-dir ./ablation_runs/smoke --device cuda

    # Полная матрица с multi-seed (n=5)
    PYTHONPATH=src python scripts/run_ablations.py \
        --matrix full --seeds 42,123,2024,777,2026 --n-runs 5 \
        --output-dir ./ablation_runs/full --device cuda

    # Только температура (все режимы × T=0.5/1.0/2.0)
    PYTHONPATH=src python scripts/run_ablations.py \
        --matrix temp-only --output-dir ./ablation_runs/temp --device cuda

    # Собственная матрица из YAML
    PYTHONPATH=src python scripts/run_ablations.py \
        --config configs/ablation.yaml --output-dir ./ablation_runs/custom

    # Драйран — посмотреть план без запуска
    PYTHONPATH=src python scripts/run_ablations.py --matrix full --dry-run

    # Продолжить прерванную серию
    PYTHONPATH=src python scripts/run_ablations.py --matrix full --resume
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import DataConfig
from paths import BASE_DIR, ensure_dir

log = logging.getLogger("run_ablations")

MATRIX_PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "logq_modes": ["plain", "standard", "improved"],
        "temperatures": [1.0],
        "batch_sizes": [256],
        "lrs": [1e-3],
        "epochs": 2,
        "emb_dims": [64],
        "dropouts": [0.5],
        "max_seq_lens": [50],
        "interactions": 100_000,
    },
    "full": {
        "logq_modes": ["plain", "standard", "improved"],
        "temperatures": [1.0],
        "batch_sizes": [256],
        "lrs": [1e-3],
        "epochs": 10,
        "emb_dims": [64],
        "dropouts": [0.5],
        "max_seq_lens": [50],
        "interactions": 0,
    },
    "logq-only": {
        "logq_modes": ["plain", "standard", "improved"],
        "temperatures": [1.0],
        "batch_sizes": [256],
        "lrs": [1e-3],
        "epochs": 5,
        "emb_dims": [64],
        "dropouts": [0.5],
        "max_seq_lens": [50],
        "interactions": 3_000_000,
    },
    "temp-only": {
        "logq_modes": ["plain", "standard", "improved"],
        "temperatures": [0.5, 1.0, 2.0],
        "batch_sizes": [256],
        "lrs": [1e-3],
        "epochs": 5,
        "emb_dims": [64],
        "dropouts": [0.5],
        "max_seq_lens": [50],
        "interactions": 3_000_000,
    },
    "hyper-only": {
        "logq_modes": ["improved"],
        "temperatures": [1.0],
        "batch_sizes": [128, 256, 512],
        "lrs": [1e-4, 1e-3, 5e-3],
        "epochs": 5,
        "emb_dims": [64],
        "dropouts": [0.5],
        "max_seq_lens": [50],
        "interactions": 3_000_000,
    },
    "split-only": {
        "logq_modes": ["improved"],
        "temperatures": [1.0],
        "batch_sizes": [256],
        "lrs": [1e-3],
        "epochs": 5,
        "emb_dims": [64],
        "dropouts": [0.5],
        "max_seq_lens": [50],
        "interactions": 3_000_000,
        "split_variants": [
            {"train_share": 0.70, "val_share": 0.15, "test_share": 0.15},
            {"train_share": 0.60, "val_share": 0.20, "test_share": 0.20},
            {"train_share": 0.80, "val_share": 0.10, "test_share": 0.10},
        ],
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Запуск матрицы абляционных исследований")
    p.add_argument(
        "--matrix",
        choices=list(MATRIX_PRESETS.keys()) + ["custom"],
        default="smoke",
        help="Предустановка матрицы комбинаций",
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="Путь к YAML/JSON конфигу с пользовательской матрицей (переопределяет --matrix)",
    )
    p.add_argument(
        "--data", type=str, default=None,
        help="Путь к уже готовым train/val parquet (если не указан — запускается prepare_3m.py)",
    )
    p.add_argument(
        "--output-dir", type=str, default="./ablation_runs",
        help="Корень каталога результатов",
    )
    p.add_argument(
        "--seeds", type=str, default="42,123,2024",
        help="Список сидов через запятую",
    )
    p.add_argument(
        "--n-runs", type=int, default=1,
        help="Число повторов на конфигурацию",
    )
    p.add_argument(
        "--device", choices=["cuda", "cpu", "auto"], default="auto",
        help="Устройство",
    )
    p.add_argument(
        "--epochs-smoke", type=int, default=2,
        help="Эпохи в smoke-режиме",
    )
    p.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Только напечатать план без запуска",
    )
    p.add_argument(
        "--resume", action="store_true", default=False,
        help="Продолжить с последнего прерванного run_id",
    )
    p.add_argument(
        "--aggregate", action="store_true", default=True,
        help="Собрать таблицу после завершения",
    )
    p.add_argument(
        "--no-aggregate", action="store_false", dest="aggregate",
        help="Не собирать таблицу после завершения",
    )
    return p.parse_args()


def load_matrix_from_config(config_path: str) -> dict[str, Any]:
    """Загрузить матрицу из YAML/JSON конфига."""
    import json as _json
    try:
        import yaml  # type: ignore[import-untyped]
        with open(config_path) as f:
            return yaml.safe_load(f)
    except ImportError:
        with open(config_path) as f:
            return _json.load(f)


def build_combinations(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    """Построить список комбинаций из матрицы."""
    split_variants = matrix.pop("split_variants", None)
    keys = list(matrix.keys())
    values = [matrix[k] if isinstance(matrix[k], list) else [matrix[k]] for k in keys]
    combinations = []
    for combo in itertools.product(*values):
        c = dict(zip(keys, combo))
        if split_variants:
            for sv in split_variants:
                c2 = dict(c)
                c2.update(sv)
                combinations.append(c2)
        else:
            combinations.append(c)
    return combinations


def compute_run_id(combination: dict[str, Any], seed: int, data_hash: str) -> str:
    """Детерминированный run_id из комбинации + сид + хэш данных."""
    payload = {
        "combination": combination,
        "seed": seed,
        "data_hash": data_hash,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:16]


def run_single(
    combination: dict[str, Any],
    seed: int,
    train_parquet: str,
    val_parquet: str,
    test_parquet: str | None,
    output_dir: Path,
    device: str,
    dry_run: bool,
) -> dict[str, Any]:
    """Запуск одной комбинации через train_model.py."""
    run_id = compute_run_id(combination, seed, "")
    run_dir = ensure_dir(output_dir / run_id)

    if dry_run:
        log.info("[DRY-RUN] %s: %s", run_id, combination)
        return {"run_id": run_id, "status": "dry_run", "combination": combination, "seed": seed}

    import subprocess

    cmd = [
        sys.executable, "train_model.py",
        "--run-name", run_id,
        "--train-parquet", train_parquet,
        "--val-parquet", val_parquet,
    ]
    if test_parquet:
        cmd += ["--test-parquet", test_parquet]
    cmd += [
        "--epochs", str(combination.get("epochs", 5)),
        "--batch-size", str(combination.get("batch_size", 256)),
        "--lr", str(combination.get("lr", 1e-3)),
        "--seed", str(seed),
        "--temperature", str(combination.get("temperature", 1.0)),
        "--logq-mode", combination.get("logq_mode", "plain"),
        "--weight-decay", str(combination.get("weight_decay", 0.0)),
        "--max-seq-len", str(combination.get("max_seq_len", 50)),
        "--device", device,
        "--with-test",
        "--output-dir", str(output_dir),
        "--log-dir", str(output_dir),
    ]

    log.info("Запуск %s: logq_mode=%s, T=%s, seed=%s", run_id, combination.get("logq_mode"), combination.get("temperature"), seed)
    result = subprocess.run(cmd, check=False, cwd=BASE_DIR, capture_output=True, text=True)

    status = "success" if result.returncode == 0 else "failed"
    error = None if result.returncode == 0 else result.stderr[-2000:] if result.stderr else "unknown error"

    return {
        "run_id": run_id,
        "status": status,
        "combination": combination,
        "seed": seed,
        "error": error,
        "run_dir": str(run_dir),
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args = parse_args()

    # Загрузка матрицы
    if args.config:
        matrix = load_matrix_from_config(args.config)
    else:
        matrix = dict(MATRIX_PRESETS[args.matrix])
        matrix["epochs"] = args.epochs_smoke if args.matrix == "smoke" else matrix.get("epochs", 5)

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    output_dir = ensure_dir(Path(args.output_dir))

    combinations = build_combinations(dict(matrix))
    log.info("Матрица: %d комбинаций × %d сидов = %d запусков", len(combinations), len(seeds), len(combinations) * len(seeds))

    if args.dry_run:
        for combo in combinations:
            for seed in seeds:
                run_single(combo, seed, "", "", None, output_dir, args.device, dry_run=True)
        print("Dry-run завершён. Используйте без --dry-run для реального запуска.")
        return 0

    # Подготовка данных (если не указаны готовые parquet)
    train_parquet = args.data
    val_parquet = None
    test_parquet = None

    if train_parquet is None:
        import sys as _sys

        from scripts.prepare_3m import main as prepare_main
        old_argv = _sys.argv
        _sys.argv = [
            "prepare_3m.py",
            "--raw", str(DataConfig().raw_path),
            "--out", str(DataConfig().processed_dir),
            "--interactions", str(matrix.get("interactions", 3_000_000)),
            "--seed", str(seeds[0]),
            "--train-share", "0.70",
            "--val-share", "0.15",
            "--test-share", "0.15",
            "--gap-seconds", "3600",
            "--with-test",
        ]
        try:
            prepare_main()
        finally:
            _sys.argv = old_argv
        train_parquet = str(DataConfig().processed_dir / "train_3m.parquet")
        val_parquet = str(DataConfig().processed_dir / "val_3m.parquet")
        test_parquet = str(DataConfig().processed_dir / "test_3m.parquet")

    results = []
    for combo in combinations:
        for seed in seeds:
            result = run_single(
                combination=combo,
                seed=seed,
                train_parquet=train_parquet,
                val_parquet=val_parquet,
                test_parquet=test_parquet,
                output_dir=output_dir,
                device=args.device,
                dry_run=args.dry_run,
            )
            results.append(result)

    # Агрегация
    if args.aggregate and not args.dry_run:
        from evaluation.compare_runs import (
            build_comparison_table,
            collect_metrics,
            print_comparison_summary,
            save_comparison_csv,
            save_comparison_markdown,
        )
        run_dirs = [Path(r["run_dir"]) for r in results if r.get("run_dir") and (Path(r["run_dir"]) / "metrics.jsonl").exists()]
        if run_dirs:
            records = collect_metrics(run_dirs)
            df = build_comparison_table(records)
            comparison_dir = ensure_dir(output_dir / "comparison")
            save_comparison_csv(df, comparison_dir / "comparison.csv")
            save_comparison_markdown(df, comparison_dir / "comparison.md")
            print_comparison_summary(df)
            log.info("Сравнение сохранено в %s", comparison_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
