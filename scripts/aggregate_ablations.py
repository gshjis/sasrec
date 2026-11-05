"""Агрегация результатов абляционных исследований.

Обходит все <run_id>/ директории, парсит run_meta.json и финальную строку
metrics.jsonl (лучшая эпоха), строит сравнительную таблицу.

Пример:
    PYTHONPATH=src python scripts/aggregate_ablations.py \
        --input-dir ./ablation_runs/full \
        --output-dir ./ablation_runs/full/comparison
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation.compare_runs import (
    build_comparison_table,
    collect_metrics,
    find_best_config,
    print_comparison_summary,
    save_comparison_csv,
    save_comparison_markdown,
)

log = logging.getLogger("aggregate_ablations")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Агрегация результатов абляций")
    p.add_argument(
        "--input-dir", type=str, required=True,
        help="Корень каталога с результатами абляций",
    )
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Куда сохранить сравнение (по умолчанию input-dir/comparison)",
    )
    return p.parse_args()


def find_run_dirs(input_dir: Path) -> list[Path]:
    """Найти все директории запусков с metrics.jsonl."""
    input_dir = Path(input_dir)
    run_dirs: list[Path] = []
    for child in input_dir.iterdir():
        if child.is_dir() and (child / "metrics.jsonl").exists():
            run_dirs.append(child)
        elif child.is_dir() and child.name == "comparison":
            continue
        elif child.is_dir():
            # Рекурсивный поиск на один уровень вглубь
            for grandchild in child.iterdir():
                if grandchild.is_dir() and (grandchild / "metrics.jsonl").exists():
                    run_dirs.append(grandchild)
    return run_dirs


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "comparison"

    run_dirs = find_run_dirs(input_dir)
    log.info("Найдено запусков с metrics.jsonl: %d", len(run_dirs))

    if not run_dirs:
        log.warning("Не найдено запусков для агрегации")
        return 1

    records = collect_metrics(run_dirs)
    df = build_comparison_table(records)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_comparison_csv(df, output_dir / "comparison.csv")
    save_comparison_markdown(df, output_dir / "comparison.md")
    print_comparison_summary(df)

    # Лучшая конфигурация по нескольким метрикам
    for metric in ["recall@10", "ndcg@10", "mrr@10", "coverage@10"]:
        best = find_best_config(df, metric=metric)
        if best:
            log.info(
                "Лучшая по %s: %s (%.4f)", metric, best.get("run_name"), best.get(metric)
            )

    log.info("Результаты сохранены в %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
