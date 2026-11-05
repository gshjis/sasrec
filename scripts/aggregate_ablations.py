"""Агрегация результатов абляционных исследований с статистической значимостью.

Обходит все <run_id>/ директории, парсит run_meta.json и финальную строку
metrics.jsonl (лучшая эпоха), строит сравнительную таблицу со статистической
значимостью (парный t-тест между режимами, поправка Бонферрони, коэфеннт
Коэна d).

Пример:
    PYTHONPATH=src python scripts/aggregate_ablations.py \
        --input-dir ./ablation_runs/full \
        --output-dir ./ablation_runs/full/comparison
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

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
    p.add_argument(
        "--alpha", type=float, default=0.05,
        help="Уровень значимости для статистических тестов",
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
            for grandchild in child.iterdir():
                if grandchild.is_dir() and (grandchild / "metrics.jsonl").exists():
                    run_dirs.append(grandchild)
    return run_dirs


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Коэффициент Коэна d для двух независимых выборок."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    dof = nx + ny - 2
    pooled_std = math.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof
    )
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


def _paired_ttest(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Парный t-тест Стьюдента: возвращает (t-stat, p-value)."""
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan")
    diff = x - y
    if np.std(diff, ddof=1) == 0:
        return 0.0, 1.0
    return stats.ttest_rel(x, y)


def _bonferroni_correction(p_values: list[float], alpha: float) -> list[tuple[float, bool]]:
    """Поправка Бонферрони: возвращает (adjusted_p, significant) для каждого p."""
    m = len(p_values)
    adjusted = [min(1.0, p * m) for p in p_values]
    return [(p, p < alpha) for p in adjusted]


def _group_by_config_and_seed(records: list[dict]) -> dict[tuple, dict[str, list[float]]]:
    """Группирует метрики по (конфигурация, сид).

    Возвращает: {(logq_mode, temperature): {metric: [значения по сидам]}}
    """
    grouped: dict[tuple, dict[str, list[float]]] = {}
    for rec in records:
        key = (rec.get("logq_mode"), rec.get("temperature"))
        if key not in grouped:
            grouped[key] = {}
        for metric in ["recall@10", "recall@50", "recall@100",
                       "ndcg@10", "ndcg@50", "ndcg@100",
                       "mrr@10", "mrr@50", "mrr@100",
                       "coverage@10", "pareto_coverage@10",
                       "pareto_head_coverage@10", "pareto_tail_coverage@10",
                       "entropy"]:
            if metric in rec and rec[metric] is not None:
                grouped[key].setdefault(metric, []).append(float(rec[metric]))
    return grouped


def compute_statistical_significance(
    grouped: dict[tuple, dict[str, list[float]]],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Вычисляет статистическую значимость между парами режимов.

    Сравнивает все пары (logq_mode, temperature) по каждой метрике.
    Возвращает результаты парных t-тестов с поправкой Бонферрони.
    """
    keys = list(grouped.keys())
    results: dict[str, Any] = {
        "pairwise_tests": [],
        "summary": {},
    }

    metrics = [
        "recall@10", "recall@50", "recall@100",
        "ndcg@10", "ndcg@50", "ndcg@100",
        "mrr@10", "mrr@50", "mrr@100",
        "coverage@10", "pareto_coverage@10",
        "pareto_head_coverage@10", "pareto_tail_coverage@10",
        "entropy",
    ]

    p_values: list[float] = []
    test_entries: list[dict[str, Any]] = []

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            key_i, key_j = keys[i], keys[j]
            group_i = grouped[key_i]
            group_j = grouped[key_j]

            for metric in metrics:
                vals_i = group_i.get(metric, [])
                vals_j = group_j.get(metric, [])
                if len(vals_i) < 2 or len(vals_j) < 2:
                    continue

                arr_i = np.array(vals_i)
                arr_j = np.array(vals_j)

                # Выравниваем по минимальному числу сидов
                n = min(len(arr_i), len(arr_j))
                arr_i = arr_i[:n]
                arr_j = arr_j[:n]

                t_stat, p_val = _paired_ttest(arr_i, arr_j)
                d = _cohens_d(arr_i, arr_j)

                mean_i = float(np.mean(arr_i))
                mean_j = float(np.mean(arr_j))
                diff = mean_i - mean_j

                test_entries.append({
                    "mode_i": key_i[0],
                    "temp_i": key_i[1],
                    "mode_j": key_j[0],
                    "temp_j": key_j[1],
                    "metric": metric,
                    "mean_i": mean_i,
                    "mean_j": mean_j,
                    "diff": diff,
                    "t_stat": float(t_stat),
                    "p_value": float(p_val),
                    "cohens_d": float(d),
                    "n": int(n),
                })
                p_values.append(float(p_val))

    # Поправка Бонферрони
    adjusted = _bonferroni_correction(p_values, alpha)
    for entry, (adj_p, significant) in zip(test_entries, adjusted):
        entry["p_adjusted"] = adj_p
        entry["significant"] = bool(significant)

    results["pairwise_tests"] = test_entries
    return results


def print_statistical_summary(results: dict[str, Any]) -> None:
    """Печатает таблицу статистической значимости."""
    tests = results.get("pairwise_tests", [])
    if not tests:
        log.info("Нет данных для статистических тестов")
        return

    log.info("\n=== Статистическая значимость (парный t-тест, поправка Бонферрони) ===")
    header = (
        f"{'Сравнение':<40} {'Метрика':<15} {'Δ':>10} {'p-val':>10} "
        f"{'p-adj':>10} {'Зн-й':>6} {'d':>8}"
    )
    log.info(header)
    log.info("-" * len(header))

    for t in tests:
        comparison = f"{t['mode_i']}(T={t['temp_i']}) vs {t['mode_j']}(T={t['temp_j']})"
        sig_mark = "***" if t["significant"] else "   "
        log.info(
            f"{comparison:<40} {t['metric']:<15} {t['diff']:>10.4f} "
            f"{t['p_value']:>10.4f} {t['p_adjusted']:>10.4f} "
            f"{sig_mark:<6} {t['cohens_d']:>8.3f}"
        )


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

    # Статистическая значимость
    grouped = _group_by_config_and_seed(records)
    if len(grouped) >= 2:
        stats_results = compute_statistical_significance(grouped, alpha=args.alpha)
        print_statistical_summary(stats_results)

        # Сохраняем результаты статистики
        stats_path = output_dir / "statistical_significance.json"
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats_results, f, ensure_ascii=False, indent=2, default=str)
        log.info("Статистическая значимость сохранена в %s", stats_path)

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