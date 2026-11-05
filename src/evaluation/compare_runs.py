"""Агрегация и сравнение метрик из нескольких запусков экспериментов.

Читает ``metrics.jsonl`` из директорий запусков (формат, который пишет
``train_model.py``), строит сравнительную таблицу по последней эпохе каждого
запуска, сохраняет её в CSV/Markdown и выводит сводку в stdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]
    _PANDAS_AVAILABLE = False

# Колонки сравнительной таблицы в порядке вывода.
COMPARISON_COLUMNS: list[str] = [
    "run_name",
    "logq_mode",
    "temperature",
    "epoch",
    "loss",
    "recall@10",
    "recall@50",
    "recall@100",
    "ndcg@10",
    "ndcg@50",
    "ndcg@100",
    "mrr@10",
    "mrr@50",
    "mrr@100",
    "coverage@10",
    "pareto_coverage@10",
    "pareto_head_coverage@10",
    "pareto_tail_coverage@10",
    "entropy",
]

METRICS_FILENAME = "metrics.jsonl"


def _require_pandas() -> None:
    if not _PANDAS_AVAILABLE or pd is None:
        raise ImportError(
            "pandas необходим для работы модуля compare_runs. "
            "Установите его: pip install pandas"
        )


def collect_metrics(log_dirs: list[Path]) -> list[dict]:
    """Собрать записи метрик из ``metrics.jsonl`` каждого запуска.

    Для каждой директории в ``log_dirs`` читается файл ``<dir>/metrics.jsonl``.
    Каждая непустая строка парсится как JSON; к записи добавляется ключ
    ``run_name`` (имя родительской директории). Битые строки пропускаются с
    предупреждением в stderr.
    """
    records: list[dict] = []
    for log_dir in log_dirs:
        d = Path(log_dir)
        run_name = d.name
        metrics_path = d / METRICS_FILENAME

        if not metrics_path.exists():
            print(
                f"[compare_runs] пропуск {d}: файл {METRICS_FILENAME} не найден",
                file=sys.stderr,
            )
            continue

        with metrics_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record: dict[str, Any] = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(
                        f"[compare_runs] пропуск строки {line_no} в {metrics_path}: {exc}",
                        file=sys.stderr,
                    )
                    continue
                if not isinstance(record, dict):
                    continue
                record["run_name"] = run_name
                records.append(record)

    return records


def build_comparison_table(records: list[dict]) -> pd.DataFrame:
    """Построить сравнительную таблицу по последней эпохе каждого запуска.

    Колонки: ``COMPARISON_COLUMNS``. Для каждого ``run_name`` берётся строка
    с максимальным ``epoch`` (при равных — последняя встречающаяся).
    Таблица сортируется по ``recall@10`` по убыванию.
    """
    _require_pandas()
    assert pd is not None

    if not records:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)

    df = pd.DataFrame(records)

    numeric_cols = [
        "epoch",
        "temperature",
        "loss",
        "recall@10",
        "recall@50",
        "recall@100",
        "ndcg@10",
        "ndcg@50",
        "ndcg@100",
        "mrr@10",
        "mrr@50",
        "mrr@100",
        "coverage@10",
        "pareto_coverage@10",
        "pareto_head_coverage@10",
        "pareto_tail_coverage@10",
        "entropy",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    present_cols = [c for c in COMPARISON_COLUMNS if c in df.columns]
    df = df[present_cols]

    if df.empty:
        return df

    # Последняя эпоха для каждого run_name.
    if "epoch" in df.columns:
        df = df.sort_values("epoch", kind="mergesort")
        df = df.drop_duplicates(subset=["run_name"], keep="last")

    # Сортировка по recall@10 убыв.
    if "recall@10" in df.columns:
        df = df.sort_values("recall@10", ascending=False, kind="mergesort")

    df = df.reset_index(drop=True)
    return df


def save_comparison_csv(df: pd.DataFrame, path: Path) -> None:
    """Сохранить таблицу сравнения в CSV (без индекса, UTF-8)."""
    _require_pandas()
    assert pd is not None

    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding="utf-8")


def save_comparison_markdown(df: pd.DataFrame, path: Path) -> None:
    """Сохранить таблицу сравнения в Markdown."""
    _require_pandas()
    assert pd is not None

    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)

    try:
        text = df.to_markdown(index=False)
    except Exception:  # noqa: BLE001  # tabulate не установлен
        text = df.to_string(index=False)

    p.write_text(text, encoding="utf-8")


def print_comparison_summary(df: pd.DataFrame) -> None:
    """Напечатать сводку сравнения в stdout."""
    _require_pandas()
    assert pd is not None

    if df.empty:
        print("[compare_runs] Нет данных для сводки.")
        return

    print("\n=== Сравнение запусков (последняя эпоха каждого run) ===\n")
    try:
        print(df.to_markdown(index=False))
    except Exception:  # noqa: BLE001
        print(df.to_string(index=False))

    best = find_best_config(df, metric="recall@10")
    if best:
        print(
            "\nЛучшая конфигурация по recall@10: "
            f"{best.get('run_name')} "
            f"(epoch={best.get('epoch')}, "
            f"logq_mode={best.get('logq_mode')}, "
            f"temperature={best.get('temperature')}, "
            f"recall@10={best.get('recall@10')})"
        )
    print()


def find_best_config(df: pd.DataFrame, metric: str = "recall@10") -> dict:
    """Вернуть строку (конфигурацию) с максимальным значением метрики.

    Возвращает пустой словарь, если таблица пуста или метрики нет в данных.
    """
    _require_pandas()
    assert pd is not None

    if df.empty or metric not in df.columns:
        return {}

    if df[metric].isna().all():
        return {}

    best_idx = df[metric].idxmax()
    row = df.loc[best_idx]
    return {
        k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()
    }
