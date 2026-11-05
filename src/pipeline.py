"""Сквозной конвейер эксперимента: PREPARE → TRAIN → EVALUATE → COMPARE.

Каждый этап обёрнут в run_stage с retry (экспоненциальная задержка),
таймингом и сбором PipelineResult. run_meta сохраняется в начале и конце.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from config import ExperimentConfig
from paths import BASE_DIR, ensure_dir
from run_meta import RunMeta, new_run_meta, save_run_meta

log = logging.getLogger("pipeline")


class PipelineStage(Enum):
    PREPARE = "prepare"
    TRAIN = "train"
    EVALUATE = "evaluate"
    COMPARE = "compare"


@dataclass
class PipelineResult:
    stage: PipelineStage
    status: str = "running"  # success | failed | skipped
    started_at: str | None = None
    finished_at: str | None = None
    duration_sec: float | None = None
    error: str | None = None
    artifacts: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def run_stage(
    name: str,
    callable_fn,
    retry: int = 3,
    backoff: float = 2.0,
    timeout: int | None = None,
) -> PipelineResult:
    """Запуск этапа с retry, таймингом и обработкой ошибок."""
    result = PipelineResult(stage=PipelineStage(name))
    result.started_at = _now_iso()
    last_error: Exception | None = None

    for attempt in range(1, retry + 1):
        try:
            log.info(
                "[STAGE %s] попытка %d/%d", name, attempt, retry
            )
            start = time.time()
            callable_fn()
            elapsed = time.time() - start
            result.status = "success"
            result.finished_at = _now_iso()
            result.duration_sec = round(elapsed, 2)
            log.info(
                "[STAGE %s] завершён за %.2f сек", name, elapsed
            )
            return result
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            result.error = str(exc)
            log.warning(
                "[STAGE %s] попытка %d/%d не удалась: %s",
                name,
                attempt,
                retry,
                exc,
            )
            if attempt < retry:
                time.sleep(backoff**attempt)

    result.status = "failed"
    result.finished_at = _now_iso()
    result.duration_sec = None
    log.error(
        "[STAGE %s] не прошёл после %d попыток: %s",
        name,
        retry,
        last_error,
    )
    return result


@dataclass
class Pipeline:
    config: ExperimentConfig
    run_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.run_dir is None:
            self.run_dir = ensure_dir(
                Path(self.config.log_dir) / f"{self.config.run_name}_{self.config.mode}"
            )
        self.results: list[PipelineResult] = []
        self.meta: RunMeta | None = None

    def _prepare(self) -> None:
        """Этап PREPARE: вызов scripts/prepare_3m.py через subprocess."""
        import sys as _sys

        from scripts.prepare_3m import main as prepare_main

        # Перенаправляем argv для prepare_3m.py
        old_argv = _sys.argv
        _sys.argv = [
            "prepare_3m.py",
            "--raw", self.config.data.raw_path,
            "--out", self.config.data.processed_dir,
            "--interactions", str(self.config.data.interactions),
            "--seed", str(self.config.data.seed),
            "--train-share", str(self.config.data.train_share),
            "--val-share", str(self.config.data.val_share),
            "--test-share", str(self.config.data.test_share),
            "--gap-seconds", str(self.config.data.gap_seconds),
            "--with-test" if self.config.with_test else "--no-test",
        ]
        try:
            prepare_main()
        finally:
            _sys.argv = old_argv

    def _train(self) -> None:
        """Этап TRAIN: вызов train_model.py через subprocess для каждого режима logQ."""
        import sys as _sys

        from train_model import main as train_main

        logq_modes = ["plain", "standard", "improved"]
        for mode in logq_modes:
            old_argv = _sys.argv
            _sys.argv = [
                "train_model.py",
                "--run-name", f"{self.config.run_name}_{mode}",
                "--mode", self.config.mode,
                "--train-parquet",
                str(Path(self.config.data.processed_dir) / "train_3m.parquet"),
                "--val-parquet",
                str(Path(self.config.data.processed_dir) / "val_3m.parquet"),
                "--test-parquet",
                str(Path(self.config.data.processed_dir) / "test_3m.parquet"),
                "--epochs", str(self.config.train.epochs),
                "--batch-size", str(self.config.train.batch_size),
                "--lr", str(self.config.model.lr),
                "--seed", str(self.config.train.seed),
                "--temperature", str(self.config.train.temperature),
                "--logq-mode", mode,
                "--weight-decay", str(self.config.model.weight_decay),
                "--max-seq-len", str(self.config.model.max_seq_len),
                "--device", self.config.train.device,
                "--with-test",
                "--output-dir", self.config.output_dir,
                "--log-dir", self.config.log_dir,
            ]
            try:
                exit_code = train_main()
                if exit_code != 0:
                    raise RuntimeError(
                        f"train_model.py ({mode}) вернул код {exit_code}"
                    )
            finally:
                _sys.argv = old_argv

    def _evaluate(self) -> None:
        """Этап EVALUATE: оценка уже встроена в train_model.py (--with-test).
        Здесь можно добавить дополнительную постобработку, если нужно.
        """
        log.info("[STAGE evaluate] оценка уже выполнена в этапе train (--with-test)")

    def _compare(self) -> None:
        """Этап COMPARE: агрегация metrics.jsonl из всех запусков."""
        import glob

        from evaluation.compare_runs import (
            build_comparison_table,
            collect_metrics,
            print_comparison_summary,
            save_comparison_csv,
            save_comparison_markdown,
        )

        runs_dir = Path(self.config.output_dir)
        log_dirs = [
            Path(d)
            for d in glob.glob(str(runs_dir / "*"))
            if (Path(d) / "metrics.jsonl").exists()
        ]
        if not log_dirs:
            log.warning(
                "[STAGE compare] не найдено директорий с metrics.jsonl в %s",
                runs_dir,
            )
            return

        records = collect_metrics(log_dirs)
        df = build_comparison_table(records)

        comparison_dir = ensure_dir(runs_dir / "comparison")
        save_comparison_csv(df, comparison_dir / "comparison.csv")
        save_comparison_markdown(df, comparison_dir / "comparison.md")
        print_comparison_summary(df)

        log.info(
            "[STAGE compare] результаты сохранены в %s",
            comparison_dir,
        )

    def run(
        self,
        stages: list[PipelineStage] | None = None,
        retry: int = 3,
    ) -> list[PipelineResult]:
        """Последовательное выполнение этапов конвейера."""
        if stages is None:
            stages = [
                PipelineStage.PREPARE,
                PipelineStage.TRAIN,
                PipelineStage.EVALUATE,
                PipelineStage.COMPARE,
            ]

        stage_map = {
            PipelineStage.PREPARE: self._prepare,
            PipelineStage.TRAIN: self._train,
            PipelineStage.EVALUATE: self._evaluate,
            PipelineStage.COMPARE: self._compare,
        }

        self.meta = new_run_meta(
            run_name=self.config.run_name,
            mode=self.config.mode,
            seed=self.config.train.seed,
            args=self.config.to_dict(),
            repo_dir=BASE_DIR,
        )

        overall_status = "success"
        for stage in stages:
            callable_fn = stage_map[stage]
            result = run_stage(
                name=stage.value,
                callable_fn=callable_fn,
                retry=retry,
            )
            self.results.append(result)
            if result.status == "failed":
                overall_status = "failed"
                log.error(
                    "Конвейер остановлен на этапе %s: %s",
                    stage.value,
                    result.error,
                )
                break

        # Сохраняем run_meta
        if self.meta is not None:
            self.meta.status = overall_status
            self.meta.finished_at = _now_iso()
            if self.results and self.results[0].started_at:
                started = datetime.fromisoformat(self.results[0].started_at)
                finished = datetime.fromisoformat(self.meta.finished_at)
                self.meta.duration_sec = round(
                    (finished - started).total_seconds(), 2
                )
            self.meta.artifact_paths = [
                str(r.artifacts) for r in self.results if r.artifacts
            ]
            save_run_meta(self.run_dir / "run_meta.json", self.meta)

        log.info(
            "Конвейер завершён со статусом %s. Результаты: %s",
            overall_status,
            self.run_dir,
        )
        return self.results
