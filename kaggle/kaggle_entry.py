"""Тонкий оркестратор для Kaggle ядра.

Последовательно запускает:
  1. Клон репозитория (или pull)
  2. pip install -r requirements.txt
  3. Скачивание данных с HF (с retry)
  4. Подготовку выборки (prepare_3m.py) — train/val/test
  5. Обучение 3 режима logQ с ЕДИНОЙ температурой
  6. Сравнение результатов (comparison.csv)
  7. Сохранение run_meta.json

Режимы: smoke (быстро), full (полный прогон), ablation (разные logQ).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Абсолютные пути
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import DataConfig, ExperimentConfig, TrainConfig
from paths import ensure_dir
from pipeline import Pipeline, PipelineStage
from run_meta import (
    RunMeta,
    save_run_meta,
)

REPO_URL = "https://github.com/gshjis/sasrec.git"
REPO_DIR = Path("/kaggle/working/repo")
WORK_DIR = Path("/kaggle/working")
DATA_DIR = WORK_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

log = logging.getLogger("kaggle_entry")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SASRec training on Kaggle")
    p.add_argument(
        "--mode",
        choices=["smoke", "full", "ablation"],
        default="smoke",
        help="Режим прогона",
    )
    p.add_argument(
        "--epochs", type=int, default=5, help="Количество эпох"
    )
    p.add_argument(
        "--temperature", type=float, default=1.0,
        help="Единая температура для ВСЕХ режимов logQ",
    )
    p.add_argument(
        "--interactions", type=int, default=0,
        help="0 = все доступные взаимодействия",
    )
    p.add_argument(
        "--logq-modes", nargs="*",
        default=["plain", "standard", "improved"],
        help="Режимы logQ для сравнения",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Сид воспроизводимости",
    )
    p.add_argument(
        "--retry", type=int, default=3, help="Попытки для сетевых этапов",
    )
    return p.parse_args()


def setup_logging(log_dir: Path) -> None:
    ensure_dir(log_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "kaggle_entry.log", encoding="utf-8"),
        ],
    )


def run(cmd: list[str], cwd: Path, retry: int = 3) -> None:
    """Запуск subprocess с retry и понятной ошибкой."""
    last_error: Exception | None = None
    for attempt in range(1, retry + 1):
        try:
            log.info("[CMD attempt %d/%d] %s", attempt, retry, " ".join(cmd))
            subprocess.run(
                cmd,
                cwd=cwd,
                check=True,
                env={**os.environ, "PYTHONPATH": str(REPO_DIR / "src")},
            )
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            log.warning(
                "Команда не удалась (попытка %d/%d): %s", attempt, retry, exc
            )
            if attempt < max(1, retry - 1):
                time.sleep(2 ** attempt)
    raise RuntimeError(
        f"Команда не выполнилась после {retry} попыток: {last_error}"
    )


def clone_repo() -> None:
    """Этап 1: клон или pull репозитория."""
    if (REPO_DIR / ".git").exists():
        run(["git", "pull", "--ff-only"], cwd=REPO_DIR)
    else:
        if REPO_DIR.exists():
            shutil.rmtree(REPO_DIR)
        run(
            ["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)],
            cwd=WORK_DIR,
        )


def install_deps() -> None:
    """Этап 2: зависимости."""
    req = REPO_DIR / "requirements.txt"
    run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
        cwd=REPO_DIR,
    )


def download_data(raw_dir: Path, retry: int) -> Path:
    """Этап 3: скачивание данных с HF с retry."""
    from data_extractor.config import HfConfig
    from data_extractor.hf_loader import HfYambdaLoader

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_parquet = raw_dir / "flat" / "50m" / "listens.parquet"

    if raw_parquet.exists():
        log.info("Сырые данные уже есть: %s", raw_parquet)
        return raw_parquet

    last_error: Exception | None = None
    for attempt in range(1, retry + 1):
        try:
            log.info(
                "Скачивание yandex/yambda (попытка %d/%d)...", attempt, retry
            )
            loader = HfYambdaLoader(
                HfConfig(
                    repo_id="yandex/yambda",
                    filename="flat/50m/listens.parquet",
                    download_dir=str(raw_dir),
                )
            )
            result = loader.download()
            return Path(result)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            log.warning(
                "Скачивание не удалось (попытка %d/%d): %s", attempt, retry, exc
            )
            if attempt < retry:
                time.sleep(2 ** attempt)
    raise RuntimeError(
        f"Не удалось скачать данные после {retry} попыток: {last_error}"
    )


def main() -> int:
    args = parse_args()
    config = ExperimentConfig(
        run_name=f"kaggle_{args.mode}",
        mode=args.mode,
        data=DataConfig(
            interactions=args.interactions,
            seed=args.seed,
        ),
        train=TrainConfig(
            epochs=args.epochs,
            temperature=args.temperature,
            seed=args.seed,
        ),
    )
    log_dir = ensure_dir(Path("/kaggle/working") / "logs" / config.run_name)
    setup_logging(log_dir)

    run_meta = RunMeta(
        run_name=config.run_name,
        mode=config.mode,
        seed=config.train.seed,
        args=vars(args),
        started_at=datetime.now(UTC).isoformat(),
    )

    try:
        clone_repo()
        install_deps()
        download_data(RAW_DIR, args.retry)

        pipeline = Pipeline(config=config)
        pipeline.run(
            stages=[
                PipelineStage.PREPARE,
                PipelineStage.TRAIN,
                PipelineStage.EVALUATE,
                PipelineStage.COMPARE,
            ],
            retry=args.retry,
        )

        run_meta.status = "success"
        log.info("[DONE] Kaggle прогон завершён успешно")

    except Exception as exc:
        log.exception("Критическая ошибка")
        run_meta.status = "failed"
        run_meta.error = str(exc)
        return 1
    finally:
        run_meta.finished_at = datetime.now(UTC).isoformat()
        save_run_meta(log_dir / "run_meta.json", run_meta)

    return 0


if __name__ == "__main__":
    sys.exit(main())
