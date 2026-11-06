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

Важно: на Kaggle ядро запускается из /kaggle/src/, поэтому на верхнем
уровне нет импортов из src/ — они доступны только после клона репозитория
и настройки PYTHONPATH. Всё взаимодействие с кодом проекта происходит через
subprocess с PYTHONPATH=/kaggle/working/repo/src.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_URL = "https://github.com/gshjis/sasrec.git"
REPO_DIR = Path("/kaggle/working/repo")
WORK_DIR = Path("/kaggle/working")
DATA_DIR = WORK_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
LOGS_DIR = WORK_DIR / "logs"

log = logging.getLogger("kaggle_entry")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SASRec training on Kaggle")
    p.add_argument(
        "--mode",
        choices=["smoke", "full", "ablation"],
        default="full",
        help="Режим прогона",
    )
    p.add_argument(
        "--epochs", type=int, default=20, help="Количество эпох"
    )
    p.add_argument(
        "--temperature", type=float, default=0.5,
        help="Единая температура для ВСЕХ режимов logQ",
    )
    p.add_argument(
        "--interactions", type=int, default=0,
        help="3_000_000 = смоук-прогон, 0 = все доступные взаимодействия",
    )
    p.add_argument(
        "--logq-modes", nargs="*",
        default=["improved"],
        help="Режимы logQ для сравнения (по умолчанию только improved)",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Сид воспроизводимости",
    )
    p.add_argument(
        "--retry", type=int, default=3, help="Попытки для сетевых этапов",
    )
    return p.parse_args()


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
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
        run(["git", "checkout", "AI-refactoring"], cwd=REPO_DIR)
        run(["git", "pull", "--ff-only"], cwd=REPO_DIR)
    else:
        if REPO_DIR.exists():
            shutil.rmtree(REPO_DIR)
        run(
            [
                "git", "clone", "--depth", "1", "--branch", "AI-refactoring",
                REPO_URL, str(REPO_DIR),
            ],
            cwd=WORK_DIR,
        )


def install_deps() -> None:
    """Этап 2: зависимости (с логгированием вывода pip)."""
    req = REPO_DIR / "requirements.txt"
    cmd = [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)]
    log.info("[CMD] %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(REPO_DIR / "src")},
    )
    if result.returncode != 0:
        log.error("pip install вернул код %d", result.returncode)
        if result.stderr:
            for line in result.stderr.splitlines()[-50:]:
                log.error("pip stderr: %s", line)
        raise RuntimeError(f"pip install не удался (код {result.returncode})")
    if result.stdout:
        for line in result.stdout.splitlines()[-100:]:
            log.info("pip: %s", line)
    log.info("Зависимости установлены")


def download_data(raw_dir: Path, retry: int) -> Path:
    """Этап 3: скачивание данных с HF с retry."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_parquet = raw_dir / "flat" / "50m" / "listens.parquet"

    if raw_parquet.exists():
        log.info("Сырые данные уже есть: %s", raw_parquet)
        return raw_parquet

    # Используем subprocess для вызова data_extractor из клонированного репо
    script = (
        "from data_extractor.config import HfConfig\n"
        "from data_extractor.hf_loader import HfYambdaLoader\n"
        "loader = HfYambdaLoader(HfConfig(\n"
        "    repo_id='yandex/yambda',\n"
        "    filename='flat/50m/listens.parquet',\n"
        f"    download_dir={str(raw_dir)!r}\n"
        "))\n"
        "print(loader.download())\n"
    )

    last_error: Exception | None = None
    for attempt in range(1, retry + 1):
        try:
            log.info(
                "Скачивание yandex/yambda (попытка %d/%d)...", attempt, retry
            )
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=REPO_DIR,
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(REPO_DIR / "src")},
            )
            log.info("Скачано: %s", result.stdout.strip())
            return raw_parquet
        except subprocess.CalledProcessError as exc:
            last_error = exc
            log.warning(
                "Скачивание не удалось (попытка %d/%d): %s\nstderr: %s",
                attempt, retry, exc, exc.stderr,
            )
            if attempt < retry:
                time.sleep(2 ** attempt)
    raise RuntimeError(
        f"Не удалось скачать данные после {retry} попыток: {last_error}"
    )


def save_run_meta_json(path: Path, run_name: str, mode: str, seed: int,
                       args: dict, status: str, error: str | None = None,
                       started_at: str | None = None,
                       finished_at: str | None = None) -> None:
    """Сохранить run_meta.json (простая версия без импорта из src)."""
    meta = {
        "run_name": run_name,
        "mode": mode,
        "seed": seed,
        "args": args,
        "status": status,
        "error": error,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    log_dir = LOGS_DIR / f"kaggle_{args.mode}"
    setup_logging(log_dir)

    started_at = datetime.now(UTC).isoformat()

    meta_path = log_dir / "run_meta.json"
    save_run_meta_json(
        meta_path,
        run_name=f"kaggle_{args.mode}",
        mode=args.mode,
        seed=args.seed,
        args=vars(args),
        status="running",
        started_at=started_at,
    )

    try:
        clone_repo()
        install_deps()
        download_data(RAW_DIR, args.retry)

        # Этап 4: подготовка данных (prepare_3m.py)
        run(
            [
                sys.executable, "scripts/prepare_3m.py",
                "--raw", str(RAW_DIR / "flat" / "50m" / "listens.parquet"),
                "--out", str(DATA_DIR / "processed"),
                "--interactions", str(args.interactions),
                "--seed", str(args.seed),
                "--train-share", "0.70",
                "--val-share", "0.15",
                "--test-share", "0.15",
                "--gap-seconds", "3600",
                "--with-test",
            ],
            cwd=REPO_DIR,
        )

        # Этап 5: обучение 3 режима logQ с единой температурой
        common = [
            sys.executable, "train_model.py",
            "--train-parquet", str(DATA_DIR / "processed" / "train_3m.parquet"),
            "--val-parquet", str(DATA_DIR / "processed" / "val_3m.parquet"),
            "--test-parquet", str(DATA_DIR / "processed" / "test_3m.parquet"),
            "--data-dir", str(DATA_DIR),
            "--epochs", str(args.epochs),
            "--batch-size", "256",
            "--device", "auto",
        ]

        for mode in args.logq_modes:
            log.info("[STEP] Обучение режима: %s", mode)
            run(
                common + [
                    "--logq-mode", mode,
                    "--temperature", str(args.temperature),
                    "--seed", str(args.seed),
                    "--run-name", f"kaggle_{args.mode}_{mode}",
                    "--log-dir", str(LOGS_DIR),
                ],
                cwd=REPO_DIR,
            )

        # Этап 6: сравнение результатов
        run(
            [
                sys.executable, "-m", "src.evaluation.compare_runs",
                str(LOGS_DIR / f"kaggle_{args.mode}_plain"),
                str(LOGS_DIR / f"kaggle_{args.mode}_standard"),
                str(LOGS_DIR / f"kaggle_{args.mode}_improved"),
            ],
            cwd=REPO_DIR,
        )

        log.info("[DONE] Kaggle прогон завершён успешно")
        save_run_meta_json(
            meta_path,
            run_name=f"kaggle_{args.mode}",
            mode=args.mode,
            seed=args.seed,
            args=vars(args),
            status="success",
            started_at=started_at,
            finished_at=datetime.now(UTC).isoformat(),
        )
        return 0

    except Exception as exc:
        log.exception("Критическая ошибка")
        save_run_meta_json(
            meta_path,
            run_name=f"kaggle_{args.mode}",
            mode=args.mode,
            seed=args.seed,
            args=vars(args),
            status="failed",
            error=str(exc),
            started_at=started_at,
            finished_at=datetime.now(UTC).isoformat(),
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
