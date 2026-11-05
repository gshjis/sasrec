"""Точка входа для Kaggle ядра.

Пробный прогон: 3 млн взаимодействий, 5 эпох, два режима (без logQ и с logQ).

Шаги:
  1. Клонирует репозиторий проекта в /kaggle/working/repo.
  2. Ставит зависимости (torch на Kaggle уже есть).
  3. Скачивает данные Yambda 50m с Hugging Face.
  4. Готовит 3M выборку с сессиями (scripts/prepare_3m.py).
  5. Запускает обучение на 5 эпох дважды: без --logQ и с --logQ.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/gshjis/sasrec.git"
REPO_DIR = Path("/kaggle/working/repo")
WORK_DIR = Path("/kaggle/working")
DATA_DIR = WORK_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERACTIONS = 3_000_000
EPOCHS = 5


def run(cmd: list[str], cwd: Path) -> None:
    print(f"[CMD] {' '.join(cmd)}", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_DIR / "src")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main() -> None:
    # 1. Клон репозитория
    if (REPO_DIR / ".git").exists():
        run(["git", "pull", "--ff-only"], cwd=REPO_DIR)
    else:
        if REPO_DIR.exists():
            shutil.rmtree(REPO_DIR)
        run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], cwd=WORK_DIR)

    # 2. Зависимости (на Kaggle torch уже установлен)
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "polars",
            "pyarrow",
            "huggingface_hub",
        ],
        cwd=REPO_DIR,
    )

    # 3. Данные с Hugging Face
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_parquet = RAW_DIR / "flat" / "50m" / "listens.parquet"
    if not raw_parquet.exists():
        print("[STEP] Скачивание данных Yambda 50m с Hugging Face...", flush=True)
        run(
            [
                sys.executable,
                "-c",
                (
                    "from data_extractor.config import HfConfig\n"
                    "from data_extractor.hf_loader import HfYambdaLoader\n"
                    "loader = HfYambdaLoader(HfConfig("
                    "repo_id='yandex/yambda', "
                    "filename='flat/50m/listens.parquet', "
                    f"download_dir={str(RAW_DIR)!r}))\n"
                    "print('Скачано:', loader.download())\n"
                ),
            ],
            cwd=REPO_DIR,
        )

    # 4. Подготовка 3M выборки с сессиями
    train_parquet = DATA_DIR / "processed" / "train_3m.parquet"
    val_parquet = DATA_DIR / "processed" / "val_3m.parquet"
    if not (train_parquet.exists() and val_parquet.exists()):
        print(f"[STEP] Подготовка {INTERACTIONS:,} взаимодействий...", flush=True)
        run(
            [
                sys.executable,
                "scripts/prepare_3m.py",
                "--raw",
                str(raw_parquet),
                "--out",
                str(DATA_DIR / "processed"),
                "--interactions",
                str(INTERACTIONS),
            ],
            cwd=REPO_DIR,
        )

    # 5. Обучение: без logQ и с logQ
    common = [
        "--train-parquet",
        str(train_parquet),
        "--val-parquet",
        str(val_parquet),
        "--data-dir",
        str(DATA_DIR),
        "--epochs",
        str(EPOCHS),
        "--batch-size",
        "256",
        "--device",
        "cuda" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else "cpu",
    ]

    print("[STEP] Прогон 1/2: in-batch negatives без logQ", flush=True)
    run(
        [
            sys.executable,
            "train_model.py",
            *common,
            "--log-dir",
            str(WORK_DIR / "logs_no_logq"),
        ],
        cwd=REPO_DIR,
    )

    print("[STEP] Прогон 2/2: in-batch negatives с logQ", flush=True)
    run(
        [
            sys.executable,
            "train_model.py",
            *common,
            "--log-dir",
            str(WORK_DIR / "logs_logq"),
            "--logQ",
        ],
        cwd=REPO_DIR,
    )

    print(
        "[DONE] Логи: /kaggle/working/logs_no_logq и /kaggle/working/logs_logq",
        flush=True,
    )


if __name__ == "__main__":
    main()
