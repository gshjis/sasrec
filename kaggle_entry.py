"""Точка входа для Kaggle ядра.

Клонирует репозиторий (если запускается не из репозитория), скачивает
данные с Hugging Face (уже реализовано в проекте через HfYambdaLoader),
формирует тестовую выборку из первых 3 млн взаимодействий и запускает
обучение SASRec на 5 эпох (с и без --logQ).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/<USER>/<REPO>.git"  # <-- замените на свой
REPO_DIR = Path("/kaggle/working/repo")
WORK_DIR = Path("/kaggle/working")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    # 1. Клонируем репозиторий в /kaggle/working/repo (если ещё не cloned)
    if not (REPO_DIR / ".git").exists():
        if REPO_DIR.exists():
            shutil.rmtree(REPO_DIR)
        run(["git", "clone", REPO_URL, str(REPO_DIR)])
    else:
        run(["git", "pull"], cwd=REPO_DIR)

    # 2. Переходим в директорию репозитория
    repo_src = REPO_DIR / "src"
    if not repo_src.exists():
        print(f"ОШИБКА: src/ не найден в {REPO_DIR}")
        sys.exit(1)

    # 3. Пути к данным
    data_dir = WORK_DIR / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_parquet = processed_dir / "train_3m.parquet"
    val_parquet = processed_dir / "val_3m.parquet"

    # 4. Если данных нет — качаем с HF через встроенный HfYambdaLoader
    if not train_parquet.exists() or not val_parquet.exists():
        print("[STEP] Скачивание данных с Hugging Face (yandex/yambda, flat/50m/listens.parquet)...")
        run(
            [
                sys.executable,
                "-c",
                (
                    "from data_extractor.config import HfConfig\n"
                    "from data_extractor.hf_loader import HfYambdaLoader\n"
                    "loader = HfYambdaLoader(HfConfig(repo_id='yandex/yambda', "
                    "filename='flat/50m/listens.parquet', download_dir='/kaggle/working/data/raw'))\n"
                    "loader.download()\n"
                ),
            ],
            cwd=REPO_DIR,
        )

    # 5. Формируем тестовую выборку: первые 3 млн взаимодействий
    if not train_parquet.exists() or not val_parquet.exists():
        print("[STEP] Формирование 3M выборки...")
        run(
            [
                sys.executable,
                "-c",
                (
                    "import polars as pl\n"
                    "df = pl.read_parquet('/kaggle/working/data/raw/flat/50m/listens.parquet')\n"
                    "df3 = df.head(3_000_000)\n"
                    "n = df3.height\n"
                    "n_train = int(n * 0.7)\n"
                    "n_val = int(n * 0.15)\n"
                    "df3[:n_train].write_parquet('/kaggle/working/data/processed/train_3m.parquet')\n"
                    "df3[n_train:n_train + n_val].write_parquet('/kaggle/working/data/processed/val_3m.parquet')\n"
                    "print(f'Train: {n_train}, Val: {n_val}')\n"
                ),
            ],
            cwd=REPO_DIR,
        )

    # 6. Запуск обучения без logQ
    print("[STEP] Обучение без logQ (5 эпох)...")
    run(
        [
            sys.executable,
            "train_model.py",
            "--train-parquet",
            str(train_parquet),
            "--val-parquet",
            str(val_parquet),
            "--data-dir",
            str(data_dir),
            "--log-dir",
            str(WORK_DIR / "logs_no_logQ"),
            "--epochs",
            "5",
            "--batch-size",
            "256",
        ],
        cwd=REPO_DIR,
    )

    # 7. Запуск обучения с logQ
    print("[STEP] Обучение с logQ (5 эпох)...")
    run(
        [
            sys.executable,
            "train_model.py",
            "--train-parquet",
            str(train_parquet),
            "--val-parquet",
            str(val_parquet),
            "--data-dir",
            str(data_dir),
            "--log-dir",
            str(WORK_DIR / "logs_logQ"),
            "--epochs",
            "5",
            "--batch-size",
            "256",
            "--logQ",
        ],
        cwd=REPO_DIR,
    )

    print("[DONE] Оба прогона завершены. Логи в /kaggle/working/logs_no_logQ и /kaggle/working/logs_logQ")


if __name__ == "__main__":
    main()
