"""Абсолютные пути проекта, независимые от текущей рабочей директории."""

from __future__ import annotations

import os
from pathlib import Path

# Корень репозитория: src/paths.py -> src/ -> корень
BASE_DIR = Path(__file__).resolve().parents[1]

# Директории проекта (относительно корня репозитория)
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LOGS_DIR = BASE_DIR / "logs"
RUNS_DIR = BASE_DIR / "runs"
SCRIPTS_DIR = BASE_DIR / "scripts"
KAGGLE_DIR = BASE_DIR / "kaggle"


def get_env(name: str, default: str) -> str:
    """Прочитать переменную окружения с дефолтом."""
    return os.environ.get(name, default)


def data_dir() -> Path:
    return Path(get_env("INFORMATICA_DATA_DIR", str(DATA_DIR)))


def logs_dir() -> Path:
    return Path(get_env("INFORMATICA_LOGS_DIR", str(LOGS_DIR)))


def runs_dir() -> Path:
    return Path(get_env("INFORMATICA_RUNS_DIR", str(RUNS_DIR)))


def ensure_dir(path: Path | str) -> Path:
    """Создать директорию (включая родительские), если не существует."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
