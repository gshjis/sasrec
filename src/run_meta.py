"""Метаданные воспроизводимости эксперимента (run_meta.json)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import polars as pl  # noqa: F401
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False


@dataclass
class RunMeta:
    """Метаданные одного запуска эксперимента."""
    run_name: str
    mode: str
    commit_sha: str | None = None
    commit_message: str | None = None
    branch: str | None = None
    python_version: str | None = None
    torch_version: str | None = None
    polars_version: str | None = None
    seed: int | None = None
    args: dict[str, Any] = field(default_factory=dict)
    data_hash: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    duration_sec: float | None = None
    status: str = "running"
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_paths: list[str] = field(default_factory=list)
    error: str | None = None


def capture_git_info(repo_dir: Path | None = None) -> dict[str, str | None]:
    """Получить информацию о git-коммите, ветке и сообщении."""
    if repo_dir is None:
        repo_dir = Path.cwd()
    info: dict[str, str | None] = {"sha": None, "message": None, "branch": None}
    try:
        info["sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    try:
        info["message"] = subprocess.check_output(
            ["git", "log", "-1", "--format=%B"], cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    try:
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return info


def capture_environment() -> dict[str, str | None]:
    """Собрать версии интерпретатора и библиотек."""
    env: dict[str, str | None] = {
        "python": sys.version,
        "torch": None,
        "polars": None,
    }
    if _TORCH_AVAILABLE:
        import torch
        env["torch"] = torch.__version__
    if _POLARS_AVAILABLE:
        import polars as pl
        env["polars"] = pl.__version__
    return env


def compute_data_hash(path: Path | str) -> str | None:
    """Вычислить SHA-256 файла с данными для контроля целостности."""
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_run_meta(path: Path | str, meta: RunMeta) -> None:
    """Сохранить RunMeta в JSON-файл."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, ensure_ascii=False, indent=2, default=str)


def load_run_meta(path: Path | str) -> RunMeta:
    """Загрузить RunMeta из JSON-файла."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        d = json.load(f)
    return RunMeta(**d)


def new_run_meta(run_name: str, mode: str, seed: int | None = None,
                 args: dict[str, Any] | None = None,
                 repo_dir: Path | None = None) -> RunMeta:
    """Создать новый RunMeta с заполненными полями окружения и git."""
    git = capture_git_info(repo_dir)
    env = capture_environment()
    return RunMeta(
        run_name=run_name,
        mode=mode,
        commit_sha=git["sha"],
        commit_message=git["message"],
        branch=git["branch"],
        python_version=env["python"],
        torch_version=env["torch"],
        polars_version=env["polars"],
        seed=seed,
        args=args or {},
        started_at=datetime.now(UTC).isoformat(),
    )
