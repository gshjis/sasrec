"""Сохранение и загрузка чекпоинтов модели (best / last)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: Path | str, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None,
                    epoch: int, metrics: dict[str, Any] | None = None,
                    config: dict[str, Any] | None = None) -> None:
    """Сохранить чекпоинт модели."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics or {},
        "config": config or {},
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(state, p)


def load_checkpoint(path: Path | str, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    device: str = "cpu") -> tuple[int, dict[str, Any]]:
    """Загрузить чекпоинт. Возвращает (epoch, metrics)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    checkpoint = torch.load(p, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})
    return epoch, metrics


class CheckpointManager:
    """Отслеживает лучший чекпоинт по метрике и хранит последние N."""

    def __init__(self, dir_path: Path | str, keep_last: int = 3,
                 monitor: str = "val_loss", mode: str = "min") -> None:
        self.dir = Path(dir_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.monitor = monitor
        self.mode = mode  # "min" или "max"
        self.best_value: float | None = None
        self.best_path: Path | None = None
        self.saved_paths: list[Path] = []

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < self.best_value
        return value > self.best_value

    def save(self, epoch: int, model: torch.nn.Module,
             optimizer: torch.optim.Optimizer | None,
             metrics: dict[str, Any],
             config: dict[str, Any] | None = None) -> dict[str, Path]:
        """Сохранить last и, при улучшении, best чекпоинт."""
        saved: dict[str, Path] = {}

        last_path = self.dir / "last.pt"
        save_checkpoint(last_path, model, optimizer, epoch, metrics, config)
        saved["last"] = last_path
        self.saved_paths.append(last_path)

        monitor_value = metrics.get(self.monitor)
        if monitor_value is not None and self._is_better(float(monitor_value)):
            self.best_value = float(monitor_value)
            best_path = self.dir / "best.pt"
            save_checkpoint(best_path, model, optimizer, epoch, metrics, config)
            saved["best"] = best_path

        # Очистка старых чекпоинтов, оставляем keep_last (кроме best/last)
        if len(self.saved_paths) > self.keep_last:
            old = self.saved_paths.pop(0)
            if old.exists() and old.name not in ("best.pt", "last.pt"):
                old.unlink(missing_ok=True)

        return saved
