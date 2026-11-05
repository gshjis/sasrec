"""Базовая модель Random — рекомендует случайные айтемы."""

from __future__ import annotations

from typing import Any

import torch


class Random:
    """Random: рекомендует случайные айтемы с равномерным распределением.

    Не имеет обучаемых параметров. score_all_items возвращает одинаковые
    логиты для всех айтемов (эквивалентно случайному ранжированию).
    """

    def __init__(self, num_items: int, seed: int = 42) -> None:
        self.num_items = num_items
        self.seed = seed
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def score_all_items(self, user_emb: torch.Tensor) -> torch.Tensor:
        """Возвращает случайные логиты для всех айтемов.

        Args:
            user_emb: тензор [batch, emb_dim] — игнорируется.

        Returns:
            тензор [batch, num_items] со случайными логитами.
        """
        batch_size = user_emb.shape[0]
        return torch.rand(
            batch_size, self.num_items,
            generator=self.generator,
        )

    def state_dict(self) -> dict[str, Any]:
        return {"num_items": self.num_items, "seed": self.seed}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.num_items = state_dict["num_items"]
        self.seed = state_dict["seed"]
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)
