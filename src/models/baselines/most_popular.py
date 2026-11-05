"""Базовая модель MostPopular — рекомендует самые популярные айтемы."""

from __future__ import annotations

from typing import Any

import torch


class MostPopular:
    """MostPopular: рекомендует айтемы по частоте встречаемости в train.

    Не имеет обучаемых параметров..score_all_items возвращает логиты,
    пропорциональные частоте айтема (или 0 для неизвестных).
    """

    def __init__(self, num_items: int, item_to_token: dict[int, int],
                 item_freq: dict[int, float]) -> None:
        self.num_items = num_items
        self.item_to_token = item_to_token
        self.item_freq = item_freq

        # Логиты для всех токенов: частотаアイтема (или 0 для неизвестных)
        logits = torch.zeros(num_items, dtype=torch.float32)
        for item_id, token in item_to_token.items():
            freq = item_freq.get(item_id, 0.0)
            logits[token] = freq
        self.register_buffer("logits", logits)

    def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        setattr(self, name, tensor)

    def score_all_items(self, user_emb: torch.Tensor) -> torch.Tensor:
        """Возвращает логиты для всех айтемов (одинаковые для всех пользователей).

        Args:
            user_emb: тензор [batch, emb_dim] — игнорируется.

        Returns:
            тензор [batch, num_items] с логитами популярности.
        """
        batch_size = user_emb.shape[0]
        return self.logits.unsqueeze(0).expand(batch_size, -1).clone()

    def state_dict(self) -> dict[str, Any]:
        return {"logits": self.logits, "item_to_token": self.item_to_token,
                "item_freq": self.item_freq, "num_items": self.num_items}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.logits = state_dict["logits"]
        self.item_to_token = state_dict["item_to_token"]
        self.item_freq = state_dict["item_freq"]
        self.num_items = state_dict["num_items"]
