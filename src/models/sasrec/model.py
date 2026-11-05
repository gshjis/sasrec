"""Модель SASRec (Self-Attentive Sequential Recommendation) на PyTorch."""

from __future__ import annotations

import torch
from torch import nn


class SASRecBlock(nn.Module):
    """Один блок SASRec: Self-Attention + Feed-Forward."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # Self-attention с причинной маской
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.ln1(x + self.dropout(attn_out))

        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.ln2(x + self.dropout(ff_out))
        return x


class SASRec(nn.Module):
    """SASRec: Self-Attentive Sequential Recommendation.

    Задача: next-item prediction. На вход — префикс сессии (исключая последний элемент),
    на выходе — эмбеддинг пользователя, который затем скалярно умножается на эмбеддинги
    всех айтемов для full-sort scoring.
    """

    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 50,
        embed_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        # Эмбеддинги айтемов и позиций (индекс 0 зарезервирован под PAD)
        self.item_emb = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [SASRecBlock(embed_dim, num_heads, dropout) for _ in range(num_blocks)]
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Прямой проход.

        Args:
            input_ids: [batch, seq_len] токены айтемов (включая PAD=0).
            attention_mask: [batch, seq_len] 1 для реальных токенов, 0 для PAD.

        Returns:
            user_emb: [batch, embed_dim] эмбеддинг пользователя (последняя
                не-PAD позиция после всех блоков).
        """
        batch_size, seq_len = input_ids.shape

        # Позиционные индексы (0..seq_len-1)
        positions = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand_as(input_ids)
        )

        x = self.item_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)

        # Причинная маска: позиция i не видит позиции > i
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )  # [seq_len, seq_len]

        # PAD-маска: True где PAD (запрещено внимание к позиции)
        pad_blocked = ~attention_mask.bool().unsqueeze(1)  # [batch, 1, seq_len]
        # combined: [batch, seq_len, seq_len] — True где запрещено внимание
        combined_mask = causal_mask.unsqueeze(0) | pad_blocked  # broadcast

        for block in self.blocks:
            x = block(x, combined_mask)

        x = self.layer_norm(x)  # [batch, seq_len, embed_dim]

        # Берём эмбеддинг последнего реального токена (не PAD) для каждого примера
        last_idx = attention_mask.sum(dim=1) - 1  # [batch]
        user_emb = x[torch.arange(batch_size), last_idx]  # [batch, embed_dim]

        return user_emb

    def score_all_items(self, user_emb: torch.Tensor) -> torch.Tensor:
        """Скор всех айтемов каталога для каждого пользователя.

        Args:
            user_emb: [batch, embed_dim]

        Returns:
            logits: [batch, num_items] — скалярное произведение с эмбеддингами
                всех айтемов.
        """
        item_embeddings = self.item_emb.weight  # [num_items, embed_dim]
        logits = user_emb @ item_embeddings.T  # [batch, num_items]
        return logits
