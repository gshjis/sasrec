from __future__ import annotations

from typing import Any

import polars as pl

UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"


class Tokenizer:
    def __init__(self, item_column: str = "item_id") -> None:
        self.item_column = item_column
        self.item_to_token: dict[Any, int] = {}
        self.token_to_item: dict[int, Any] = {}
        self._fitted: bool = False
        self.vocab_size: int = 0

    def fit(self, train_dataset: pl.DataFrame) -> Tokenizer:

        unique_items = train_dataset[self.item_column].unique().to_list()

        self.item_to_token = {}
        self.token_to_item = {}

        # Специальные токены
        self.item_to_token[PAD_TOKEN] = 0
        self.item_to_token[UNK_TOKEN] = 1
        self.token_to_item[0] = PAD_TOKEN
        self.token_to_item[1] = UNK_TOKEN

        # Обычные айтемы начинаются с 2
        next_token = 2
        for item in unique_items:
            if item not in self.item_to_token:
                self.item_to_token[item] = next_token
                self.token_to_item[next_token] = item
                next_token += 1

        self.vocab_size = len(self.item_to_token)
        self._fitted = True
        return self

    def transform(self, dataset: pl.DataFrame) -> pl.DataFrame:

        if not self._fitted:
            raise RuntimeError("Токенизатор не обучен. Сначала вызовите fit().")

        unk_id = self.item_to_token[UNK_TOKEN]

        # Маппинг только для реальных айтемов (специальные токены не встречаются в данных)
        item_mapping = {
            k: v for k, v in self.item_to_token.items() if isinstance(k, int)
        }

        df = (
            dataset.with_columns(
                pl.col(self.item_column)
                .replace_strict(item_mapping, default=unk_id, return_dtype=pl.UInt32)
                .alias("token_id")
            )
            .drop(self.item_column)
            .rename({"token_id": self.item_column})
        )

        return df

    def fit_transform(self, train_dataset: pl.DataFrame) -> pl.DataFrame:
        self.fit(train_dataset)
        return self.transform(train_dataset)
