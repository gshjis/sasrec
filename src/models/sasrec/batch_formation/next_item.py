from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from .base import BatchFormation


class _NextItemSeqDataset(Dataset):
    def __init__(
        self,
        parquet_path: str | Path,
        session_column: str = "session_id",
        user_column: str = "uid",
        item_column: str = "item_id",
        sort_column: str = "timestamp",
        max_seq_len: int = 50,
    ) -> None:
        self.parquet_path = str(parquet_path)
        self.session_column = session_column
        self.user_column = user_column
        self.item_column = item_column
        self.sort_column = sort_column
        self.max_seq_len = max_seq_len

        self._pf = pq.ParquetFile(self.parquet_path)

        index_df = pl.read_parquet(
            self.parquet_path,
            columns=[user_column, session_column, item_column, sort_column],
        ).with_row_index("row_idx")

        self._session_index: dict[tuple[int, int], tuple[int, int]] = {}
        self._session_keys: list[tuple[int, int]] = []

        for key, group in index_df.group_by(
            [user_column, session_column], maintain_order=True
        ):
            uid = int(key[0])
            sid = int(key[1])
            start = int(group["row_idx"].min())  # type: ignore[reportArgumentType]
            end = int(group["row_idx"].max()) + 1  # type: ignore[reportArgumentType]
            self._session_index[(uid, sid)] = (start, end)
            self._session_keys.append((uid, sid))

        self._columns = pq.read_table(
            self.parquet_path,
            columns=[item_column, sort_column],
            memory_map=True,
        )
        self._items = self._columns.column(item_column)
        self._timestamps = self._columns.column(sort_column)

    def __len__(self) -> int:
        return len(self._session_keys)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        uid, session_id = self._session_keys[idx]
        start, end = self._session_index[(uid, session_id)]
        length_rows = end - start

        items = self._items.slice(start, length_rows).to_numpy(zero_copy_only=False)
        timestamps = self._timestamps.slice(start, length_rows).to_numpy(
            zero_copy_only=False
        )

        # Внутри сессии события упорядочиваются по времени.
        order = np.argsort(timestamps, kind="stable")
        item_ids = items[order].astype(np.int64, copy=False).tolist()

        if len(item_ids) < 2:
            input_ids = torch.tensor([], dtype=torch.long)  # type: ignore[call-overload]
            target = torch.tensor(-1, dtype=torch.long)  # type: ignore[call-overload]
            length = torch.tensor(0, dtype=torch.long)  # type: ignore[call-overload]
        else:
            input_seq = item_ids[:-1]
            input_seq = input_seq[-self.max_seq_len :]
            input_ids = torch.tensor(input_seq, dtype=torch.long)  # type: ignore[call-overload]
            target = torch.tensor(item_ids[-1], dtype=torch.long)  # type: ignore[call-overload]
            length = torch.tensor(len(input_ids), dtype=torch.long)  # type: ignore[call-overload]

        return {
            "input_ids": input_ids,
            "target": target,
            "length": length,
        }


def _next_item_collate_fn(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    valid = [item for item in batch if item["length"].item() > 0]
    if not valid:
        empty = torch.zeros((0,), dtype=torch.long)
        return {
            "input_ids": torch.zeros((0, 0), dtype=torch.long),
            "attention_mask": torch.zeros((0, 0), dtype=torch.long),
            "targets": empty,
            "lengths": empty,
        }

    max_len = max(item["length"].item() for item in valid)
    batch_size = len(valid)

    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    targets = torch.zeros((batch_size,), dtype=torch.long)
    lengths = torch.zeros((batch_size,), dtype=torch.long)

    for i, item in enumerate(valid):
        seq = item["input_ids"]
        seq_len = item["length"].item()
        input_ids[i, :seq_len] = seq
        attention_mask[i, :seq_len] = 1
        targets[i] = item["target"]
        lengths[i] = seq_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "targets": targets,
        "lengths": lengths,
    }


class NextItemBatchFormation(BatchFormation):
    def create_dataset(
        self,
        parquet_path: str | Path,
        session_column: str = "session_id",
        user_column: str = "uid",
        item_column: str = "item_id",
        sort_column: str = "timestamp",
        **kwargs: Any,
    ) -> Dataset:
        return _NextItemSeqDataset(
            parquet_path=parquet_path,
            session_column=session_column,
            user_column=user_column,
            item_column=item_column,
            sort_column=sort_column,
            max_seq_len=self.max_seq_len,
        )

    def create_collate_fn(self) -> Any:
        return _next_item_collate_fn
