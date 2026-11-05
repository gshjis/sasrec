from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, Dataset


class BatchFormation(ABC):
    def __init__(
        self,
        batch_size: int = 256,
        max_seq_len: int = 50,
        num_workers: int = 0,
        shuffle: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.shuffle = shuffle

    @abstractmethod
    def create_dataset(
        self,
        parquet_path: str | Path,
        session_column: str = "session_id",
        user_column: str = "uid",
        item_column: str = "item_id",
        sort_column: str = "timestamp",
        **kwargs: Any,
    ) -> Dataset: ...

    @abstractmethod
    def create_collate_fn(self) -> Any: ...

    def create_dataloader(
        self,
        parquet_path: str | Path,
        session_column: str = "session_id",
        user_column: str = "uid",
        item_column: str = "item_id",
        sort_column: str = "timestamp",
        shuffle: bool | None = None,
        **kwargs: Any,
    ) -> DataLoader:
        dataset = self.create_dataset(
            parquet_path=parquet_path,
            session_column=session_column,
            user_column=user_column,
            item_column=item_column,
            sort_column=sort_column,
            **kwargs,
        )
        collate_fn = self.create_collate_fn()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle if shuffle is None else shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
