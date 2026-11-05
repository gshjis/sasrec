from __future__ import annotations

from pathlib import Path

import polars as pl
from huggingface_hub import hf_hub_download

from .config import HfConfig


class HfYambdaLoader:
    def __init__(self, config: HfConfig) -> None:
        self.config = config
        self._downloaded_path: Path | None = None

    def download(self) -> Path:
        download_dir = Path(self.config.download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)

        local_path = hf_hub_download(
            repo_id=self.config.repo_id,
            filename=self.config.filename,
            repo_type="dataset",
            local_dir=download_dir,
        )

        self._downloaded_path = Path(local_path)
        return self._downloaded_path

    def load(self) -> pl.DataFrame:
        if self._downloaded_path is None:
            raise RuntimeError("Требуется вызов download() для скачивания данных.")

        return pl.read_parquet(self._downloaded_path)

    @property
    def downloaded_path(self) -> Path | None:
        return self._downloaded_path
