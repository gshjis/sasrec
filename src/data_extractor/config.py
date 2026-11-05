from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HfConfig:
    repo_id: str
    filename: str
    download_dir: str = "./data"
    max_rows: int | None = None
