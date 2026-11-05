"""Единая конфигурация эксперимента (dataclass)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from paths import DATA_DIR, LOGS_DIR, PROCESSED_DIR, RUNS_DIR

LogQMode = Literal["plain", "standard", "improved"]
RunMode = Literal["smoke", "full", "ablation"]


@dataclass(frozen=True)
class DataConfig:
    """Конфигурация подготовки данных."""
    raw_path: str = str(DATA_DIR / "raw" / "yambda_50m_listens.parquet")
    processed_dir: str = str(PROCESSED_DIR)
    interactions: int = 3_000_000
    gap_seconds: int = 3600
    train_share: float = 0.70
    val_share: float = 0.15
    test_share: float = 0.15
    min_session_len: int = 10
    min_played_ratio_pct: float = 50.0
    gap_std_threshold: float = 2.0
    time_outlier_multiplier: float = 1.5
    seed: int = 42

    def __post_init__(self) -> None:
        total = self.train_share + self.val_share + self.test_share
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"train_share + val_share + test_share должно быть 1.0, получено {total}"
            )


@dataclass(frozen=True)
class ModelConfig:
    """Конфигурация модели SASRec."""
    embed_dim: int = 64
    num_blocks: int = 2
    num_heads: int = 1
    dropout: float = 0.5
    max_seq_len: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    """Конфигурация обучения."""
    epochs: int = 5
    batch_size: int = 256
    temperature: float = 1.0
    logq_mode: LogQMode = "plain"
    seed: int = 42
    device: str = "auto"
    early_stopping_patience: int | None = None
    save_checkpoints: bool = True


@dataclass(frozen=True)
class EvalConfig:
    """Конфигурация оценки."""
    ks: tuple[int, ...] = (10, 50, 100)
    pareto_coverage_share: float = 0.8


@dataclass(frozen=True)
class ExperimentConfig:
    """Главный конфиг эксперимента."""
    run_name: str = "default"
    mode: RunMode = "full"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output_dir: str = str(RUNS_DIR)
    log_dir: str = str(LOGS_DIR)
    hf_repo: str = "yandex/yambda"
    hf_file: str = "flat/50m/listens.parquet"
    with_test: bool = True
    dry_run: bool = False
    strict_device: bool = False

    @classmethod
    def smoke(cls) -> ExperimentConfig:
        """Предустановка для быстрого smoke-прогона."""
        return cls(
            run_name="smoke",
            mode="smoke",
            data=DataConfig(interactions=100_000, seed=42),
            train=TrainConfig(epochs=2, batch_size=256, temperature=1.0, seed=42),
            with_test=False,
            dry_run=False,
        )

    @classmethod
    def full(cls) -> ExperimentConfig:
        """Предустановка для полного прогона."""
        return cls(
            run_name="full",
            mode="full",
            data=DataConfig(interactions=0, seed=42),
            train=TrainConfig(epochs=10, batch_size=256, temperature=1.0, seed=42),
            with_test=True,
        )

    @classmethod
    def ablation(cls) -> ExperimentConfig:
        """Предустановка для абляционных исследований."""
        return cls(
            run_name="ablation",
            mode="ablation",
            data=DataConfig(interactions=3_000_000, seed=42),
            train=TrainConfig(epochs=5, batch_size=256, temperature=1.0, seed=42),
            with_test=True,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentConfig:
        data = DataConfig(**d.get("data", {}))
        model = ModelConfig(**d.get("model", {}))
        train = TrainConfig(**d.get("train", {}))
        eval_ = EvalConfig(**d.get("eval", {}))
        return cls(
            run_name=d.get("run_name", "default"),
            mode=d.get("mode", "full"),
            data=data,
            model=model,
            train=train,
            eval=eval_,
            output_dir=d.get("output_dir", str(RUNS_DIR)),
            log_dir=d.get("log_dir", str(LOGS_DIR)),
            hf_repo=d.get("hf_repo", "yandex/yambda"),
            hf_file=d.get("hf_file", "flat/50m/listens.parquet"),
            with_test=d.get("with_test", True),
            dry_run=d.get("dry_run", False),
            strict_device=d.get("strict_device", False),
        )
