"""Experiment configuration.

Configs are plain dataclasses that can be loaded from / dumped to YAML
(see ``configs/base.yaml``). Paths are resolved relative to ``data_root``,
which defaults to ``$RECSYS_DATA_ROOT`` or ``./data`` (on Colab point it to
a Google Drive folder so that preprocessed data and checkpoints survive
session restarts).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    dataset_size: str = "50m"
    # Global Temporal Split (Yambda paper): train | gap | test, plus a val day before the gap.
    test_days: int = 1
    val_days: int = 1
    gap_minutes: int = 30
    listen_plus_threshold: float = 0.5
    min_item_count: int = 5
    max_vocab_size: int = 200_000
    max_seq_len: int = 256
    window_stride: int = 128
    session_gap_minutes: int = 30


@dataclass
class ModelConfig:
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    use_feedback_tokens: bool = True
    n_ratio_bins: int = 12
    n_length_bins: int = 7


@dataclass
class TrainConfig:
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 5
    warmup_steps: int = 500
    n_negatives: int = 8192
    amp: bool = True
    eval_every: int = 1000


@dataclass
class RLConfig:
    method: str = "rloo"  # rloo | grpo | offpolicy_reinforce | dpo
    reward: str = "ratio"  # ratio | listen_time | like_ratio | debiased_ratio | ...
    group_size: int = 16
    temperature: float = 1.0
    kl_beta: float = 0.1
    sft_alpha: float = 0.0
    lr: float = 5e-5
    steps: int = 3000
    rollout_len: int = 1


@dataclass
class Config:
    seed: int = 42
    run_name: str = "debug"
    data_root: str = field(default_factory=lambda: os.environ.get("RECSYS_DATA_ROOT", "data"))
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    rl: RLConfig = field(default_factory=RLConfig)

    @property
    def raw_dir(self) -> Path:
        return Path(self.data_root) / "raw"

    @property
    def processed_dir(self) -> Path:
        return Path(self.data_root) / "processed" / self.data.dataset_size

    @property
    def run_dir(self) -> Path:
        return Path(self.data_root) / "runs" / self.run_name


def _from_dict(cls: type, values: dict[str, Any]) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise KeyError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in values.items():
        default = getattr(cls(), name)
        kwargs[name] = _from_dict(type(default), value) if is_dataclass(default) else value
    return cls(**kwargs)


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load a YAML config; top-level ``overrides`` are applied on top."""
    values: dict[str, Any] = {}
    if path is not None:
        values = yaml.safe_load(Path(path).read_text()) or {}
    values.update(overrides)
    return _from_dict(Config, values)


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(asdict(cfg), sort_keys=False, allow_unicode=True))
