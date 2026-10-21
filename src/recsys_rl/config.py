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
    # Feature bins: bin i holds edges[i-1] < x <= edges[i] (len(edges) + 1 bins); chosen in the EDA.
    # Played ratio is bimodal (53% full plays, 24% <= 5%), so the edges follow its shape.
    ratio_bin_edges: list[int] = field(default_factory=lambda: [0, 5, 10, 25, 49, 75, 99, 100])
    # Roughly train quantiles of track length in seconds (median 200 s, 5-95%: 125-310 s).
    length_bin_edges: list[int] = field(default_factory=lambda: [120, 150, 180, 210, 240, 300])


@dataclass
class ModelConfig:
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    use_feedback_tokens: bool = True
    n_ratio_bins: int = 9  # len(DataConfig.ratio_bin_edges) + 1
    n_length_bins: int = 7  # len(DataConfig.length_bin_edges) + 1


@dataclass
class TrainConfig:
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 5
    warmup_steps: int = 500
    n_negatives: int = 8192
    grad_clip: float = 1.0
    amp: bool = True
    # Next-item target: "all" (every listen, the task statement) or "listen_plus"
    # (only Listen+ targets contribute to the loss, next-positive ablation).
    target: str = "all"
    eval_every: int = 1000
    patience: int = 5  # early stopping: evaluations without a val NDCG@10 improvement
    max_steps: int = 0  # 0 = run all epochs; > 0 caps the run (smoke tests)


@dataclass
class RewardModelConfig:
    # Encoder initialised from runs/<base_run>/best.pt (its ModelConfig is used); null = random
    # init from Config.model (the gold RM, to be less correlated with the proxy).
    base_run: str | None = "base"
    n_ensemble: int = 4  # MLP heads on a shared encoder; their spread = epistemic uncertainty
    head_hidden: int = 256
    audio_embeddings: bool = False  # needs raw/embeddings.parquet (yambda-download --embeddings)
    user_fraction: float = 1.0  # share of train users whose windows are used (seeded subsample)
    bootstrap: bool = True  # Poisson(1) weight per (window, ensemble member)
    skip_threshold: float = 0.3  # skip target: played ratio below it
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 1
    warmup_steps: int = 200
    grad_clip: float = 1.0
    amp: bool = True
    eval_every: int = 1000
    patience: int = 3  # early stopping: evaluations without a val loss improvement
    max_steps: int = 0  # 0 = run all epochs; > 0 caps the run (smoke tests)


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
    rm: RewardModelConfig = field(default_factory=RewardModelConfig)
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


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            value = _merge(out[key], value)
        out[key] = value
    return out


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load a YAML config; ``overrides`` are merged on top (nested dicts key by key, so
    ``train={"lr": 3e-4}`` keeps the other ``train`` values of the file)."""
    values: dict[str, Any] = {}
    if path is not None:
        values = yaml.safe_load(Path(path).read_text()) or {}
    return _from_dict(Config, _merge(values, overrides))


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """``["train.lr=3e-4", "run_name=x"]`` -> nested dict; values are parsed as YAML."""
    out: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} is not key=value")
        *parents, leaf = key.split(".")
        node = out
        for name in parents:
            node = node.setdefault(name, {})
        value = yaml.safe_load(raw)
        if isinstance(value, str):
            try:
                value = float(value)  # PyYAML (YAML 1.1) reads "3e-4" as a string
            except ValueError:
                pass
        node[leaf] = value
    return out


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(asdict(cfg), sort_keys=False, allow_unicode=True))
