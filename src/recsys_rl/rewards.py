"""Reward functions on top of reward-model predictions (stages 4-6, plan §7).

``Reward`` maps the RM probabilities of ``G`` candidates per context (``RewardModel.score``,
``[E, B, G, H]``) to a scalar reward ``[B, G]``. Predictions are ensemble means.

=========================  ====================================================  ========
name                       reward                                                plan §7
=========================  ====================================================  ========
``ratio``                  E[played ratio]                                       R1
``listen_time``            E[ratio] * min(track length, cap), minutes            R2
``like_ratio``             p(like) + E[ratio] / 10 (Matveev et al., 2026)        R3
``debiased_ratio``         E[ratio] - train mean ratio of the track-length bin   R4
``item_debiased_ratio``    E[ratio] - train mean ratio of the track (shrunk)     R4
``continue``               p(the session continues after the track)              R6
=========================  ====================================================  ========

R5 penalties are added to any of them: ``- repeat_penalty * 1{candidate in the context}
- dislike_penalty * p(dislike) - uncertainty_penalty * std over the ensemble of E[ratio]``.

Track lengths and mean ratios come from train listens only (``items`` and the train rows of the
store), as the leak rules require.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import polars as pl
import torch
from torch import nn

from recsys_rl.config import DataConfig, RLConfig
from recsys_rl.data import N_SPECIAL_TOKENS, EventStore
from recsys_rl.evaluate import EvalSet
from recsys_rl.model import to_tensors
from recsys_rl.reward_model import (
    HEAD_INDEX,
    RewardModel,
    context_repeat_counts,
    item_length_bins,
)

REWARDS = (
    "ratio",
    "listen_time",
    "like_ratio",
    "debiased_ratio",
    "item_debiased_ratio",
    "continue",
)


def ratio_by_length_bin(store: EventStore, train_end: int, n_bins: int) -> np.ndarray:
    """``[n_bins]`` mean clipped played ratio (0..1) of train listens per track-length bin
    (the global train mean for empty bins)."""
    train = np.asarray(store.columns["timestamp"]) < train_end
    ratio = np.minimum(np.asarray(store.columns["played_ratio_pct"])[train], 100) / 100
    bins = np.asarray(store.columns["length_bin"])[train].astype(np.int64)
    sums = np.bincount(bins, weights=ratio, minlength=n_bins)
    counts = np.bincount(bins, minlength=n_bins)
    return np.where(counts > 0, sums / np.maximum(counts, 1), ratio.mean())


def item_mean_ratio(items: pl.DataFrame, global_mean: float, prior: float = 10.0) -> np.ndarray:
    """``[n_items]`` train mean ratio of every vocabulary item shrunk towards ``global_mean``
    with ``prior`` pseudo-listens (rare tracks are noisy); specials get ``global_mean``."""
    out = np.full(N_SPECIAL_TOKENS + len(items), global_mean)
    count = items["count"].to_numpy().astype(np.float64)
    mean = items["mean_ratio"].fill_null(global_mean).to_numpy()
    out[items["idx"].to_numpy()] = (count * mean + prior * global_mean) / (count + prior)
    return out


class Reward(nn.Module):
    def __init__(
        self,
        cfg: RLConfig,
        item_length: torch.Tensor,
        item_length_bin: torch.Tensor,
        length_bin_ratio: torch.Tensor,
        item_ratio: torch.Tensor,
    ):
        """``item_length`` (seconds, already capped), ``item_length_bin``, ``item_ratio`` are
        ``[n_items]`` by vocabulary index; ``length_bin_ratio`` is ``[n_length_bins]``."""
        super().__init__()
        if cfg.reward not in REWARDS:
            raise ValueError(f"unknown reward {cfg.reward!r}, expected one of {REWARDS}")
        self.cfg = cfg
        self.register_buffer("item_length", item_length.float())
        self.register_buffer("item_length_bin", item_length_bin.long())
        self.register_buffer("length_bin_ratio", length_bin_ratio.float())
        self.register_buffer("item_ratio", item_ratio.float())

    @classmethod
    def build(
        cls,
        cfg: RLConfig,
        items: pl.DataFrame,
        store: EventStore,
        train_end: int,
        data_cfg: DataConfig,
    ) -> Reward:
        length = np.zeros(N_SPECIAL_TOKENS + len(items))
        track_length = items["track_length"]
        length[items["idx"].to_numpy()] = track_length.fill_null(track_length.median()).to_numpy()
        by_bin = ratio_by_length_bin(store, train_end, len(data_cfg.length_bin_edges) + 1)
        train = np.asarray(store.columns["timestamp"]) < train_end
        global_mean = float(
            (np.minimum(np.asarray(store.columns["played_ratio_pct"])[train], 100) / 100).mean()
        )
        return cls(
            cfg,
            torch.from_numpy(np.minimum(length, cfg.listen_time_cap)),
            item_length_bins(items, data_cfg),
            torch.from_numpy(by_bin),
            torch.from_numpy(item_mean_ratio(items, global_mean)),
        )

    def forward(
        self, probs: torch.Tensor, candidates: torch.Tensor, repeat_count: torch.Tensor
    ) -> torch.Tensor:
        """``[B, G]`` reward from ``probs`` ``[E, B, G, H]`` (``RewardModel.score``) of
        ``candidates`` ``[B, G]`` that occur ``repeat_count`` times in their context."""
        mean = probs.mean(0)
        ratio = mean[..., HEAD_INDEX["ratio"]]
        name = self.cfg.reward
        if name == "ratio":
            reward = ratio
        elif name == "listen_time":
            reward = ratio * self.item_length[candidates] / 60
        elif name == "like_ratio":
            reward = mean[..., HEAD_INDEX["like"]] + ratio / 10
        elif name == "debiased_ratio":
            reward = ratio - self.length_bin_ratio[self.item_length_bin[candidates]]
        elif name == "item_debiased_ratio":
            reward = ratio - self.item_ratio[candidates]
        else:  # continue
            reward = mean[..., HEAD_INDEX["continue"]]
        cfg = self.cfg
        if cfg.repeat_penalty:
            reward = reward - cfg.repeat_penalty * (repeat_count > 0).float()
        if cfg.dislike_penalty:
            reward = reward - cfg.dislike_penalty * mean[..., HEAD_INDEX["dislike"]]
        if cfg.uncertainty_penalty and probs.shape[0] > 1:
            std = probs[..., HEAD_INDEX["ratio"]].std(0)
            reward = reward - cfg.uncertainty_penalty * std
        return reward


@torch.no_grad()
def score_recommendations(
    rm: RewardModel,
    reward: Reward,
    store: EventStore,
    eval_set: EvalSet,
    recs: np.ndarray,
    batch_size: int = 128,
    amp_dtype: torch.dtype | None = None,
) -> Mapping[str, np.ndarray]:
    """RM view of top-K lists ``recs`` (``[n_users, k]`` vocabulary indices) after the frozen
    contexts of ``eval_set``: ``reward``, ensemble-mean ``ratio`` / ``like`` and ``ratio_std``,
    each ``[n_users, k]``."""
    rm.eval()
    device = next(rm.parameters()).device
    lengths = eval_set.context_ends - eval_set.context_starts
    out: dict[str, list[np.ndarray]] = {"reward": [], "ratio": [], "like": [], "ratio_std": []}
    for b in range(0, eval_set.n_users, batch_size):
        sl = slice(b, b + batch_size)
        batch = to_tensors(
            store.gather(eval_set.context_starts[sl], lengths[sl], rm.max_seq_len, eval_set.cutoff),
            device,
        )
        cand = torch.from_numpy(np.ascontiguousarray(recs[sl])).long().to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            probs = rm.score(batch, cand).float()
        repeat = context_repeat_counts(batch["item_idx"], batch["mask"], cand)
        ratio = probs[..., HEAD_INDEX["ratio"]]
        values = {
            "reward": reward(probs, cand, repeat),
            "ratio": ratio.mean(0),
            "like": probs[..., HEAD_INDEX["like"]].mean(0),
            "ratio_std": ratio.std(0) if len(probs) > 1 else torch.zeros_like(ratio[0]),
        }
        for name, value in values.items():
            out[name].append(value.cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in out.items()}
