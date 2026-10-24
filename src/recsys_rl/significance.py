"""Paired bootstrap over users for the chosen policies (plan §3, stage 5).

The val / test differences between policies are small (NDCG@10 +0.002 to +0.008) compared with
the spread over users, so every comparison is a paired test: the same users are resampled with
replacement for both methods, and the statistic is the mean per-user difference.

- ``collect_per_user`` evaluates every method on the frozen contexts of one split and returns
  one row per (method, repeat-filter mode, user): relevance of the top-100 (``ndcg@10``,
  ``recall@100``, ...), and for vocabulary-based methods the KL to ``pi_ref``, the share of
  history tracks, the mean length and log10 train count, ``npcount@10`` (mean train count over
  that of the 10 most popular tracks: nPCOUNT of Petrov & Macdonald, 2024, 1 = MostPop) and the
  number of distinct artists of the top-10, and the proxy / gold RM view of the top-10 (played
  ``ratio``, length-``debiased`` ratio of R4, ``like``, ``continue`` of R6). The RM columns do
  not depend on the reward a method was trained or reranked with, so R1 and R4 methods share
  them.
  Methods: ``pi_ref``, the Repeat baseline (relevance only), RL checkpoints, reranker points
  ``(reward, beta)`` of ``rerank.tilt`` and greedy slate points ``(reward, beta, diversity)``
  of ``rerank.slate_tops``. ``exclude_seen`` rows exist for ``pi_ref`` and the RL
  checkpoints (relevance only; the reranker has no repeat-filter mode).
- ``bootstrap_table`` compares (method, baseline) pairs on every metric with the same bootstrap
  resamples, per split and repeat-filter mode: means, difference, percentile CI, a two-sided
  bootstrap p-value and its Holm adjustment over the pairs of the call for each metric.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from recsys_rl import evaluate, metrics
from recsys_rl.config import Config
from recsys_rl.data import EventStore, PreparedData, prepare
from recsys_rl.evaluate import EvalSet
from recsys_rl.model import load_model
from recsys_rl.rerank import candidate_scores, slate_tops, tilt
from recsys_rl.reward_model import RewardModel
from recsys_rl.rewards import NO_PENALTIES, Reward, score_recommendations
from recsys_rl.rl import item_features, load_frozen_rm, policy_eval
from recsys_rl.train import amp_dtype
from recsys_rl.utils import get_device

KEYS = ("split", "method", "exclude_seen", "user")


def _top10_profile(
    recs: np.ndarray,
    histories: list[np.ndarray],
    item_length: np.ndarray,
    item_log_count: np.ndarray,
    item_artist: np.ndarray | None,
    max_pcount: float,
    rms: Mapping[str, RewardModel],
    reward: Reward,
    store: EventStore,
    eval_set: EvalSet,
    dtype: torch.dtype | None,
) -> dict[str, np.ndarray]:
    """Per-user composition and RM view of the top-10 (vocabulary indices)."""
    top = recs[:, :10]
    out = {
        "repeat@10": np.array([np.isin(r, h).mean() for r, h in zip(top, histories, strict=True)]),
        "length@10": np.nanmean(item_length[top], axis=1),
        "log_count@10": np.nanmean(item_log_count[top], axis=1),
        "npcount@10": np.nanmean(10 ** item_log_count[top], axis=1) / max_pcount,
    }
    if item_artist is not None:
        cat = np.where(item_artist[top] >= 0, item_artist[top], -1 - top)  # as distinct_categories
        out["artists@10"] = 1.0 + (np.diff(np.sort(cat, axis=1), axis=1) != 0).sum(1)
    for name, rm in rms.items():
        scores = score_recommendations(rm, reward, store, eval_set, top, amp_dtype=dtype)
        out[f"{name}_ratio@10"] = scores["ratio"].mean(1)
        out[f"{name}_debiased@10"] = scores["reward"].mean(1)
        out[f"{name}_like@10"] = scores["like"].mean(1)
        out[f"{name}_continue@10"] = scores["continue"].mean(1)
    return out


def collect_per_user(
    cfg: Config,
    prepared: PreparedData | None = None,
    device: torch.device | None = None,
    split: str = "val",
    policies: Mapping[str, str | Path] | None = None,
    rerank: Mapping[str, tuple[str | Mapping[str, Any], float]] | None = None,
    slates: Mapping[str, tuple[str | Mapping[str, Any], float, float]] | None = None,
    n_candidates: int = 100,
    slate_size: int = 10,
    k: int = 100,
    exclude_seen: bool = True,
) -> pl.DataFrame:
    """Per-user metrics of ``pi_ref`` (``runs/<rl.base_run>/best.pt``), Repeat, the
    ``policies`` (name -> checkpoint) and the ``rerank`` points (name -> (reward, beta), where
    the reward is a name or a dict of ``Reward.variant`` changes such as R5 penalties), scored
    by the proxy / gold RMs of ``cfg.rl``, and the greedy ``slates`` (name -> (reward, beta,
    diversity), lists of ``slate_size``); see the module docstring. Relevance is at 10 and
    ``k``; ``exclude_seen = False`` skips the repeat-filter rows."""
    r = cfg.rl
    device = device or get_device()
    prepared = prepared or prepare(cfg)
    store, items, bounds = prepared.store.in_memory(), prepared.items, prepared.bounds
    ref = load_model(Path(cfg.data_root) / "runs" / r.base_run / "best.pt", device).eval()
    seq_len = ref.backbone.config.n_positions
    rms = {"proxy": load_frozen_rm(cfg, r.reward_run, device)}
    if r.gold_run:
        rms["gold"] = load_frozen_rm(cfg, r.gold_run, device)
    reward = Reward.build(r, items, store, bounds.train_end, cfg.data, cfg.raw_dir).to(device)
    profile_reward = reward.variant(reward="debiased_ratio", **NO_PENALTIES)
    dtype = amp_dtype(device, r.amp)

    es = evaluate.build_eval_set(store, bounds, split, seq_len, cfg.data.listen_plus_threshold)
    idx_to_item = evaluate.idx_to_item_array(items)
    histories = es.histories(store, "item_idx")
    item_length, item_log_count, item_artist = item_features(items, ref.n_items, cfg.raw_dir)
    max_pcount = float(np.sort(items["count"].to_numpy())[-10:].mean())  # MostPop by train count
    frames = []

    def add(method: str, exclude: bool, raw_recs: np.ndarray, **extra: np.ndarray) -> None:
        values = metrics.relevance_per_user(raw_recs, es.targets, ks=(10, k)) | extra
        keys = {"split": split, "method": method, "exclude_seen": exclude, "user": es.users}
        frames.append(pl.DataFrame(keys | values))

    def add_vocab(method: str, recs: np.ndarray, kl: np.ndarray) -> None:
        profile = _top10_profile(
            recs, histories, item_length, item_log_count, item_artist, max_pcount, rms,
            profile_reward, store, es, dtype,
        )  # fmt: skip
        add(method, False, idx_to_item[recs], kl_ref=kl, **profile)

    def add_unseen(method: str, model) -> None:
        if not exclude_seen:
            return
        recs = evaluate.recommend_model(
            model, store, es, idx_to_item, k=k, exclude_seen=True, amp_dtype=dtype
        )
        add(method, True, recs)

    add("Repeat", False, evaluate.recommend_repeat(store, es, k=k))
    recs, kl, _ = policy_eval(ref, ref, store, es, k=k, amp_dtype=dtype)
    add_vocab("pi_ref", recs, kl)
    add_unseen("pi_ref", ref)
    for name, path in (policies or {}).items():
        policy = load_model(path, device).eval()
        recs, kl, _ = policy_eval(policy, ref, store, es, k=k, amp_dtype=dtype)
        add_vocab(name, recs, kl)
        add_unseen(name, policy)
        del policy
    by_reward: dict[tuple, list[tuple[str, float, float | None]]] = {}
    points = [(n, spec, beta, None) for n, (spec, beta) in (rerank or {}).items()]
    points += [(n, spec, beta, div) for n, (spec, beta, div) in (slates or {}).items()]
    for name, spec, beta, diversity in points:
        changes = {"reward": spec} if isinstance(spec, str) else dict(spec)
        by_reward.setdefault(tuple(sorted(changes.items())), []).append((name, beta, diversity))
    for changes, group in by_reward.items():  # the RM scores the candidates once per reward
        variant = reward.variant(**dict(changes))
        cands = candidate_scores(
            ref, rms["proxy"], variant, store, es, n_candidates, k, amp_dtype=dtype
        )
        for name, beta, diversity in group:
            if diversity is None:
                top, kl, _ = tilt(cands["logp"], cands["reward"], cands["h_out"], beta, k)
            else:
                top, kl, _ = slate_tops(cands, variant, beta, diversity, slate_size, k)
            add_vocab(name, cands["idx"].gather(1, top).numpy(), kl.numpy())
    return pl.concat(frames, how="diagonal_relaxed")


def _bootstrap_means(
    diffs: np.ndarray, n_boot: int, rng: np.random.Generator, chunk: int = 500
) -> np.ndarray:
    """``[n_vectors, n_boot]`` means of ``diffs`` ``[n_vectors, n_users]`` over the same user
    resamples for every vector."""
    n = diffs.shape[1]
    out = []
    for start in range(0, n_boot, chunk):
        c = min(chunk, n_boot - start)
        idx = rng.integers(0, n, (c, n)) + n * np.arange(c)[:, None]
        counts = np.bincount(idx.ravel(), minlength=c * n).reshape(c, n)  # times each user drawn
        out.append(diffs @ counts.T / n)
    return np.concatenate(out, axis=1)


def holm(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (family-wise error control, uniformly more powerful
    than Bonferroni): the i-th smallest p times ``n - i``, made monotone and capped at 1."""
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p))))
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def bootstrap_table(
    per_user: pl.DataFrame,
    pairs: Sequence[tuple[str, str]],
    metric_names: Sequence[str] | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> pl.DataFrame:
    """One row per (split, repeat-filter mode, pair, metric) where both methods have the metric
    for every user: ``value`` and ``baseline_value`` (means), ``diff``, ``rel_diff``
    (diff / baseline), the ``1 - alpha`` percentile CI of the diff and the two-sided bootstrap
    p-value ``2 * min(P(d* <= 0), P(d* >= 0))`` with the +1 correction (so never 0), and
    ``p_holm``: the Holm adjustment of ``p_value`` over all pairs of this call with the same
    split, repeat-filter mode and metric (the family depends on the pairs passed)."""
    names = metric_names or [c for c in per_user.columns if c not in KEYS]
    rows = []
    for (split, exclude), group in per_user.group_by(
        ["split", "exclude_seen"], maintain_order=True
    ):
        wide = {m: g.sort("user") for (m,), g in group.group_by("method")}
        specs, diffs = [], []
        for method, baseline in pairs:
            if method not in wide or baseline not in wide:
                continue
            a, b = wide[method], wide[baseline]
            if not a["user"].equals(b["user"]):
                raise ValueError(f"{method} and {baseline} are evaluated on different users")
            for name in names:
                if name not in a.columns or name not in b.columns:
                    continue
                x, y = a[name].to_numpy(), b[name].to_numpy()
                if x.dtype == object or np.isnan(x).any() or np.isnan(y).any():
                    continue  # null for this method (e.g. RM columns of Repeat)
                specs.append((method, baseline, name, x.mean(), y.mean()))
                diffs.append(x - y)
        if not diffs:
            continue
        rng = np.random.default_rng(seed)
        boot = _bootstrap_means(np.stack(diffs), n_boot, rng)
        lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2], axis=1)
        below, above = (boot <= 0).sum(1), (boot >= 0).sum(1)
        p = np.minimum(1.0, 2 * (np.minimum(below, above) + 1) / (n_boot + 1))
        for i, (method, baseline, name, value, base_value) in enumerate(specs):
            rows.append(
                {
                    "split": split,
                    "exclude_seen": exclude,
                    "method": method,
                    "baseline": baseline,
                    "metric": name,
                    "value": value,
                    "baseline_value": base_value,
                    "diff": value - base_value,
                    "rel_diff": (value - base_value) / base_value if base_value else np.nan,
                    "ci_low": lo[i],
                    "ci_high": hi[i],
                    "p_value": p[i],
                }
            )
    families: dict[tuple, list[int]] = {}
    for i, row in enumerate(rows):
        families.setdefault((row["split"], row["exclude_seen"], row["metric"]), []).append(i)
    for members in families.values():
        adjusted = holm(np.array([rows[i]["p_value"] for i in members]))
        for i, value in zip(members, adjusted, strict=True):
            rows[i]["p_holm"] = float(value)
    return pl.DataFrame(rows)
