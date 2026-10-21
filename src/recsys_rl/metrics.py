"""Evaluation metrics (stage 1).

Relevance: Recall@K, NDCG@K, MAP@K on Listen+ / Like targets.
Beyond-accuracy: Coverage@K, Gini, mean pairwise Jaccard of top-K lists across users,
repeat rate (share of items from user history), mean popularity, mean track length, ILD.
Reward-model quality (stage 3): ROC-AUC, average precision (PR-AUC), calibration bins and ECE.

Conventions:
- ``recs`` is an ``int`` array ``[n_users, K_max]`` of item ids, best first, unique within a row.
- ``targets`` is a list of 1-D item-id arrays aligned with the rows of ``recs`` (the Listen+
  items of the evaluation day). Duplicates are allowed; the main metrics use the set of items.
- Users without targets must be dropped by the caller (the Yambda protocol averages over
  users that have at least one target).
- Item ids only need to be consistent between ``recs``, ``targets`` and the lookup arrays
  (``values``, ``embeddings``); OOV targets are simply never hit.

Main relevance definitions (per user, then averaged over users), with ``G`` the target set:
- Recall@K = |top-K ∩ G| / |G|
- NDCG@K = DCG@K / IDCG@K, binary gains, IDCG over min(|G|, K) hits
- MAP@K = sum_{i <= K, rec_i in G} Precision@i / min(|G|, K)

``yambda_paper_metrics`` reproduces the formulas behind the published Yambda numbers (used
for the SASRec sanity check only); see its docstring.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import rankdata


def _check(recs: np.ndarray, k: int) -> np.ndarray:
    recs = np.asarray(recs)
    if recs.ndim != 2:
        raise ValueError(f"recs must be 2-D [n_users, K], got shape {recs.shape}")
    if not 0 < k <= recs.shape[1]:
        raise ValueError(f"k={k} is out of range for recs with {recs.shape[1]} columns")
    top = np.sort(recs[:, :k], axis=1)
    if (top[:, 1:] == top[:, :-1]).any():
        raise ValueError("recs must not contain duplicate items within a row")
    return recs[:, :k]


def _discounts(k: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(2, k + 2))


def hit_mask(recs: np.ndarray, targets: Sequence[np.ndarray]) -> np.ndarray:
    """Boolean ``[n_users, K]`` mask: ``recs[u, i]`` is among the targets of user ``u``."""
    recs = _check(recs, np.shape(recs)[1])
    if len(targets) != recs.shape[0]:
        raise ValueError(f"{len(targets)} target lists for {recs.shape[0]} users")
    return np.stack([np.isin(row, t) for row, t in zip(recs, targets, strict=True)])


def relevance_metrics(
    recs: np.ndarray, targets: Sequence[np.ndarray], ks: Sequence[int] = (10, 100)
) -> dict[str, float]:
    """Recall@K, NDCG@K and MAP@K averaged over users (see module docstring)."""
    recs = _check(recs, max(ks))
    n_targets = np.array([len(np.unique(t)) for t in targets], dtype=np.float64)
    if (n_targets == 0).any():
        raise ValueError("every user must have at least one target")
    hits = hit_mask(recs, targets).astype(np.float64)
    discounts = _discounts(recs.shape[1])
    ranks = np.arange(1, recs.shape[1] + 1)
    out = {}
    for k in ks:
        h = hits[:, :k]
        n_ideal = np.minimum(n_targets, k).astype(int)
        idcg = np.cumsum(discounts[:k])[n_ideal - 1]
        precision_at_i = np.cumsum(h, axis=1) / ranks[:k]
        out[f"recall@{k}"] = float(np.mean(h.sum(1) / n_targets))
        out[f"ndcg@{k}"] = float(np.mean((h * discounts[:k]).sum(1) / idcg))
        out[f"map@{k}"] = float(np.mean((precision_at_i * h).sum(1) / n_ideal))
    return out


def yambda_paper_metrics(
    recs: np.ndarray, targets: Sequence[np.ndarray], ks: Sequence[int] = (10, 100)
) -> dict[str, float]:
    """Recall/NDCG as computed for the numbers published in the Yambda paper.

    The reference code (``benchmarks/yambda/evaluation/metrics.py`` on the HF hub) documents
    two bugs that affected the paper: Recall divided hits by the number of target *events*
    (duplicates included), and NDCG was pooled as ``sum(DCG) / sum(IDCG)`` instead of a mean of
    per-user ratios. Use only to compare against the published SASRec baseline.
    """
    recs = _check(recs, max(ks))
    n_events = np.array([len(t) for t in targets], dtype=np.float64)
    if (n_events == 0).any():
        raise ValueError("every user must have at least one target")
    hits = hit_mask(recs, targets).astype(np.float64)
    discounts = _discounts(recs.shape[1])
    out = {}
    for k in ks:
        h = hits[:, :k]
        idcg = np.cumsum(discounts[:k])[np.minimum(n_events, k).astype(int) - 1]
        out[f"recall@{k}"] = float(np.mean(h.sum(1) / n_events))
        out[f"ndcg@{k}"] = float((h * discounts[:k]).sum() / idcg.sum())
    return out


def coverage(recs: np.ndarray, k: int, n_items: int) -> float:
    """Share of the catalogue (``n_items``) that appears in at least one top-K list."""
    return len(np.unique(_check(recs, k))) / n_items


def gini(recs: np.ndarray, k: int, n_items: int) -> float:
    """Gini index of item exposure in top-K lists over the whole catalogue (0 = uniform)."""
    top = _check(recs, k)
    _, counts = np.unique(top, return_counts=True)
    if len(counts) > n_items:
        raise ValueError(f"{len(counts)} distinct items recommended but n_items={n_items}")
    x = np.sort(np.concatenate([np.zeros(n_items - len(counts)), counts]))
    i = np.arange(1, n_items + 1)
    return float(((2 * i - n_items - 1) * x).sum() / (n_items * x.sum()))


def mean_pairwise_jaccard(recs: np.ndarray, k: int, n_pairs: int = 10_000, seed: int = 0) -> float:
    """Mean Jaccard similarity of top-K lists over random pairs of distinct users.

    All pairs are used when there are at most ``n_pairs`` of them; otherwise pairs are sampled
    (10k users give 5e7 pairs, too many to enumerate).
    """
    top = _check(recs, k)
    n = top.shape[0]
    if n < 2:
        raise ValueError("need at least two users")
    if n * (n - 1) // 2 <= n_pairs:
        a, b = np.triu_indices(n, k=1)
    else:
        rng = np.random.default_rng(seed)
        a = rng.integers(0, n, n_pairs)
        b = (a + rng.integers(1, n, n_pairs)) % n  # b != a
    sorted_top = np.sort(top, axis=1)
    inter = np.array(
        [
            len(np.intersect1d(sorted_top[i], sorted_top[j], assume_unique=True))
            for i, j in zip(a, b, strict=True)
        ]
    )
    return float(np.mean(inter / (2 * k - inter)))


def repeat_rate(recs: np.ndarray, histories: Sequence[np.ndarray], k: int) -> float:
    """Share of recommended items that the user has already interacted with."""
    top = _check(recs, k)
    if len(histories) != top.shape[0]:
        raise ValueError(f"{len(histories)} histories for {top.shape[0]} users")
    return float(np.mean([np.isin(row, h).mean() for row, h in zip(top, histories, strict=True)]))


def mean_item_value(recs: np.ndarray, values: np.ndarray, k: int) -> float:
    """Mean of a per-item statistic (train popularity, track length, ...) over top-K items.

    ``values`` is indexed by item id; NaN entries (unknown items) are ignored.
    """
    return float(np.nanmean(np.asarray(values, dtype=np.float64)[_check(recs, k)]))


def ild(recs: np.ndarray, embeddings: np.ndarray, k: int, batch_size: int = 1024) -> float:
    """Intra-list diversity: mean pairwise cosine distance within a top-K list.

    ``embeddings`` is ``[n_items, d]`` indexed by item id (e.g. Yambda audio embeddings).
    """
    top = _check(recs, k)
    if k < 2:
        raise ValueError("ILD needs k >= 2")
    emb = np.asarray(embeddings, dtype=np.float64)
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    per_user = []
    for start in range(0, top.shape[0], batch_size):
        e = emb[top[start : start + batch_size]]  # [B, k, d]
        sim = np.einsum("bid,bjd->bij", e, e)
        off_diag = sim.sum((1, 2)) - np.trace(sim, axis1=1, axis2=2)
        per_user.append(1.0 - off_diag / (k * (k - 1)))
    return float(np.mean(np.concatenate(per_user)))


def _binary(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y, p = np.asarray(y_true, dtype=bool).ravel(), np.asarray(y_score, dtype=np.float64).ravel()
    if len(y) != len(p):
        raise ValueError(f"{len(y)} labels for {len(p)} scores")
    return y, p


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney statistic (ties count 1/2); NaN without both classes."""
    y, p = _binary(y_true, y_score)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(p)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """PR-AUC as average precision, sum over thresholds of (R_n - R_{n-1}) * P_n
    (tied scores form one threshold, as in scikit-learn); NaN without positives."""
    y, p = _binary(y_true, y_score)
    if not y.any():
        return float("nan")
    order = np.argsort(-p, kind="stable")
    y, p = y[order], p[order]
    last = np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]  # last index of each distinct score
    tp = np.cumsum(y)[last]
    precision = tp / (last + 1)
    recall_step = np.diff(np.r_[0, tp]) / tp[-1]
    return float((recall_step * precision).sum())


def calibration_bins(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10
) -> dict[str, np.ndarray]:
    """Reliability diagram data: equal-width bins of the prediction in [0, 1].

    Returns per-bin ``count``, ``mean_pred`` and ``mean_true`` (NaN for empty bins);
    ``y_true`` may be soft (the played ratio).
    """
    y, p = np.asarray(y_true, dtype=np.float64).ravel(), np.asarray(y_pred, np.float64).ravel()
    b = np.clip((p * n_bins).astype(np.int64), 0, n_bins - 1)
    count = np.bincount(b, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_pred = np.bincount(b, weights=p, minlength=n_bins) / count
        mean_true = np.bincount(b, weights=y, minlength=n_bins) / count
    return {"count": count, "mean_pred": mean_pred, "mean_true": mean_true}


def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """Count-weighted mean |mean_pred - mean_true| over the bins of ``calibration_bins``."""
    c = calibration_bins(y_true, y_pred, n_bins)
    nonempty = c["count"] > 0
    gap = np.abs(c["mean_pred"] - c["mean_true"])[nonempty]
    return float((gap * c["count"][nonempty]).sum() / c["count"].sum())
