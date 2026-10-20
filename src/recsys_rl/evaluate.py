"""Top-K evaluation on the val / test day (stage 2; reused by later stages).

Protocol (research plan §3): the model and user states are frozen at the start of the
evaluation period. For every user with at least one train listen and at least one Listen+
event in the period, the context is the user's last ``seq_len`` listens before the cutoff
(``train_end`` for val, ``val_end`` for test, see ``data``) and the targets are the raw ids of
the period's Listen+ events. Recommendations are raw item ids, so OOV targets are never hit.

Users are taken from the ``EventStore``, which holds listens only: a user whose train history
has likes/dislikes but no listens is dropped (``data.temporal_split`` would keep them).

Recommenders:
- ``recommend_model``: full softmax of a ``GPT2Recommender`` over the vocabulary;
- ``recommend_most_pop``: the most frequent Listen+ items of the last days before the cutoff;
- ``recommend_repeat``: the user's own most recently listened distinct items, padded with
  MostPop.

``exclude_seen`` is the repeat-filter evaluation mode: items the user listened to before the
cutoff are not recommended. It is a separately reported mode, never applied silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import torch

from recsys_rl import metrics
from recsys_rl.config import DataConfig
from recsys_rl.data import DAY_SECONDS, N_SPECIAL_TOKENS, EventStore, PreparedData, SplitBounds
from recsys_rl.model import GPT2Recommender, to_tensors


@dataclass
class EvalSet:
    name: str
    cutoff: int  # contexts and their feedback features use only events before it
    users: np.ndarray  # user indices in the store
    history_starts: np.ndarray  # first row of each user's history
    context_starts: np.ndarray  # first row of the context (last seq_len listens before cutoff)
    context_ends: np.ndarray  # first row at or after the cutoff
    targets: list[np.ndarray]  # raw ids of the period's Listen+ events, duplicates kept

    @property
    def n_users(self) -> int:
        return len(self.users)

    def histories(self, store: EventStore, column: str = "item_id") -> list[np.ndarray]:
        """Every listen before the cutoff (not just the context), one array per user."""
        values = store.columns[column]
        return [
            np.asarray(values[lo:hi])
            for lo, hi in zip(self.history_starts, self.context_ends, strict=True)
        ]


def build_eval_set(
    store: EventStore, bounds: SplitBounds, split: str, seq_len: int, threshold: float
) -> EvalSet:
    """``split`` is ``"val"`` or ``"test"``; ``threshold`` is the Listen+ played ratio."""
    if split == "val":
        cutoff, start, end = bounds.train_end, bounds.val_start, bounds.val_end
    elif split == "test":
        cutoff, start, end = bounds.val_end, bounds.test_start, np.iinfo(np.int64).max
    else:
        raise ValueError(f"unknown split {split!r}")
    lo = store.offsets[:-1]
    has_train = store.cutoff_index(bounds.train_end) > lo
    period_lo, period_hi = store.cutoff_index(start), store.cutoff_index(end)
    ratio = store.columns["played_ratio_pct"]
    item_id = store.columns["item_id"]
    min_ratio = round(threshold * 100)
    users, targets = [], []
    for u in np.flatnonzero(has_train):
        rows = np.arange(period_lo[u], period_hi[u])
        rows = rows[np.asarray(ratio[rows]) >= min_ratio]
        if len(rows):
            users.append(u)
            targets.append(np.asarray(item_id[rows]))
    users = np.array(users, dtype=np.int64)
    ends = store.cutoff_index(cutoff)[users]
    return EvalSet(
        name=split,
        cutoff=cutoff,
        users=users,
        history_starts=lo[users],
        context_starts=np.maximum(lo[users], ends - seq_len),
        context_ends=ends,
        targets=targets,
    )


@torch.no_grad()
def recommend_model(
    model: GPT2Recommender,
    store: EventStore,
    eval_set: EvalSet,
    idx_to_item: np.ndarray,
    k: int = 100,
    exclude_seen: bool = False,
    batch_size: int = 256,
    amp_dtype: torch.dtype | None = None,
) -> np.ndarray:
    """``[n_users, k]`` raw item ids ranked by the model; ``idx_to_item`` maps vocab indices."""
    model.eval()
    device = next(model.parameters()).device
    seq_len = model.backbone.config.n_positions
    lengths = eval_set.context_ends - eval_set.context_starts
    out = []
    for b in range(0, eval_set.n_users, batch_size):
        sl = slice(b, b + batch_size)
        batch = store.gather(eval_set.context_starts[sl], lengths[sl], seq_len, eval_set.cutoff)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model.next_item_logits(to_tensors(batch, device)).float()
        if exclude_seen:
            seen = [
                np.asarray(store.columns["item_idx"][lo:hi])
                for lo, hi in zip(
                    eval_set.history_starts[sl], eval_set.context_ends[sl], strict=True
                )
            ]
            rows = np.repeat(np.arange(len(seen)), [len(s) for s in seen])
            cols = np.concatenate(seen)  # OOV/PAD columns are -inf already
            logits[torch.from_numpy(rows).to(device), torch.from_numpy(cols).long().to(device)] = (
                float("-inf")
            )
        top = logits.topk(k, dim=1)
        if torch.isinf(top.values).any():
            raise ValueError(f"fewer than k={k} rankable items for some user")
        out.append(top.indices.cpu().numpy())
    return idx_to_item[np.concatenate(out)]


def popular_items(store: EventStore, cutoff: int, days: float, threshold: float) -> np.ndarray:
    """Raw item ids by number of Listen+ events in ``[cutoff - days, cutoff)``, most popular
    first (items without such events are absent)."""
    ts = store.columns["timestamp"]
    rows = (ts >= cutoff - days * DAY_SECONDS) & (ts < cutoff)
    rows &= store.columns["played_ratio_pct"] >= round(threshold * 100)
    counts = (
        pl.DataFrame({"item_id": store.columns["item_id"][rows]})
        .group_by("item_id")
        .len()
        .sort(["len", "item_id"], descending=[True, False])
    )
    return counts["item_id"].to_numpy()


def _fill(first: np.ndarray, rest: np.ndarray, k: int, exclude: np.ndarray | None = None):
    """``first`` (unique, in order) followed by unseen items of ``rest``, cut to ``k``."""
    rest = rest[~np.isin(rest, first)]
    if exclude is not None:
        rest = rest[~np.isin(rest, exclude)]
    row = np.concatenate([first, rest])[:k]
    if len(row) < k:
        raise ValueError(f"only {len(row)} candidates for k={k}; increase MostPop days")
    return row


def recommend_most_pop(
    store: EventStore,
    eval_set: EvalSet,
    k: int = 100,
    days: float = 7,
    threshold: float = 0.5,
    exclude_seen: bool = False,
) -> np.ndarray:
    pop = popular_items(store, eval_set.cutoff, days, threshold)
    if not exclude_seen:
        return np.tile(pop[:k], (eval_set.n_users, 1))
    empty = pop[:0]
    return np.stack([_fill(empty, pop, k, exclude=h) for h in eval_set.histories(store)])


def recommend_repeat(
    store: EventStore, eval_set: EvalSet, k: int = 100, days: float = 7, threshold: float = 0.5
) -> np.ndarray:
    """Most recently listened distinct items of the user, then MostPop."""
    pop = popular_items(store, eval_set.cutoff, days, threshold)
    rows = []
    for history in eval_set.histories(store):
        recent = history[::-1]
        _, first_pos = np.unique(recent, return_index=True)
        rows.append(_fill(recent[np.sort(first_pos)][:k], pop, k))
    return np.stack(rows)


def evaluate_recs(
    recs: np.ndarray, eval_set: EvalSet, n_items: int, ks: tuple[int, ...] = (10, 100)
) -> dict[str, float]:
    """Relevance metrics, the Yambda-paper variants (``paper_*``) and Coverage@K.

    ``n_items`` is the catalogue size for coverage (the vocabulary without special tokens).
    """
    out = metrics.relevance_metrics(recs, eval_set.targets, ks)
    paper = metrics.yambda_paper_metrics(recs, eval_set.targets, ks)
    out |= {f"paper_{name}": value for name, value in paper.items()}
    out |= {f"coverage@{k}": metrics.coverage(recs, k, n_items) for k in ks}
    return out


def idx_to_item_array(items: pl.DataFrame) -> np.ndarray:
    """Vocabulary index -> raw item id (special tokens map to 0, they are never ranked)."""
    out = np.zeros(N_SPECIAL_TOKENS + len(items), dtype=np.int64)
    out[items["idx"].to_numpy()] = items["item_id"].to_numpy()
    return out


def results_table(
    prepared: PreparedData,
    data_cfg: DataConfig,
    models: dict[str, GPT2Recommender] | None = None,
    splits: tuple[str, ...] = ("val", "test"),
    most_pop_days: float = 7,
    amp_dtype: torch.dtype | None = None,
) -> pl.DataFrame:
    """One row per (split, method, repeat-filter mode): MostPop, Repeat and every model."""
    store, items = prepared.store, prepared.items
    idx_to_item = idx_to_item_array(items)
    rows = []
    for split in splits:
        es = build_eval_set(
            store, prepared.bounds, split, data_cfg.max_seq_len, data_cfg.listen_plus_threshold
        )
        runs = [
            ("MostPop", False, recommend_most_pop(store, es, days=most_pop_days)),
            ("MostPop", True, recommend_most_pop(store, es, days=most_pop_days, exclude_seen=True)),
            ("Repeat", False, recommend_repeat(store, es, days=most_pop_days)),
        ]
        for name, model in (models or {}).items():
            for exclude in (False, True):
                recs = recommend_model(
                    model, store, es, idx_to_item, exclude_seen=exclude, amp_dtype=amp_dtype
                )
                runs.append((name, exclude, recs))
        for method, exclude, recs in runs:
            row = {"split": split, "method": method, "exclude_seen": exclude, "users": es.n_users}
            rows.append(row | evaluate_recs(recs, es, len(items)))
    return pl.DataFrame(rows)
