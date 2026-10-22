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

Qualitative inspection: Yambda is anonymised (tracks, artists, albums are numeric ids, there
are no titles), so ``explain_user`` describes one user's context and top-K by artist / album
ids, train popularity and overlap with the history, next to Repeat and MostPop and the true
period targets; ``recs_profile`` aggregates the same tags over all users of an eval set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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

    def subset(self, positions: np.ndarray | list[int]) -> EvalSet:
        """The eval set restricted to the users at ``positions`` (indices into ``users``)."""
        p = np.asarray(positions, dtype=np.int64)
        return EvalSet(
            name=self.name,
            cutoff=self.cutoff,
            users=self.users[p],
            history_starts=self.history_starts[p],
            context_starts=self.context_starts[p],
            context_ends=self.context_ends[p],
            targets=[self.targets[i] for i in p],
        )


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


def item_meta(artists: pl.DataFrame, albums: pl.DataFrame, items: pl.DataFrame) -> pl.DataFrame:
    """Every known track (sorted by ``item_id``) with ``artist_id``, ``album_id`` and, for
    vocabulary items, train-only ``pop_rank`` (1 = most listened), ``mean_ratio`` and
    ``track_length``. A track with several artists or albums keeps the first one."""

    def first(mapping: pl.DataFrame, column: str) -> pl.LazyFrame:
        return (
            mapping.lazy()
            .select(pl.col("item_id").cast(pl.Int64), pl.col(column).cast(pl.Int64))
            .unique("item_id", keep="first", maintain_order=True)
        )

    stats = items.lazy().select(
        pl.col("item_id").cast(pl.Int64),
        pop_rank=pl.col("idx").cast(pl.Int64) - N_SPECIAL_TOKENS + 1,
        mean_ratio="mean_ratio",
        track_length="track_length",
    )
    return (
        first(artists, "artist_id")
        .join(first(albums, "album_id"), on="item_id", how="full", coalesce=True)
        .join(stats, on="item_id", how="full", coalesce=True)
        .sort("item_id")
        .collect()
    )


def load_item_meta(raw_dir: Path, items: pl.DataFrame) -> pl.DataFrame:
    """``item_meta`` from the downloaded ``{artist,album}_item_mapping.parquet``."""
    return item_meta(
        pl.read_parquet(raw_dir / "artist_item_mapping.parquet"),
        pl.read_parquet(raw_dir / "album_item_mapping.parquet"),
        items,
    )


def _tag_items(
    item_ids: np.ndarray, meta: pl.DataFrame, history: np.ndarray, targets: np.ndarray
) -> pl.DataFrame:
    """Raw ids (in order) with their metadata, how often the user heard them before the
    cutoff, whether the artist / album was heard and how often they are period targets."""
    hist = pl.DataFrame({"item_id": history.astype(np.int64)}).join(
        meta.select("item_id", "artist_id", "album_id"), on="item_id", how="left"
    )
    heard = hist.group_by("item_id").agg(heard=pl.len().cast(pl.Int64))
    hits = pl.DataFrame({"item_id": targets.astype(np.int64)}).group_by("item_id").len("target")
    return (
        pl.DataFrame({"item_id": item_ids.astype(np.int64)})
        .join(meta, on="item_id", how="left", maintain_order="left")
        .join(heard, on="item_id", how="left", maintain_order="left")
        .join(hits, on="item_id", how="left", maintain_order="left")
        .with_columns(
            pl.col("heard", "target").fill_null(0),
            artist_heard=pl.col("artist_id").is_in(hist["artist_id"].drop_nulls().implode()),
            album_heard=pl.col("album_id").is_in(hist["album_id"].drop_nulls().implode()),
        )
    )


@dataclass
class Explanation:
    uid: int  # raw user id
    history: pl.DataFrame  # the last listens of the context, oldest first, as the model saw them
    recs: pl.DataFrame  # top-k of the model, Repeat and MostPop with item tags
    targets: pl.DataFrame  # the period's Listen+ events with their rank in the model ranking


@torch.no_grad()
def explain_user(
    model: GPT2Recommender,
    store: EventStore,
    eval_set: EvalSet,
    position: int,
    idx_to_item: np.ndarray,
    meta: pl.DataFrame,
    k: int = 10,
    n_history: int = 20,
    exclude_seen: bool = False,
    most_pop_days: float = 7,
    threshold: float = 0.5,
) -> Explanation:
    """Qualitative view of the recommendations for the user at ``position`` of ``eval_set``.

    ``prob`` is the full-softmax probability of the model (after the repeat filter if
    ``exclude_seen``); ``model_rank`` of a target is null for OOV and filtered items.
    """
    model.eval()
    device = next(model.parameters()).device
    one = eval_set.subset([position])
    length = int(one.context_ends[0] - one.context_starts[0])
    batch = store.gather(
        one.context_starts,
        one.context_ends - one.context_starts,
        model.backbone.config.n_positions,
        one.cutoff,
    )
    logits = model.next_item_logits(to_tensors(batch, device)).float()[0]
    history = one.histories(store)[0]
    if exclude_seen:
        seen = np.asarray(store.columns["item_idx"][one.history_starts[0] : one.context_ends[0]])
        logits[torch.from_numpy(seen).long().to(device)] = float("-inf")
    probs = logits.softmax(0).cpu().numpy()
    order = logits.argsort(descending=True).cpu().numpy()
    order = order[np.isfinite(logits.cpu().numpy()[order])]
    ranking = pl.DataFrame(
        {
            "item_id": idx_to_item[order].astype(np.int64),
            "model_rank": np.arange(1, len(order) + 1),
            "prob": probs[order],
        }
    )
    targets = one.targets[0]

    recs = []
    for method, ids in {
        "model": idx_to_item[order[:k]],
        "Repeat": recommend_repeat(store, one, k, most_pop_days, threshold)[0],
        "MostPop": recommend_most_pop(
            store, one, k, most_pop_days, threshold, exclude_seen=exclude_seen
        )[0],
    }.items():
        tagged = _tag_items(ids, meta, history, targets).join(
            ranking.select("item_id", "prob"), on="item_id", how="left", maintain_order="left"
        )
        recs.append(tagged.with_columns(method=pl.lit(method), rank=pl.int_range(1, len(ids) + 1)))
    recs = pl.concat(recs).select(
        "method", "rank", "item_id", "prob", pl.exclude("method", "rank", "item_id", "prob")
    )

    rows = slice(max(0, length - n_history), length)
    context = pl.DataFrame(
        {
            "timestamp": batch["timestamp"][0, rows].astype(np.int64),
            "item_id": batch["item_id"][0, rows].astype(np.int64),
            "played_ratio_pct": batch["played_ratio_pct"][0, rows].astype(np.int64),
            "is_organic": batch["is_organic"][0, rows].astype(bool),
            "liked": batch["liked"][0, rows].astype(bool),
            "disliked": batch["disliked"][0, rows].astype(bool),
            "in_vocab": batch["item_idx"][0, rows] >= N_SPECIAL_TOKENS,
        }
    ).join(
        meta.select("item_id", "artist_id", "album_id", "pop_rank"),
        on="item_id",
        how="left",
        maintain_order="left",
    )
    target_rows = (
        _tag_items(targets, meta, history, targets)
        .drop("target")
        .join(
            ranking.select("item_id", "model_rank"), on="item_id", how="left", maintain_order="left"
        )
    )
    return Explanation(int(store.uids[one.users[0]]), context, recs, target_rows)


def recs_profile(
    recs: np.ndarray, eval_set: EvalSet, store: EventStore, meta: pl.DataFrame, k: int = 10
) -> dict[str, float]:
    """What the top-``k`` consists of, averaged over users: tracks heard before the cutoff,
    unheard tracks of heard artists, tracks of new (or unknown) artists; median ``pop_rank`` of the
    recommended vocabulary items (null ranks are ignored)."""
    keys = meta["item_id"].to_numpy()

    def lookup(ids: np.ndarray, column: str) -> np.ndarray:
        values = meta[column].fill_null(-1).to_numpy()
        pos = np.minimum(np.searchsorted(keys, ids), len(keys) - 1)
        return np.where(keys[pos] == ids, values[pos], -1)

    top = recs[:, :k].astype(np.int64)
    artists = lookup(top, "artist_id")
    heard, known_artist = np.zeros(top.shape, bool), np.zeros(top.shape, bool)
    for u, history in enumerate(eval_set.histories(store)):
        history = history.astype(np.int64)
        heard[u] = np.isin(top[u], history)
        hist_artists = lookup(history, "artist_id")
        known_artist[u] = np.isin(artists[u], hist_artists[hist_artists >= 0]) & ~heard[u]
    pop_rank = lookup(top, "pop_rank")
    return {
        f"heard@{k}": float(heard.mean()),
        f"known_artist@{k}": float(known_artist.mean()),
        f"new_artist@{k}": float((~heard & ~known_artist).mean()),
        f"median_pop_rank@{k}": float(np.median(pop_rank[pop_rank > 0])),
    }
