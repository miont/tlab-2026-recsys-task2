"""Data pipeline (stage 1 of the research plan).

Responsibilities:
- read ``flat/<size>/multi_event.parquet`` with polars;
- Global Temporal Split: train | val day | gap | test day (Yambda protocol);
- item vocabulary built on the train period only (top-N + ``<OOV>``);
- feedback features: played-ratio bins, track-length bins, organic flag, like flag;
- session boundaries (gap > ``session_gap_minutes``);
- sliding training windows cached as numpy arrays under ``Config.processed_dir``.

``prepare(cfg)`` builds (or loads from the cache) everything a model needs: the vocabulary
with train-only item statistics, an ``EventStore`` with one row per listen of every period
(train, gaps, val, test; contexts for evaluation come from it) and the train windows.

Split layout, matching ``flat_split_train_val_test`` in the Yambda benchmarks
(timestamps are seconds, quantised to 5 s; the last one is 26_000_000)::

    train: t < train_end
    gap
    val:   val_start <= t < val_end      (val_start = train_end + gap)
    gap
    test:  t >= test_start               (test_start = last_timestamp - test_days)

Events inside the gaps belong to no split. Val and test keep only users seen in train.
The model is frozen at the start of an evaluation period, so the context for val is
``t < train_end`` and the context for test is ``t < val_end`` (train + val).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl

from recsys_rl.config import Config, DataConfig

DAY_SECONDS = 24 * 60 * 60

# Reserved vocabulary indices; real items start at N_SPECIAL_TOKENS.
PAD_IDX = 0
OOV_IDX = 1
N_SPECIAL_TOKENS = 2


def scan_events(cfg: Config) -> pl.LazyFrame:
    """Lazy frame over ``multi_event.parquet`` (sorted by uid, timestamp)."""
    return pl.scan_parquet(cfg.raw_dir / "flat" / cfg.data.dataset_size / "multi_event.parquet")


@dataclass(frozen=True)
class SplitBounds:
    train_end: int
    val_start: int
    val_end: int
    test_start: int

    @classmethod
    def from_last_timestamp(cls, last_timestamp: int, data_cfg: DataConfig) -> SplitBounds:
        gap = data_cfg.gap_minutes * 60
        test_start = last_timestamp - data_cfg.test_days * DAY_SECONDS
        val_end = test_start - gap
        val_start = val_end - data_cfg.val_days * DAY_SECONDS
        train_end = val_start - gap
        if train_end <= 0:
            raise ValueError(f"split leaves no train period (train_end={train_end})")
        return cls(train_end=train_end, val_start=val_start, val_end=val_end, test_start=test_start)

    @classmethod
    def from_events(cls, events: pl.LazyFrame, data_cfg: DataConfig) -> SplitBounds:
        last = events.select(pl.col("timestamp").max()).collect().item()
        return cls.from_last_timestamp(int(last), data_cfg)


def temporal_split(events: pl.LazyFrame, bounds: SplitBounds) -> dict[str, pl.LazyFrame]:
    """Split events into ``train`` / ``val`` / ``test`` (see module docstring)."""
    ts = pl.col("timestamp")
    train = events.filter(ts < bounds.train_end)
    train_users = train.select("uid").unique()
    val = events.filter((ts >= bounds.val_start) & (ts < bounds.val_end))
    test = events.filter(ts >= bounds.test_start)
    return {
        "train": train,
        "val": val.join(train_users, on="uid", how="semi", maintain_order="left"),
        "test": test.join(train_users, on="uid", how="semi", maintain_order="left"),
    }


def is_listen() -> pl.Expr:
    return pl.col("event_type") == "listen"


def is_listen_plus(threshold: float) -> pl.Expr:
    """Listen with ``played_ratio_pct >= threshold * 100`` (Yambda: 50)."""
    return is_listen() & (pl.col("played_ratio_pct") >= round(threshold * 100))


def add_session_ids(events: pl.LazyFrame, gap_minutes: int) -> pl.LazyFrame:
    """Add ``session_id`` (0, 1, ... per user): a new session starts after a pause > gap.

    Expects rows sorted by (uid, timestamp). Uses only past events, so it is leak-free.
    """
    new_session = (pl.col("timestamp").diff() > gap_minutes * 60).fill_null(False)
    return events.with_columns(session_id=new_session.cast(pl.UInt32).cum_sum().over("uid"))


def build_vocab(train: pl.LazyFrame, data_cfg: DataConfig) -> pl.DataFrame:
    """Item vocabulary from train listens: ``item_id -> idx`` for the most listened items.

    Keeps items with at least ``min_item_count`` listens, at most ``max_vocab_size`` of them;
    indices start at ``N_SPECIAL_TOKENS`` (``PAD_IDX``, ``OOV_IDX`` are reserved).
    Ties are broken by ``item_id`` so the vocabulary is deterministic.
    """
    return (
        train.filter(is_listen())
        .group_by("item_id")
        .agg(count=pl.len())
        .filter(pl.col("count") >= data_cfg.min_item_count)
        .sort(["count", "item_id"], descending=[True, False])
        .head(data_cfg.max_vocab_size)
        .with_row_index("idx", offset=N_SPECIAL_TOKENS)
        .select("item_id", pl.col("idx").cast(pl.UInt32), "count")
        .collect()
    )


def encode_items(events: pl.LazyFrame, vocab: pl.DataFrame) -> pl.LazyFrame:
    """Add ``item_idx``: vocabulary index, ``OOV_IDX`` for items outside the vocabulary."""
    return events.join(
        vocab.lazy().select("item_id", item_idx="idx"),
        on="item_id",
        how="left",
        maintain_order="left",
    ).with_columns(pl.col("item_idx").fill_null(OOV_IDX))


def eval_targets(split: pl.LazyFrame, threshold: float) -> pl.DataFrame:
    """Per-user Listen+ items of an evaluation period: columns ``uid``, ``item_id`` (list).

    Raw item ids in event order, duplicates kept (needed by ``yambda_paper_metrics``);
    users without Listen+ events are absent.
    """
    return (
        split.filter(is_listen_plus(threshold))
        .group_by("uid")
        .agg(pl.col("item_id"))
        .sort("uid")
        .collect()
    )


def bin_index(expr: pl.Expr, edges: Sequence[float]) -> pl.Expr:
    """Bin number ``i`` such that ``edges[i-1] < x <= edges[i]`` (0 below the first edge)."""
    return pl.sum_horizontal([(expr > e).cast(pl.UInt8) for e in edges]).alias("bin")


def item_stats(train: pl.LazyFrame, vocab: pl.DataFrame) -> pl.DataFrame:
    """Vocabulary plus train-only item statistics: mean clipped ratio (0..1), track length."""
    stats = (
        train.filter(is_listen())
        .group_by("item_id")
        .agg(
            mean_ratio=pl.col("played_ratio_pct").clip(0, 100).mean() / 100,
            track_length=pl.col("track_length_seconds").median(),
        )
    )
    return vocab.lazy().join(stats, on="item_id", how="left").sort("idx").collect()


def attach_feedback(
    listens: pl.LazyFrame, events: pl.LazyFrame, event_type: str, gap_minutes: int
) -> pl.LazyFrame:
    """Add ``<event_type>_delay``: seconds from a listen to the like/dislike attributed to it.

    A feedback event is attributed to the latest earlier listen of the same item by the same
    user, if it happens before that listen's session ends plus ``gap_minutes``; -1 otherwise.
    ``listens`` must carry ``row`` and ``session_id``. Whether the flag may be used as an input
    feature depends on *when* it is observed, see ``EventStore.gather``.
    """
    session_end = listens.group_by("uid", "session_id").agg(session_end=pl.col("timestamp").max())
    feedback = (
        events.filter(pl.col("event_type") == event_type)
        .select("uid", "item_id", feedback_ts="timestamp")
        .sort("feedback_ts")
    )
    candidates = listens.select("uid", "item_id", "row", "session_id", listen_ts="timestamp").sort(
        "listen_ts"
    )
    attributed = (
        feedback.join_asof(
            candidates,
            left_on="feedback_ts",
            right_on="listen_ts",
            by=["uid", "item_id"],
            strategy="backward",
            check_sortedness=False,  # sorted by timestamp; polars cannot verify it per group
        )
        .join(session_end, on=["uid", "session_id"])  # inner: drops feedback without a listen
        .filter(pl.col("feedback_ts") <= pl.col("session_end") + gap_minutes * 60)
        .group_by("row")
        .agg(delay=(pl.col("feedback_ts") - pl.col("listen_ts")).min())
    )
    return (
        listens.join(attributed, on="row", how="left", maintain_order="left")
        .with_columns(pl.col("delay").fill_null(-1).cast(pl.Int32).alias(f"{event_type}_delay"))
        .drop("delay")
    )


# Per-listen columns of the event store and their on-disk dtypes.
EVENT_COLUMNS = {
    "item_id": np.uint32,  # raw id: evaluation targets and recommendations are in raw ids
    "item_idx": np.int32,  # vocabulary index, OOV_IDX outside the vocabulary
    "timestamp": np.uint32,
    "played_ratio_pct": np.uint16,
    "track_length_seconds": np.uint32,
    "ratio_bin": np.uint8,
    "length_bin": np.uint8,
    "is_organic": np.uint8,
    "session_id": np.uint32,
    "next_gap": np.int32,  # seconds to the user's next listen, -1 for the last one
    "like_delay": np.int32,
    "dislike_delay": np.int32,
}


def build_event_table(
    events: pl.LazyFrame, vocab: pl.DataFrame, data_cfg: DataConfig
) -> pl.DataFrame:
    """All listens (every period) with model features, sorted by (uid, timestamp)."""
    listens = add_session_ids(
        events.filter(is_listen()).drop("event_type").with_row_index("row"),
        data_cfg.session_gap_minutes,
    )
    for event_type in ("like", "dislike"):
        listens = attach_feedback(listens, events, event_type, data_cfg.session_gap_minutes)
    ts = pl.col("timestamp").cast(pl.Int64)
    return (
        encode_items(listens, vocab)
        .with_columns(
            ratio_bin=bin_index(pl.col("played_ratio_pct"), data_cfg.ratio_bin_edges),
            length_bin=bin_index(pl.col("track_length_seconds"), data_cfg.length_bin_edges),
            next_gap=(ts.shift(-1).over("uid") - ts).fill_null(-1),
        )
        .select("uid", *EVENT_COLUMNS)
        .collect()
    )


@dataclass
class EventStore:
    """Listens of all users as flat per-event arrays; user ``u`` owns rows
    ``offsets[u]:offsets[u + 1]`` (chronological), ``uids[u]`` is its raw id."""

    uids: np.ndarray
    offsets: np.ndarray
    columns: dict[str, np.ndarray]

    @classmethod
    def from_table(cls, table: pl.DataFrame) -> EventStore:
        counts = table.group_by("uid", maintain_order=True).len()
        offsets = np.concatenate([[0], np.cumsum(counts["len"].to_numpy())]).astype(np.int64)
        columns = {c: table[c].to_numpy().astype(dt) for c, dt in EVENT_COLUMNS.items()}
        return cls(counts["uid"].to_numpy().astype(np.uint32), offsets, columns)

    @property
    def n_users(self) -> int:
        return len(self.uids)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "uids.npy", self.uids)
        np.save(directory / "offsets.npy", self.offsets)
        for name, values in self.columns.items():
            np.save(directory / f"{name}.npy", values)

    @classmethod
    def load(cls, directory: Path, mmap: bool = True) -> EventStore:
        mode = "r" if mmap else None
        columns = {c: np.load(directory / f"{c}.npy", mmap_mode=mode) for c in EVENT_COLUMNS}
        return cls(np.load(directory / "uids.npy"), np.load(directory / "offsets.npy"), columns)

    def in_memory(self) -> EventStore:
        """Copy memmapped columns to RAM (~1.7 GB for 50M): random window reads from a
        memmap on Google Drive go through FUSE and are slow. Columns already in RAM are shared,
        not copied."""
        columns = {
            c: np.array(v) if isinstance(v, np.memmap) else v for c, v in self.columns.items()
        }
        return EventStore(self.uids, self.offsets, columns)

    def cutoff_index(self, cutoff: int) -> np.ndarray:
        """Per user, the first row with ``timestamp >= cutoff`` (absolute row index)."""
        ts = self.columns["timestamp"]
        return np.array(
            [
                lo + np.searchsorted(ts[lo:hi], cutoff)
                for lo, hi in zip(self.offsets[:-1], self.offsets[1:], strict=True)
            ],
            dtype=np.int64,
        )

    def contexts(self, cutoff: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        """``(starts, lengths)`` of each user's last ``seq_len`` listens before ``cutoff``."""
        end = self.cutoff_index(cutoff)
        starts = np.maximum(self.offsets[:-1], end - seq_len)
        return starts, end - starts

    def train_windows(
        self, train_end: int, seq_len: int, stride: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(starts, lengths)`` of sliding windows over each user's train listens.

        Windows are aligned to the end of the train history (the most recent events are
        always used), step back by ``stride`` and the last one is clamped to the first event.
        Users with fewer than 2 train listens give no window (nothing to predict).
        """
        end = self.cutoff_index(train_end)
        starts, lengths = [], []
        for lo, hi in zip(self.offsets[:-1], end, strict=True):
            n = hi - lo
            if n < 2:
                continue
            if n <= seq_len:
                starts.append([lo])
                lengths.append([n])
                continue
            s = np.arange(hi - seq_len, lo, -stride)
            if s[-1] != lo:
                s = np.append(s, lo)
            starts.append(s)
            lengths.append(np.full(len(s), seq_len))
        return np.concatenate(starts).astype(np.int64), np.concatenate(lengths).astype(np.int64)

    def gather(
        self, starts: np.ndarray, lengths: np.ndarray, seq_len: int, cutoff: int
    ) -> dict[str, np.ndarray]:
        """Right-padded ``[B, seq_len]`` batch of every column plus ``mask`` and the
        leak-free input flags ``liked`` / ``disliked``. The raw ``*_delay`` columns stay in the
        batch for feedback *targets* (``reward_model.feedback_targets``), never as inputs.

        A like/dislike counts as an input feature of a listen only if it was given before the
        user's next listen started and before ``cutoff`` (the end of the period whose events
        may be observed: ``train_end`` for training, the context cutoff for evaluation).
        Padding rows are zero; ``item_idx`` padding is ``PAD_IDX``.
        """
        pos = np.arange(seq_len)
        mask = pos[None, :] < np.asarray(lengths)[:, None]
        idx = np.where(mask, np.asarray(starts)[:, None] + pos[None, :], 0)
        batch = {name: np.where(mask, values[idx], 0) for name, values in self.columns.items()}
        ts = batch["timestamp"].astype(np.int64)
        next_gap = batch["next_gap"]
        for name in ("like", "dislike"):
            delay = batch[f"{name}_delay"]
            seen = (delay >= 0) & ((delay < next_gap) | (next_gap < 0)) & (ts + delay < cutoff)
            batch[f"{name}d"] = (seen & mask).astype(np.uint8)
        batch["item_idx"] = np.where(mask, batch["item_idx"], PAD_IDX)
        batch["mask"] = mask
        return batch


@dataclass
class PreparedData:
    bounds: SplitBounds
    items: pl.DataFrame  # vocabulary + train-only item statistics
    store: EventStore
    window_starts: np.ndarray
    window_lengths: np.ndarray


CACHE_VERSION = 1


def prepare(cfg: Config, force: bool = False) -> PreparedData:
    """Build or load the cached data under ``cfg.processed_dir``.

    The cache is reused only if it was built with the same ``DataConfig`` and cache version;
    the meta file is written last, so an interrupted build is redone from scratch.
    """
    d = cfg.data
    bins = (len(d.ratio_bin_edges) + 1, len(d.length_bin_edges) + 1)
    if bins != (cfg.model.n_ratio_bins, cfg.model.n_length_bins):
        raise ValueError(
            f"bin edges give {bins} bins, model expects "
            f"{(cfg.model.n_ratio_bins, cfg.model.n_length_bins)}"
        )
    out = cfg.processed_dir
    meta_path = out / "meta.json"
    meta = {"version": CACHE_VERSION, "data": asdict(d)}
    if not force and meta_path.exists():
        stored = json.loads(meta_path.read_text())
        if {k: stored[k] for k in meta} == meta:
            return PreparedData(
                bounds=SplitBounds(**stored["bounds"]),
                items=pl.read_parquet(out / "items.parquet"),
                store=EventStore.load(out),
                window_starts=np.load(out / "window_starts.npy"),
                window_lengths=np.load(out / "window_lengths.npy"),
            )
    meta_path.unlink(missing_ok=True)

    events = scan_events(cfg)
    bounds = SplitBounds.from_events(events, d)
    train = temporal_split(events, bounds)["train"]
    items = item_stats(train, build_vocab(train, d))
    store = EventStore.from_table(build_event_table(events, items, d))
    starts, lengths = store.train_windows(bounds.train_end, d.max_seq_len, d.window_stride)

    store.save(out)
    items.write_parquet(out / "items.parquet")
    np.save(out / "window_starts.npy", starts)
    np.save(out / "window_lengths.npy", lengths)
    meta_path.write_text(json.dumps(meta | {"bounds": asdict(bounds)}, indent=2))
    return PreparedData(bounds, items, store, starts, lengths)
