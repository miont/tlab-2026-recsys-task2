"""Data pipeline (stage 1 of the research plan).

Responsibilities:
- read ``flat/<size>/multi_event.parquet`` with polars;
- Global Temporal Split: train | val day | gap | test day (Yambda protocol);
- item vocabulary built on the train period only (top-N + ``<OOV>``);
- feedback features: played-ratio bins, track-length bins, organic flag, like flag;
- session boundaries (gap > ``session_gap_minutes``);
- sliding training windows cached as numpy arrays under ``Config.processed_dir``.

Implemented so far: loading, split, Listen+, sessions, vocabulary, evaluation targets.
Feedback bins and windows come after the EDA, which picks the vocabulary size and bin edges.

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

from dataclasses import dataclass

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
