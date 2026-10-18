"""Exploratory analysis of Yambda (stage 1.2, research plan §2.3).

Every function takes the *train* split (or its listens) so that no statistic that later feeds
a model decision (vocabulary size, bin edges) peeks at val/test. The only exception is
``vocab_coverage``, which reports how well a train-built vocabulary covers val/test targets.

Played ratio is clipped to 100 when averaged: values > 100 mean rewinding and re-listening.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from recsys_rl.data import DAY_SECONDS, add_session_ids, is_listen, is_listen_plus

RATIO = pl.col("played_ratio_pct")
RATIO_CLIPPED = RATIO.clip(0, 100)
LENGTH = pl.col("track_length_seconds")


def bin_index(expr: pl.Expr, edges: Sequence[float]) -> pl.Expr:
    """Bin number ``i`` such that ``edges[i-1] < x <= edges[i]`` (0 below the first edge)."""
    return pl.sum_horizontal([(expr > e).cast(pl.UInt8) for e in edges]).alias("bin")


def bin_labels(edges: Sequence[float]) -> list[str]:
    bounds = ["-inf", *map(str, edges), "inf"]
    return [f"({lo}, {hi}]" for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)]


def listen_stats() -> list[pl.Expr]:
    return [
        pl.len().alias("listens"),
        RATIO_CLIPPED.mean().alias("mean_ratio"),
        (RATIO >= 50).mean().alias("listen_plus"),
        (RATIO <= 5).mean().alias("skip_le5"),
        (RATIO >= 100).mean().alias("full"),
    ]


def ratio_summary(listens: pl.LazyFrame) -> pl.DataFrame:
    """Overall shares: skips (<= 5%), Listen+ (>= 50%), full plays, rewinds (> 100%)."""
    return listens.select(*listen_stats(), (RATIO > 100).mean().alias("rewind_gt100")).collect()


def ratio_histogram(listens: pl.LazyFrame, edges: Sequence[float]) -> pl.DataFrame:
    """Share of listens per played-ratio bin (``edges`` as in ``bin_index``)."""
    return (
        listens.group_by(bin_index(RATIO, edges))
        .agg(share=pl.len())
        .with_columns(pl.col("share") / pl.col("share").sum())
        .sort("bin")
        .collect()
        .with_columns(label=pl.col("bin").map_elements(bin_labels(edges).__getitem__, pl.String))
    )


def length_quantiles(listens: pl.LazyFrame, qs: Sequence[float]) -> dict[float, float]:
    row = listens.select([LENGTH.quantile(q).alias(str(q)) for q in qs]).collect().row(0)
    return dict(zip(qs, row, strict=True))


def ratio_by_length(listens: pl.LazyFrame, edges: Sequence[float]) -> pl.DataFrame:
    """Listen stats per track-length bin, plus mean listened seconds."""
    return (
        listens.group_by(bin_index(LENGTH, edges))
        .agg(*listen_stats(), (RATIO_CLIPPED * LENGTH / 100).mean().alias("listened_seconds"))
        .sort("bin")
        .collect()
        .with_columns(label=pl.col("bin").map_elements(bin_labels(edges).__getitem__, pl.String))
    )


def with_repeat_flag(listens: pl.LazyFrame) -> pl.LazyFrame:
    """``repeat``: the user has listened to this item before (within the given frame)."""
    return listens.with_columns(repeat=pl.int_range(pl.len()).over("uid", "item_id") > 0)


def first_vs_repeat(listens: pl.LazyFrame) -> pl.DataFrame:
    """Listen stats for first vs repeated user-item listens, split by ``is_organic``."""
    return (
        with_repeat_flag(listens)
        .group_by("repeat", "is_organic")
        .agg(*listen_stats())
        .sort("repeat", "is_organic")
        .collect()
    )


def organic_profile(train: pl.LazyFrame) -> pl.DataFrame:
    """Per ``is_organic`` value: listen stats, repeat share and feedback events per 1k listens.

    Used to settle the flag's meaning (the HF card is ambiguous): library/search listens
    should be mostly repeats, and unlikes / undislikes happen mostly in the user's library.
    """
    listens = (
        with_repeat_flag(train.filter(is_listen()))
        .group_by("is_organic")
        .agg(*listen_stats(), pl.col("repeat").mean().alias("repeat_share"))
    )
    feedback = (
        train.filter(~is_listen())
        .group_by("is_organic")
        .agg([(pl.col("event_type") == e).sum().alias(e) for e in ("like", "dislike", "unlike")])
    )
    per_1k = [
        (pl.col(e) / pl.col("listens") * 1000).alias(f"{e}_per_1k")
        for e in ("like", "dislike", "unlike")
    ]
    return (
        listens.join(feedback, on="is_organic")
        .with_columns(per_1k)
        .drop("like", "dislike", "unlike")
        .sort("is_organic")
        .collect()
    )


def sessions(listens: pl.LazyFrame, gap_minutes: int) -> pl.DataFrame:
    """One row per session: size, duration (min) and pause before it (hours, null if first)."""
    return (
        add_session_ids(listens, gap_minutes)
        .group_by("uid", "session_id")
        .agg(
            size=pl.len(),
            start=pl.col("timestamp").min(),
            end=pl.col("timestamp").max(),
        )
        .sort("uid", "session_id")
        .with_columns(
            duration_min=(pl.col("end") - pl.col("start")) / 60,
            pause_h=(pl.col("start").cast(pl.Int64) - pl.col("end").shift().over("uid")) / 3600,
        )
        .collect()
    )


def daily_activity(listens: pl.LazyFrame) -> dict[str, float]:
    """Next-day return rate (user active on day d is active on d+1) and active days per user."""
    days = listens.select("uid", day=pl.col("timestamp") // DAY_SECONDS).unique().collect()
    returned = days.join(days.with_columns(pl.col("day") - 1), on=["uid", "day"], how="semi")
    return {
        "next_day_return": returned.height / days.height,
        "median_active_days": float(days.group_by("uid").len()["len"].median()),
    }


def item_ranks(train: pl.LazyFrame) -> pl.DataFrame:
    """Train listen counts with popularity rank (0 = most listened), ties by ``item_id``."""
    return (
        train.filter(is_listen())
        .group_by("item_id")
        .agg(count=pl.len())
        .sort(["count", "item_id"], descending=[True, False])
        .with_row_index("rank")
        .collect()
    )


def vocab_coverage(
    ranks: pl.DataFrame, splits: dict[str, pl.LazyFrame], sizes: Sequence[int], threshold: float
) -> pl.DataFrame:
    """For each top-N vocabulary: min item count, share of train listens covered and share of
    val/test Listen+ targets covered (an upper bound on what a model with that vocabulary
    can ever hit). Same ordering as ``data.build_vocab``."""
    counts = ranks["count"].to_numpy()
    cum = np.cumsum(counts) / counts.sum()
    target_ranks = {
        name: splits[name]
        .filter(is_listen_plus(threshold))
        .join(ranks.lazy(), on="item_id", how="left")
        .select(pl.col("rank").fill_null(np.iinfo(np.int64).max))
        .collect()["rank"]
        .to_numpy()
        for name in ("val", "test")
    }
    rows = []
    for n in sizes:
        n_eff = min(n, len(counts))
        row = {"vocab_size": n, "min_count": int(counts[n_eff - 1]), "train": float(cum[n_eff - 1])}
        row |= {name: float((r < n).mean()) for name, r in target_ranks.items()}
        rows.append(row)
    return pl.DataFrame(rows)


def plot_ratio_by_length(table: pl.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(table.height)
    ax.plot(x, table["mean_ratio"], marker="o", label="mean played ratio, %")
    ax.plot(x, table["listen_plus"] * 100, marker="o", label="Listen+ share, %")
    ax.set_xticks(x, table["label"], rotation=30)
    ax.set_xlabel("track length, s")
    ax.set_ylabel("%")
    ax2 = ax.twinx()
    ax2.bar(x, table["listened_seconds"], alpha=0.2, color="gray", label="listened seconds")
    ax2.set_ylabel("mean listened seconds")
    fig.legend(loc="upper center", ncol=3, fontsize=8)
    fig.tight_layout()
    return fig


def plot_vocab_coverage(table: pl.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    for col in ("train", "val", "test"):
        ax.plot(table["vocab_size"], table[col], marker="o", label=col)
    ax.set_xscale("log")
    ax.set_xlabel("vocabulary size (top-N train items)")
    ax.set_ylabel("covered share")
    ax.legend(title="train listens / Listen+ targets")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_hist(
    values: np.ndarray, bins: Sequence[float], xlabel: str, logx: bool = False
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(values, bins=bins)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    return fig
