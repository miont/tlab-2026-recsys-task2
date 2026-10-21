"""Reward-model evaluation (stage 3.4, research plan §5.4).

Everything works on a ``reward_model.FeedbackSet`` (the listens of the val or test day) and on
predictions ``[N, H]`` (ensemble mean) of any method:

- ``baseline_predictions``: global / per-user / per-item train means of every target and the
  additive user + item combination, smoothed towards the global mean (train only);
- ``quality_table``: ``feedback_metrics`` per method;
- ``event_slices`` + ``slice_table``: metrics by track length, popularity, first listen vs
  repeat, organic vs recommendation and position in the session;
- ``calibration_table``: reliability-diagram data;
- ``uncertainty_table``: does the ensemble spread track the error?
- ``sensitivity``: mean prediction when a single candidate feature is forced to each value
  (length bin, repeat count, organic flag), i.e. the "cheap paths" the RM offers to RL;
- ``catalog_profile``: what the RM prefers over the whole vocabulary for frozen contexts
  (the RL action space): its top-K vs the catalogue in length, popularity and repeats.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl
import torch

from recsys_rl import metrics
from recsys_rl.config import DataConfig
from recsys_rl.data import N_SPECIAL_TOKENS, EventStore, SplitBounds
from recsys_rl.evaluate import EvalSet
from recsys_rl.model import to_tensors
from recsys_rl.reward_model import (
    HEAD_INDEX,
    HEADS,
    FeedbackSet,
    RewardModel,
    feedback_metrics,
    feedback_targets,
    predict,
)

POPULARITY_EDGES = (100, 1_000, 10_000, 100_000)  # train listens of the item
SESSION_POSITION_EDGES = (0, 4, 19)  # 0-based index in the session: 1st, 2-5, 6-20, 21+
REPEAT_COUNTS = (0, 1, 2, 3, 5, 10)  # one value per repeat bin of the reward model


def baseline_predictions(
    store: EventStore,
    bounds: SplitBounds,
    fs: FeedbackSet,
    data_cfg: DataConfig,
    skip_threshold: float,
    prior: float = 10.0,
    chunk: int = 5_000_000,
) -> dict[str, np.ndarray]:
    """``[N, H]`` predictions of the train-mean baselines for the listens of ``fs``.

    Means are over train listens (labels observed before ``train_end``) and shrunk towards the
    global mean with ``prior`` pseudo-listens: ``(sum + prior * global) / (count + prior)``.
    """
    ts = store.columns["timestamp"]
    n_rows = len(ts)
    n_users, n_items = store.n_users, int(np.asarray(store.columns["item_idx"]).max()) + 1
    row_user = np.repeat(np.arange(n_users), np.diff(store.offsets))
    sums = {"user": np.zeros((n_users, len(HEADS))), "item": np.zeros((n_items, len(HEADS)))}
    counts = {k: np.zeros_like(v) for k, v in sums.items()}
    for lo in range(0, n_rows, chunk):
        sl = slice(lo, min(lo + chunk, n_rows))
        cols = {name: np.asarray(values[sl]) for name, values in store.columns.items()}
        targets, valid = feedback_targets(cols, bounds.train_end, data_cfg, skip_threshold)
        valid &= (cols["timestamp"] < bounds.train_end)[:, None]
        keys = {"user": row_user[sl], "item": cols["item_idx"]}
        for h in range(len(HEADS)):
            v = valid[:, h]
            for name, key in keys.items():
                size = len(sums[name])
                sums[name][:, h] += np.bincount(key[v], weights=targets[v, h], minlength=size)
                counts[name][:, h] += np.bincount(key[v], minlength=size)
    glob = sums["user"].sum(0) / np.maximum(counts["user"].sum(0), 1)
    means = {k: (sums[k] + prior * glob) / (counts[k] + prior) for k in sums}
    user = means["user"][row_user[fs.rows]]
    item = means["item"][np.asarray(store.columns["item_idx"])[fs.rows]]
    return {
        "global": np.broadcast_to(glob, (fs.n_events, len(HEADS))).astype(np.float32),
        "user_mean": user.astype(np.float32),
        "item_mean": item.astype(np.float32),
        "user+item": np.clip(user + item - glob, 1e-4, 1 - 1e-4).astype(np.float32),
    }


def quality_table(preds: Mapping[str, np.ndarray], fs: FeedbackSet) -> pl.DataFrame:
    """One row per method: ``feedback_metrics`` on the listens of ``fs``."""
    rows = []
    for method, pred in preds.items():
        row = {"split": fs.name, "method": method, "events": int(fs.valid[:, 0].sum())}
        rows.append(row | feedback_metrics(pred, fs.targets, fs.valid))
    return pl.DataFrame(rows)


def _bin_labels(edges: Sequence[int], unit: str = "") -> list[str]:
    labels = [f"<={edges[0]}{unit}"]
    labels += [f"{a + 1}-{b}{unit}" for a, b in zip(edges[:-1], edges[1:], strict=True)]
    return labels + [f">{edges[-1]}{unit}"]


def history_features(store: EventStore) -> pl.DataFrame:
    """Per store row: ``is_repeat`` (the user listened to the item before) and
    ``session_position`` (0-based index of the listen in its session)."""
    df = pl.DataFrame(
        {
            "u": np.repeat(np.arange(store.n_users), np.diff(store.offsets)),
            "item": np.asarray(store.columns["item_id"]),
            "s": np.asarray(store.columns["session_id"]),
        }
    )
    return df.select(
        is_repeat=pl.int_range(pl.len()).over("u", "item") > 0,
        session_position=pl.int_range(pl.len()).over("u", "s"),
    )


def event_slices(
    store: EventStore,
    items: pl.DataFrame,
    fs: FeedbackSet,
    data_cfg: DataConfig,
    history: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Slice labels of the listens of ``fs`` (``history`` from ``history_features`` can be
    passed to reuse it between splits)."""
    history = history if history is not None else history_features(store)
    hist = history[fs.rows]
    idx = np.asarray(store.columns["item_idx"])[fs.rows]
    count = np.zeros(N_SPECIAL_TOKENS + len(items))
    count[items["idx"].to_numpy()] = items["count"].to_numpy()
    pop_bin = (count[idx][:, None] > np.asarray(POPULARITY_EDGES)[None, :]).sum(1)
    length_bin = np.asarray(store.columns["length_bin"])[fs.rows]
    pos = hist["session_position"].to_numpy()
    pos_bin = (pos[:, None] > np.asarray(SESSION_POSITION_EDGES)[None, :]).sum(1)
    pop_labels = _bin_labels(POPULARITY_EDGES)
    pos_labels = ["1st", "2-5", "6-20", ">20"]
    return pl.DataFrame(
        {
            "length": np.array(_bin_labels(data_cfg.length_bin_edges, "s"))[length_bin],
            "popularity": np.array(pop_labels)[pop_bin],
            "repeat": np.where(hist["is_repeat"].to_numpy(), "repeat", "first"),
            "organic": np.where(
                np.asarray(store.columns["is_organic"])[fs.rows] == 1, "organic", "recommended"
            ),
            "session_position": np.array(pos_labels)[pos_bin],
        }
    )


SLICE_METRICS = ("ratio_mae", "ratio_mean_true", "ratio_mean_pred", "listen_plus_auc", "like_auc")


def slice_table(
    preds: Mapping[str, np.ndarray],
    fs: FeedbackSet,
    slices: pl.DataFrame,
    metric_names: Sequence[str] = SLICE_METRICS,
) -> pl.DataFrame:
    """Long table ``slice, value, method, events, <metrics>`` over the columns of ``slices``."""
    rows = []
    for column in slices.columns:
        labels = slices[column].to_numpy()
        for value in sorted(set(labels)):
            sel = labels == value
            for method, pred in preds.items():
                m = feedback_metrics(pred[sel], fs.targets[sel], fs.valid[sel])
                row = {"slice": column, "value": value, "method": method}
                row["events"] = int(fs.valid[sel, 0].sum())
                rows.append(row | {k: m.get(k, float("nan")) for k in metric_names})
    return pl.DataFrame(rows)


def calibration_table(
    preds: Mapping[str, np.ndarray], fs: FeedbackSet, head: str, n_bins: int = 10
) -> pl.DataFrame:
    h = HEAD_INDEX[head]
    v = fs.valid[:, h]
    frames = []
    for method, pred in preds.items():
        c = metrics.calibration_bins(fs.targets[v, h], pred[v, h], n_bins)
        frames.append(
            pl.DataFrame(c | {"bin": np.arange(n_bins)}).with_columns(
                method=pl.lit(method), head=pl.lit(head)
            )
        )
    return pl.concat(frames).select("method", "head", "bin", "count", "mean_pred", "mean_true")


def uncertainty_table(
    probs: np.ndarray, fs: FeedbackSet, head: str = "ratio", n_bins: int = 5
) -> pl.DataFrame:
    """Listens grouped by quantiles of the ensemble std (``probs`` is ``[E, N, H]``): if the
    spread is informative, the error of the mean prediction grows with it."""
    h = HEAD_INDEX[head]
    v = fs.valid[:, h]
    std, mean, y = probs[:, v, h].std(0), probs[:, v, h].mean(0), fs.targets[v, h]
    edges = np.quantile(std, np.linspace(0, 1, n_bins + 1)[1:-1])
    b = np.searchsorted(edges, std, side="right")
    return (
        pl.DataFrame({"bin": b, "std": std, "abs_err": np.abs(mean - y), "mean_pred": mean, "y": y})
        .group_by("bin")
        .agg(
            events=pl.len(),
            mean_std=pl.col("std").mean(),
            mae=pl.col("abs_err").mean(),
            mean_pred=pl.col("mean_pred").mean(),
            mean_true=pl.col("y").mean(),
        )
        .sort("bin")
    )


def sensitivity(
    rm: RewardModel,
    store: EventStore,
    fs: FeedbackSet,
    n_length_bins: int,
    heads: Sequence[str] = ("ratio", "listen_plus", "like", "skip"),
    amp_dtype: torch.dtype | None = None,
) -> pl.DataFrame:
    """Mean predictions over ``fs`` when one candidate feature is set to each value for every
    listen (all other inputs unchanged): a partial-dependence view of the explicit features.
    The item embedding still carries the item's own length / popularity, so this is the
    *additional* effect of the feature; ``slice_table`` shows the total one."""
    grid = [("length_bin", v) for v in range(n_length_bins)]
    grid += [("repeat_count", v) for v in REPEAT_COUNTS]
    grid += [("is_organic", v) for v in (0, 1)]
    rows = []
    for feature, value in grid:
        probs = predict(rm, store, fs, amp_dtype=amp_dtype, overrides={feature: value})
        mean = probs.mean(0)
        row = {"feature": feature, "value": value}
        for head in heads:
            h = HEAD_INDEX[head]
            row[head] = float(mean[fs.valid[:, h], h].mean())
        rows.append(row)
    return pl.DataFrame(rows)


@torch.no_grad()
def catalog_scores(
    rm: RewardModel,
    batch: Mapping[str, torch.Tensor],
    head: str = "ratio",
    item_chunk: int = 32_768,
) -> torch.Tensor:
    """``[B, n_items]`` ensemble-mean prediction of ``head`` for every vocabulary item
    recommended after each context (specials get NaN)."""
    hidden = rm.last_hidden(batch)[:, None]  # [B, 1, d]
    n_items, b = rm.encoder.n_items, len(hidden)
    counts = torch.zeros(b, n_items, device=hidden.device)
    counts.scatter_add_(1, batch["item_idx"], batch["mask"].float())
    out = torch.full((b, n_items), float("nan"), device=hidden.device)
    for lo in range(N_SPECIAL_TOKENS, n_items, item_chunk):
        cand = torch.arange(lo, min(lo + item_chunk, n_items), device=hidden.device)
        cand = cand[None].expand(b, -1)
        logits = rm.head_logits(
            hidden, cand, torch.zeros_like(cand), counts[:, lo : lo + item_chunk].long()
        )
        out[:, lo : lo + cand.shape[1]] = torch.sigmoid(logits[..., HEAD_INDEX[head]].float()).mean(
            0
        )
    return out


def catalog_profile(
    rm: RewardModel,
    store: EventStore,
    items: pl.DataFrame,
    eval_set: EvalSet,
    n_users: int = 512,
    k: int = 100,
    head: str = "ratio",
    batch_size: int = 8,
    seed: int = 0,
    amp_dtype: torch.dtype | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """RM preferences over the whole vocabulary for the frozen contexts of ``eval_set``
    (as in top-K evaluation and RL: candidates are recommendations, ``is_organic = 0``).

    Returns ``(per_item, top_k)``: the mean prediction of every item over the sampled contexts
    (with its train count and median length), and per context the composition of the RM's
    top-``k`` vs the whole catalogue (mean track length, mean log10 train count, share of items
    from the context).
    """
    rm.eval()
    device = next(rm.parameters()).device
    rng = np.random.default_rng(seed)
    users = np.sort(rng.choice(eval_set.n_users, min(n_users, eval_set.n_users), replace=False))
    idx = items["idx"].to_numpy()
    length = np.full(rm.encoder.n_items, np.nan)
    length[idx] = items["track_length"].to_numpy()
    log_count = np.full(rm.encoder.n_items, np.nan)
    log_count[idx] = np.log10(items["count"].to_numpy())
    total = torch.zeros(rm.encoder.n_items, dtype=torch.float64)
    top_rows = []
    for b in range(0, len(users), batch_size):
        sel = users[b : b + batch_size]
        starts, ends = eval_set.context_starts[sel], eval_set.context_ends[sel]
        batch = to_tensors(
            store.gather(starts, ends - starts, rm.max_seq_len, eval_set.cutoff), device
        )
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            scores = catalog_scores(rm, batch, head)
        total += scores.nan_to_num(0).sum(0).double().cpu()
        top = scores.nan_to_num(-1).topk(k, dim=1).indices.cpu().numpy()
        context = batch["item_idx"].cpu().numpy()
        for row, ctx in zip(top, context, strict=True):
            heard = np.unique(ctx[ctx >= N_SPECIAL_TOKENS])
            top_rows.append(
                {
                    "top_length": float(np.nanmean(length[row])),
                    "top_log_count": float(np.nanmean(log_count[row])),
                    "top_repeat_share": float(np.isin(row, heard).mean()),
                    "catalog_repeat_share": len(heard) / len(idx),
                }
            )
    per_item = items.select("idx", "item_id", "count", "track_length").with_columns(
        mean_score=pl.Series((total / len(users)).numpy()[idx])
    )
    top_k = pl.DataFrame(top_rows).with_columns(
        catalog_length=pl.lit(float(np.nanmean(length))),
        catalog_log_count=pl.lit(float(np.nanmean(log_count))),
    )
    return per_item, top_k
