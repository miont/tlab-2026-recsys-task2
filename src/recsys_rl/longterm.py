"""Offline validity of the long-term reward proxies (stage 6.2, plan §7.1).

R6 rewards the RM's ``continue`` head, p(the user's next listen starts within
``session_gap_minutes`` of the start of this one). Whether such a one-step proxy says anything
about longer-term behaviour is checked on the listens of the val / test day
(``reward_model.FeedbackSet``) against outcomes observed in the logs:

- ``session_outcomes``: per listen, the ``continue`` label, the number of later listens of its
  session and the minutes until the session's last listen start (both before the period cutoff;
  ``censored`` marks sessions whose end may lie beyond it), and, when a later period exists
  (val), whether the user listens again in it (``returns``);
- ``validity_table``: for every predictor (method x RM head) and outcome, the pooled Spearman
  correlation, the AUC for binary outcomes and the *within-user* correlation (Pearson after
  subtracting per-user means of both). For a fixed context a policy can only change the
  within-user part, so a proxy that only separates active from inactive users is useless as a
  reward even with a good pooled AUC;
- ``user_return_table``: per user, the mean prediction over the day against returning on the
  next period, next to the number of listens of the day as an activity baseline;
- ``decile_table``: mean outcomes by decile of a prediction, pooled or within-user;
- ``length_table``: label and predictions by track-length bin. ``continue`` is measured from the
  start of the listen, so long tracks cannot "continue" within the gap whatever the user does:
  a reward on it favours short tracks, like R1.

Everything here is observational: a high predicted ``continue`` going with longer sessions does
not mean that recommending such tracks makes sessions longer (that needs an A/B test).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from recsys_rl import metrics
from recsys_rl.data import EventStore
from recsys_rl.reward_model import HEAD_INDEX, FeedbackSet

OUTCOMES = ("continue", "remaining_listens", "remaining_minutes", "returns")
BINARY_OUTCOMES = ("continue", "returns")


def session_outcomes(
    store: EventStore, fs: FeedbackSet, session_gap_minutes: int, return_from: int | None = None
) -> pl.DataFrame:
    """One row per listen of ``fs`` (in ``fs.rows`` order): ``user``, ``length_bin``,
    ``session_position`` (listens of the session before it within the period), the outcomes
    of the module docstring and ``valid_continue``. ``returns`` (the user has a listen at or
    after ``return_from``) is null without ``return_from``."""
    rows = fs.rows
    user = np.searchsorted(store.offsets, rows, side="right") - 1
    ts = np.asarray(store.columns["timestamp"])[rows].astype(np.int64)
    gap = session_gap_minutes * 60
    df = pl.DataFrame(
        {
            "user": user,
            "session": np.asarray(store.columns["session_id"])[rows],
            "ts": ts,
            "length_bin": np.asarray(store.columns["length_bin"])[rows],
            "continue": fs.targets[:, HEAD_INDEX["continue"]],
            "valid_continue": fs.valid[:, HEAD_INDEX["continue"]],
        }
    )
    session = ["user", "session"]
    last_ts = pl.col("ts").max().over(session)
    df = df.with_columns(
        session_position=pl.int_range(pl.len()).over(session),
        remaining_listens=pl.len().over(session) - 1 - pl.int_range(pl.len()).over(session),
        remaining_minutes=(last_ts - pl.col("ts")) / 60,
        censored=last_ts + gap >= fs.cutoff,
    )
    if return_from is None:
        returns = pl.lit(None, dtype=pl.Boolean)
    else:
        last = np.asarray(store.columns["timestamp"])[store.offsets[1:] - 1].astype(np.int64)
        returns = pl.Series(last[user] >= return_from)
    return df.with_columns(returns=returns).drop("session", "ts")


def _outcome(outcomes: pl.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray]:
    """``(values, mask)`` of an outcome: ``continue`` where its label is known, session lengths
    (log1p) of uncensored sessions, ``returns`` where defined."""
    if name == "continue":
        return outcomes["continue"].to_numpy(), outcomes["valid_continue"].to_numpy()
    if name == "returns":
        values = outcomes["returns"]
        return values.fill_null(False).to_numpy().astype(
            np.float64
        ), values.is_not_null().to_numpy()
    values = np.log1p(outcomes[name].to_numpy().astype(np.float64))
    return values, ~outcomes["censored"].to_numpy()


def _demean(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    return values - (sums / counts)[inverse]


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def validity_table(
    preds: Mapping[str, np.ndarray],
    fs: FeedbackSet,
    outcomes: pl.DataFrame,
    heads: Sequence[str] = ("continue", "ratio", "like", "skip"),
) -> pl.DataFrame:
    """One row per (method, head, outcome) for ``[N, H]`` predictions on ``fs``: ``events``,
    ``spearman`` (pooled), ``within_user`` correlation and ``auc`` (binary outcomes)."""
    user = outcomes["user"].to_numpy()
    rows = []
    for outcome in OUTCOMES:
        y_all, mask = _outcome(outcomes, outcome)
        if not mask.any():
            continue
        for method, pred in preds.items():
            for head in heads:
                h = HEAD_INDEX[head]
                m = mask & fs.valid[:, h]
                x, y = pred[m, h].astype(np.float64), y_all[m]
                varies = len(x) > 1 and np.ptp(x) > 0 and np.ptp(y) > 0
                row = {
                    "split": fs.name,
                    "method": method,
                    "head": head,
                    "outcome": outcome,
                    "events": int(m.sum()),
                    "spearman": float(spearmanr(x, y).statistic) if varies else np.nan,
                    "within_user": _pearson(_demean(x, user[m]), _demean(y, user[m])),
                    "auc": metrics.roc_auc(y, x) if outcome in BINARY_OUTCOMES else np.nan,
                }
                rows.append(row)
    return pl.DataFrame(rows)


def user_return_table(
    preds: Mapping[str, np.ndarray],
    fs: FeedbackSet,
    outcomes: pl.DataFrame,
    heads: Sequence[str] = ("continue", "ratio", "like", "skip"),
) -> pl.DataFrame:
    """AUC of the per-user mean prediction of the day for returning in the next period, one row
    per (method, head), plus ``activity`` = the number of the user's listens of the day."""
    if outcomes["returns"].is_null().all():
        raise ValueError("outcomes have no 'returns' (no later period)")
    user = outcomes["user"].to_numpy()
    users, inverse = np.unique(user, return_inverse=True)
    returns = np.zeros(len(users))
    returns[inverse] = outcomes["returns"].to_numpy()
    counts = np.bincount(inverse)
    rows = [
        {
            "split": fs.name,
            "method": "activity",
            "head": "n_listens",
            "users": len(users),
            "return_rate": float(returns.mean()),
            "auc": metrics.roc_auc(returns, counts.astype(np.float64)),
        }
    ]
    for method, pred in preds.items():
        for head in heads:
            h = HEAD_INDEX[head]
            v = fs.valid[:, h]
            sums = np.bincount(inverse[v], weights=pred[v, h], minlength=len(users))
            n = np.bincount(inverse[v], minlength=len(users))
            has = n > 0
            mean = sums[has] / n[has]
            rows.append(
                {
                    "split": fs.name,
                    "method": method,
                    "head": head,
                    "users": int(has.sum()),
                    "return_rate": float(returns[has].mean()),
                    "auc": metrics.roc_auc(returns[has], mean),
                }
            )
    return pl.DataFrame(rows)


def decile_table(
    pred: np.ndarray, valid: np.ndarray, outcomes: pl.DataFrame, within_user: bool = False
) -> pl.DataFrame:
    """Mean outcomes by decile of a ``[N]`` prediction (where ``valid``); with
    ``within_user`` the deciles are of the prediction minus the user's mean, so they compare
    a user's own listens."""
    x = pred.astype(np.float64)
    if within_user:
        x = np.where(valid, x, 0.0)
        x[valid] = _demean(x[valid], outcomes["user"].to_numpy()[valid])
    df = outcomes.with_columns(pred=pl.Series(x)).filter(pl.Series(valid))
    df = df.with_columns(decile=(pl.col("pred").rank("ordinal") * 10 / (pl.len() + 1)).floor())
    lengths = ~pl.col("censored")
    return (
        df.group_by("decile")
        .agg(
            events=pl.len(),
            pred=pl.col("pred").mean(),
            continue_rate=pl.col("continue").filter(pl.col("valid_continue")).mean(),
            remaining_listens=pl.col("remaining_listens").filter(lengths).mean(),
            remaining_minutes=pl.col("remaining_minutes").filter(lengths).mean(),
            return_rate=pl.col("returns").cast(pl.Float64).mean(),
        )
        .sort("decile")
        .with_columns(pl.col("decile").cast(pl.Int64), within_user=pl.lit(within_user))
    )


def length_table(
    preds: Mapping[str, np.ndarray], fs: FeedbackSet, outcomes: pl.DataFrame
) -> pl.DataFrame:
    """Per track-length bin: listens, observed ``continue`` rate, mean remaining listens
    (uncensored sessions) and every method's mean predicted ``continue`` and ``ratio``."""
    h_cont, h_ratio = HEAD_INDEX["continue"], HEAD_INDEX["ratio"]
    df = outcomes.select(
        "length_bin", "continue", "valid_continue", "remaining_listens", "censored"
    )
    for method, pred in preds.items():
        df = df.with_columns(
            pl.Series(f"{method}_continue", np.where(fs.valid[:, h_cont], pred[:, h_cont], np.nan)),
            pl.Series(f"{method}_ratio", np.where(fs.valid[:, h_ratio], pred[:, h_ratio], np.nan)),
        )
    means = [
        pl.col(f"{m}_{head}").fill_nan(None).mean() for m in preds for head in ("continue", "ratio")
    ]
    return (
        df.group_by("length_bin")
        .agg(
            pl.len().alias("events"),
            pl.col("continue").filter(pl.col("valid_continue")).mean().alias("continue_rate"),
            pl.col("remaining_listens").filter(~pl.col("censored")).mean(),
            *means,
        )
        .sort("length_bin")
        .with_columns(split=pl.lit(fs.name))
    )
