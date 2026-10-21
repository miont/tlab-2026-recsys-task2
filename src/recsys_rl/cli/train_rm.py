"""Train a reward model (stage 3) and compare it with the train-mean baselines.

Usage::

    uv run yambda-train-rm run_name=rm_proxy                     # encoder from runs/base/best.pt
    uv run yambda-train-rm run_name=rm_gold seed=7 rm.base_run=null \\
        rm.audio_embeddings=true rm.user_fraction=0.8            # gold RM (evaluation only)
    uv run yambda-train-rm run_name=rm_smoke rm.max_steps=50 rm.eval_every=25

A run resumes from ``runs/<run_name>/last.pt`` if it exists. The quality table (val and test,
RM vs global / user / item / user+item means) is written to ``runs/<run_name>/results.csv``.
"""

from __future__ import annotations

import argparse

import polars as pl

from recsys_rl import rm_eval
from recsys_rl.config import load_config, parse_overrides
from recsys_rl.data import prepare
from recsys_rl.reward_model import (
    build_feedback_set,
    load_reward_model,
    predict,
    train_reward_model,
)
from recsys_rl.train import amp_dtype
from recsys_rl.utils import get_device

COLUMNS = (
    "split",
    "method",
    "events",
    "loss",
    "ratio_mae",
    "ratio_spearman",
    "listen_plus_auc",
    "skip_auc",
    "like_auc",
    "like_pr_auc",
    "dislike_pr_auc",
    "continue_auc",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("overrides", nargs="*", help="key=value, e.g. rm.lr=1e-4")
    args = parser.parse_args()

    cfg = load_config(args.config, **parse_overrides(args.overrides))
    device = get_device()
    prepared = prepare(cfg)
    rm = load_reward_model(train_reward_model(cfg, prepared, device), device)
    dtype = amp_dtype(device, cfg.rm.amp)
    store = prepared.store.in_memory()
    tables = []
    for split in ("val", "test"):
        fs = build_feedback_set(
            store, prepared.bounds, split, rm.max_seq_len, cfg.data, cfg.rm.skip_threshold
        )
        preds = {cfg.run_name: predict(rm, store, fs, amp_dtype=dtype).mean(0)}
        preds |= rm_eval.baseline_predictions(
            store, prepared.bounds, fs, cfg.data, cfg.rm.skip_threshold
        )
        tables.append(rm_eval.quality_table(preds, fs))
    table = pl.concat(tables)
    table.write_csv(cfg.run_dir / "results.csv")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, float_precision=4):
        print(table.select(COLUMNS))


if __name__ == "__main__":
    main()
