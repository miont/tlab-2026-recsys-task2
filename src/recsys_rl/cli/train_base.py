"""Train the GPT-2 base recommender (stage 2) and evaluate it against MostPop / Repeat.

Usage::

    uv run yambda-train                                   # configs/base.yaml
    uv run yambda-train --config configs/base.yaml run_name=no_feedback \\
        model.use_feedback_tokens=false train.lr=3e-4
    uv run yambda-train train.max_steps=200 train.eval_every=100   # smoke run

A run resumes from ``runs/<run_name>/last.pt`` if it exists. The final table (val and test,
with and without the repeat filter) is written to ``runs/<run_name>/results.csv``.
"""

from __future__ import annotations

import argparse

import polars as pl

from recsys_rl import evaluate
from recsys_rl.config import load_config, parse_overrides
from recsys_rl.data import prepare
from recsys_rl.model import load_model
from recsys_rl.train import amp_dtype, train
from recsys_rl.utils import get_device


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("overrides", nargs="*", help="key=value, e.g. train.lr=3e-4")
    args = parser.parse_args()

    cfg = load_config(args.config, **parse_overrides(args.overrides))
    device = get_device()
    prepared = prepare(cfg)
    best = train(cfg, prepared, device)
    model = load_model(best, device)
    table = evaluate.results_table(
        prepared,
        cfg.data,
        models={cfg.run_name: model},
        amp_dtype=amp_dtype(device, cfg.train.amp),
    )
    table.write_csv(cfg.run_dir / "results.csv")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(table)


if __name__ == "__main__":
    main()
