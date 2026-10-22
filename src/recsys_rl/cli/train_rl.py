"""RL fine-tuning of the base model against the proxy reward model (stage 4).

Usage::

    uv run yambda-train-rl run_name=rl_ratio_b0.1                    # R1, beta 0.1
    uv run yambda-train-rl run_name=rl_debiased_b0.1 rl.reward=debiased_ratio
    uv run yambda-train-rl run_name=rl_smoke rl.steps=20 rl.eval_every=10

Needs ``runs/<rl.base_run>/best.pt`` and the RMs ``runs/<rl.reward_run>/best.pt``,
``runs/<rl.gold_run>/best.pt`` (``rl.gold_run=null`` skips the gold metrics). A run resumes from
``runs/<run_name>/last.pt``. The per-evaluation trajectory is ``runs/<run_name>/log.jsonl``;
the final relevance table of the base model and the policy (val and test, with and without the
repeat filter) is written to ``runs/<run_name>/results.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from recsys_rl import evaluate
from recsys_rl.config import load_config, parse_overrides
from recsys_rl.data import prepare
from recsys_rl.model import load_model
from recsys_rl.rl import train_rl
from recsys_rl.train import amp_dtype
from recsys_rl.utils import get_device


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("overrides", nargs="*", help="key=value, e.g. rl.kl_beta=0.05")
    args = parser.parse_args()

    cfg = load_config(args.config, **parse_overrides(args.overrides))
    device = get_device()
    prepared = prepare(cfg)
    policy = load_model(train_rl(cfg, prepared, device), device)
    base = load_model(Path(cfg.data_root) / "runs" / cfg.rl.base_run / "best.pt", device)
    table = evaluate.results_table(
        prepared,
        cfg.data,
        models={cfg.rl.base_run: base, cfg.run_name: policy},
        amp_dtype=amp_dtype(device, cfg.rl.amp),
    )
    table.write_csv(cfg.run_dir / "results.csv")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(table)


if __name__ == "__main__":
    main()
