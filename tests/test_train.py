import json

import numpy as np
import polars as pl
import torch

from recsys_rl import evaluate
from recsys_rl.config import (
    Config,
    DataConfig,
    ModelConfig,
    TrainConfig,
    load_config,
    parse_overrides,
)
from recsys_rl.data import DAY_SECONDS, prepare
from recsys_rl.model import load_model
from recsys_rl.train import lr_lambda, proposal_log_q, train


def write_synthetic_events(raw_dir, n_users=16, n_items=300, days=4, step=1800):
    """Each user cycles through the items with a user-specific offset, one listen per step."""
    rng = np.random.default_rng(0)
    ts = np.arange(0, days * DAY_SECONDS, step)
    rows = {
        "uid": np.repeat(np.arange(n_users), len(ts)),
        "timestamp": np.tile(ts, n_users),
        "item_id": (20 * np.arange(n_users)[:, None] + np.arange(len(ts))[None, :]) % n_items + 100,
        "played_ratio_pct": rng.choice([3, 60, 100], size=n_users * len(ts)),
    }
    events = pl.DataFrame({k: np.ravel(v) for k, v in rows.items()}).with_columns(
        pl.col("uid").cast(pl.UInt32),
        pl.col("timestamp").cast(pl.UInt32),
        pl.col("item_id").cast(pl.UInt32),
        pl.col("played_ratio_pct").cast(pl.UInt16),
        track_length_seconds=pl.lit(200, pl.UInt32),
        is_organic=pl.lit(0, pl.UInt8),
        event_type=pl.lit("listen", pl.Enum(["listen", "like", "dislike"])),
    )
    raw_dir.mkdir(parents=True)
    events.write_parquet(raw_dir / "multi_event.parquet")


def make_cfg(tmp_path, **train) -> Config:
    return Config(
        data_root=str(tmp_path),
        run_name="tiny",
        data=DataConfig(max_seq_len=16, window_stride=8, min_item_count=1),
        model=ModelConfig(n_layer=1, n_head=2, n_embd=16, dropout=0.0),
        train=TrainConfig(batch_size=8, n_negatives=16, warmup_steps=2, amp=False, **train),
    )


def test_train_resume_and_results(tmp_path):
    write_synthetic_events(tmp_path / "raw" / "flat" / "50m")
    cfg = make_cfg(tmp_path, epochs=50, eval_every=4, max_steps=8, patience=100)
    prepared = prepare(cfg)
    best = train(cfg, prepared, torch.device("cpu"))
    log = [json.loads(line) for line in (cfg.run_dir / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == [4, 8]
    assert all(np.isfinite(r["loss"]) for r in log)
    assert {"val_ndcg@10", "val_recall@100", "val_paper_ndcg@10"} <= set(log[0])

    cfg.train.max_steps = 12  # resumes from last.pt instead of starting over
    train(cfg, prepared, torch.device("cpu"))
    log = [json.loads(line) for line in (cfg.run_dir / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == [4, 8, 12]

    table = evaluate.results_table(
        prepared,
        cfg.data,
        models={"gpt2": load_model(best)},
        most_pop_days=10,
    )
    assert table.height == 2 * 5  # (MostPop, Repeat, model) x repeat-filter modes
    assert set(table["method"]) == {"MostPop", "Repeat", "gpt2"}

    only_models = evaluate.results_table(  # a checkpoint path is loaded inside
        prepared, cfg.data, models={"gpt2": best}, splits=("val",), baselines=False
    )
    assert only_models.height == 2
    assert only_models.drop("method").equals(
        table.filter(pl.col("method") == "gpt2", pl.col("split") == "val").drop("method")
    )


def test_listen_plus_target_and_early_stopping(tmp_path):
    write_synthetic_events(tmp_path / "raw" / "flat" / "50m")
    cfg = make_cfg(tmp_path, epochs=50, eval_every=1, max_steps=50, patience=1)
    cfg.train.target = "listen_plus"
    cfg.train.lr = 0.0  # no improvement after the first evaluation
    train(cfg, device=torch.device("cpu"))
    log = (cfg.run_dir / "log.jsonl").read_text().splitlines()
    assert len(log) == 2


def test_proposal_and_schedule():
    items = pl.DataFrame({"idx": [2, 3], "count": [3, 1]})
    log_q = proposal_log_q(items, 4)
    assert torch.allclose(log_q.exp(), torch.tensor([0, 0, 0.75, 0.25]))
    f = lr_lambda(warmup=10, total=100)
    assert f(0) == 0.1 * 0.5 * (1 + 1)
    assert f(9) == 1.0 * 0.5 * (1 + np.cos(np.pi * 0.09))
    assert f(100) == 0.0


def test_overrides_merge_with_yaml():
    overrides = parse_overrides(["train.lr=3e-4", "run_name=x", "model.use_feedback_tokens=false"])
    assert overrides == {
        "train": {"lr": 3e-4},
        "run_name": "x",
        "model": {"use_feedback_tokens": False},
    }
    cfg = load_config("configs/base.yaml", **overrides)
    assert (cfg.train.lr, cfg.train.batch_size, cfg.run_name) == (3e-4, 128, "x")
    assert cfg.model.use_feedback_tokens is False
