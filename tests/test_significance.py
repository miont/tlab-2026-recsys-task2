import math

import numpy as np
import polars as pl
import pytest
from test_reward_model import make_cfg
from test_rl import train_base_and_rms

from recsys_rl import evaluate, metrics
from recsys_rl.config import RLConfig
from recsys_rl.significance import bootstrap_table, collect_per_user


def test_relevance_per_user_matches_means():
    rng = np.random.default_rng(0)
    recs = np.stack([rng.permutation(50)[:20] for _ in range(30)])
    targets = [rng.integers(0, 50, rng.integers(1, 6)) for _ in range(30)]
    per_user = metrics.relevance_per_user(recs, targets, ks=(5, 20))
    means = metrics.relevance_metrics(recs, targets, ks=(5, 20))
    assert per_user.keys() == means.keys()
    for name, values in per_user.items():
        assert values.shape == (30,) and np.mean(values) == pytest.approx(means[name])


def per_user_frame(values: dict[str, np.ndarray], split: str = "val") -> pl.DataFrame:
    n = len(next(iter(values.values())))
    return pl.concat(
        [
            pl.DataFrame(
                {"split": split, "method": m, "exclude_seen": False, "user": np.arange(n), "x": v}
            )
            for m, v in values.items()
        ]
    )


def test_bootstrap_table():
    rng = np.random.default_rng(1)
    base = rng.normal(0.5, 0.2, 2000)
    frame = per_user_frame(
        {
            "base": base,
            "better": base + 0.05 + rng.normal(0, 0.1, 2000),  # clear paired gain
            "same": base + rng.normal(0, 0.1, 2000),  # no gain
        }
    )
    table = bootstrap_table(
        frame, [("better", "base"), ("same", "base"), ("x", "base")], n_boot=2000
    )
    assert table["method"].to_list() == ["better", "same"]  # the unknown method is skipped
    better, same = table.row(0, named=True), table.row(1, named=True)
    assert better["diff"] == pytest.approx(better["value"] - better["baseline_value"])
    assert better["ci_low"] > 0 and better["p_value"] < 0.01
    assert same["ci_low"] < 0 < same["ci_high"] and same["p_value"] > 0.05
    assert better["ci_low"] < better["diff"] < better["ci_high"]
    # the CI of a mean difference shrinks like 1 / sqrt(n)
    assert better["ci_high"] - better["ci_low"] == pytest.approx(
        2 * 1.96 * 0.1 / math.sqrt(2000), rel=0.15
    )


def test_collect_per_user(tmp_path):
    prepared, device = train_base_and_rms(tmp_path)
    cfg = make_cfg(tmp_path, run_name="significance")
    cfg.rl = RLConfig(amp=False)
    base_path = tmp_path / "runs" / "base" / "best.pt"
    per_user = collect_per_user(
        cfg,
        prepared,
        device,
        policies={"copy": base_path},
        rerank={
            "rr_inf": ("ratio", math.inf),
            "rr_0.1": ("ratio", 0.1),
            "r4_0.1": ("debiased_ratio", 0.1),
        },
        n_candidates=30,
        k=20,
        exclude_seen=False,  # every synthetic user has heard every item
    )
    methods = per_user.filter(~pl.col("exclude_seen"))["method"].unique(maintain_order=True)
    assert methods.to_list() == ["Repeat", "pi_ref", "copy", "rr_inf", "rr_0.1", "r4_0.1"]
    assert not per_user["exclude_seen"].any()
    means = per_user.group_by("method", "exclude_seen").agg(pl.exclude(["split", "user"]).mean())

    def mean(method: str, column: str, exclude: bool = False) -> float:
        row = means.filter((pl.col("method") == method) & (pl.col("exclude_seen") == exclude))
        return row[column].item()

    # pi_ref matches the stage-2 evaluation, the reranker at beta = inf is pi_ref
    store = prepared.store.in_memory()
    es = evaluate.build_eval_set(store, prepared.bounds, "val", cfg.data.max_seq_len, 0.5)
    ref = evaluate.load_model(base_path, device)
    recs = evaluate.recommend_model(
        ref, store, es, evaluate.idx_to_item_array(prepared.items), k=20
    )
    assert mean("pi_ref", "ndcg@10") == pytest.approx(
        metrics.relevance_metrics(recs, es.targets, ks=(10, 20))["ndcg@10"]
    )
    for column in ("ndcg@10", "recall@20", "gold_ratio@10", "proxy_debiased@10"):
        assert mean("rr_inf", column) == pytest.approx(mean("pi_ref", column))
    assert mean("pi_ref", "kl_ref") == pytest.approx(0, abs=1e-6)
    assert mean("rr_0.1", "kl_ref") > 0
    assert mean("rr_0.1", "proxy_ratio@10") >= mean("pi_ref", "proxy_ratio@10")
    assert mean("Repeat", "repeat@10") is None  # relevance only

    table = bootstrap_table(per_user, [("copy", "pi_ref"), ("Repeat", "pi_ref")], n_boot=200)
    copy = table.filter(pl.col("method") == "copy")
    assert (copy["diff"].abs() < 1e-6).all() and (copy["p_value"] == 1).all()
    assert set(table.filter(pl.col("method") == "Repeat")["metric"]) >= {"ndcg@10", "recall@20"}
    assert "gold_ratio@10" not in set(table.filter(pl.col("method") == "Repeat")["metric"])
