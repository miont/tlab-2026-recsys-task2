import math

import numpy as np
import polars as pl
import pytest
import torch
from test_reward_model import make_cfg
from test_rl import train_base_and_rms

from recsys_rl import evaluate, metrics
from recsys_rl.config import DataConfig, RLConfig
from recsys_rl.data import PAD_IDX
from recsys_rl.generate import MODES, append_listen, ratio_bins, rollout_eval, simulated_feedback
from recsys_rl.model import INPUT_COLUMNS
from recsys_rl.reward_model import HEAD_INDEX, HEADS


def test_append_listen():
    item_idx = torch.tensor([[5, 6, PAD_IDX], [7, 8, 9]])
    batch = {name: torch.zeros_like(item_idx) for name in INPUT_COLUMNS}
    batch["item_idx"] = item_idx
    batch["mask"] = item_idx != PAD_IDX
    batch["ratio_bin"] = torch.tensor([[1, 2, 0], [3, 4, 5]])
    features = {name: torch.tensor([10, 11]) for name in INPUT_COLUMNS}
    out = append_listen(batch, features)
    assert out["item_idx"].tolist() == [[5, 6, 10], [8, 9, 11]]  # the full row drops its oldest
    assert out["ratio_bin"].tolist() == [[1, 2, 10], [4, 5, 11]]
    assert out["mask"].all()
    assert batch["item_idx"].tolist() == [[5, 6, PAD_IDX], [7, 8, 9]]  # input unchanged


def test_simulated_feedback():
    edges = DataConfig().ratio_bin_edges
    assert ratio_bins(torch.tensor([0, 3, 40, 100, 150]), edges).tolist() == [0, 1, 4, 7, 8]
    items = torch.tensor([2, 3, 4])
    length_bin = torch.arange(10) % 4
    probs = torch.zeros(2, 3, 1, len(HEADS))
    probs[..., HEAD_INDEX["ratio"]] = torch.tensor([0.2, 0.6, 1.0])[None, :, None]
    probs[..., HEAD_INDEX["skip"]] = torch.tensor([1.0, 0.0, 0.0])[None, :, None]
    probs[..., HEAD_INDEX["listen_plus"]] = torch.tensor([0.0, 0.0, 1.0])[None, :, None]
    probs[..., HEAD_INDEX["like"]] = torch.tensor([0.0, 0.0, 1.0])[None, :, None]
    accept = simulated_feedback(probs, items, "accept", length_bin, edges)
    assert accept["ratio_bin"].tolist() == [7, 7, 7] and not accept["liked"].any()
    assert accept["length_bin"].tolist() == [2, 3, 0] and not accept["is_organic"].any()
    expected = simulated_feedback(probs, items, "expected", length_bin, edges)
    assert expected["ratio_bin"].tolist() == [3, 5, 7]  # 20%, 60%, 100%
    sampled = simulated_feedback(probs, items, "sampled", length_bin, edges, torch.Generator())
    assert sampled["ratio_bin"].tolist() == [1, 4, 7]  # skip, partial, Listen+ (certain here)
    assert sampled["liked"].tolist() == [0, 0, 1]
    with pytest.raises(ValueError):
        simulated_feedback(probs, items, "oracle", length_bin, edges)


def test_rollout_eval(tmp_path):
    prepared, device = train_base_and_rms(tmp_path)
    cfg = make_cfg(tmp_path, run_name="generate")
    cfg.rl = RLConfig(amp=False)
    base_path = tmp_path / "runs" / "base" / "best.pt"
    rows = rollout_eval(
        cfg, prepared, device, split="val", policies={"pi_ref": None, "copy": base_path},
        rerank={"rr": 0.1, "rr_inf": math.inf}, steps=3, n_candidates=30, k=20,
    )  # fmt: skip
    combos = rows.select("policy", "mode", "decoding").unique()
    assert len(combos) == 4 * len(MODES) * 2
    assert rows["step"].unique().sort().to_list() == [1, 2, 3]
    expected = {
        "ndcg@10", "recall@20", "kl_ref", "entropy", "hit", "in_history", "in_slate", "length",
        "log_count", "slate_distinct", "slate_artists", "proxy_ratio", "gold_ratio_std",
    }  # fmt: skip
    assert expected <= set(rows.columns)

    def get(policy, mode="accept", decoding="greedy"):
        return rows.filter(
            (pl.col("policy") == policy)
            & (pl.col("mode") == mode)
            & (pl.col("decoding") == decoding)
        ).sort("step", "user")

    ref = get("pi_ref")
    # step 1 is the ordinary evaluation
    store = prepared.store.in_memory()
    es = evaluate.build_eval_set(store, prepared.bounds, "val", cfg.data.max_seq_len, 0.5)
    model = evaluate.load_model(base_path, device)
    recs = evaluate.recommend_model(
        model, store, es, evaluate.idx_to_item_array(prepared.items), k=20
    )
    step1 = ref.filter(pl.col("step") == 1)
    assert step1["ndcg@10"].mean() == pytest.approx(
        metrics.relevance_metrics(recs, es.targets, ks=(10, 20))["ndcg@10"]
    )
    assert (rows.filter(pl.col("policy") == "pi_ref")["kl_ref"].abs() < 1e-5).all()
    assert not step1["in_slate"].any()
    assert (rows["slate_distinct"] <= rows["step"]).all()
    for policy in ("copy", "rr_inf"):  # the same distribution as pi_ref: the same greedy slate
        other = get(policy)
        assert np.allclose(other["ndcg@10"], ref["ndcg@10"])
        assert np.allclose(other["proxy_ratio"], ref["proxy_ratio"])
    rr = rows.filter((pl.col("policy") == "rr") & (pl.col("step") == 1))
    assert (rr["kl_ref"] > 0).all()
    # sampled modes are seeded: a second call repeats them
    again = rollout_eval(
        cfg, prepared, device, split="val", policies={"pi_ref": None}, modes=("sampled",),
        decodings=("sample",), steps=3, k=20,
    )  # fmt: skip
    assert np.allclose(
        again.sort("step", "user")["proxy_ratio"], get("pi_ref", "sampled", "sample")["proxy_ratio"]
    )
