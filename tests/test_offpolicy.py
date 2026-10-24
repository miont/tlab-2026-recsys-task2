import copy
import json

import numpy as np
import pytest
import torch
from test_reward_model import make_cfg
from test_rl import tiny_models, train_base_and_rms

from recsys_rl.config import RLConfig
from recsys_rl.data import PAD_IDX
from recsys_rl.model import load_model
from recsys_rl.ope import ope_eval, ope_terms
from recsys_rl.reward_model import HEAD_INDEX, HEADS
from recsys_rl.rl import logged_actions, offpolicy_loss, offpolicy_terms, train_rl
from recsys_rl.train import train


def window_batch():
    item_idx = torch.tensor([[2, 3, 4, 5, 6, 7, 8, PAD_IDX], [9, 1, 10, 11, 2, 3, 4, 5]])
    return {
        "item_idx": item_idx,
        "mask": item_idx != PAD_IDX,
        "is_organic": torch.tensor([[0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]),
        "session_id": torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0]]),
        **{c: torch.zeros_like(item_idx) for c in ("ratio_bin", "length_bin", "liked", "disliked")},
    }


def test_logged_actions():
    action, trajectory = logged_actions(window_batch())
    # row 0: position 0 has no context, 3 is organic, 5 starts a new session, 7 is padding
    assert action[0].tolist() == [False, True, True, False, True, True, True, False]
    t0 = trajectory[0].tolist()
    assert t0[1] == t0[2] and len({t0[1], t0[4], t0[5]}) == 3 and t0[5] == t0[6]
    # row 1: item 1 is OOV (a break), the rest is one session
    assert action[1].tolist() == [False, False, True, True, True, True, True, True]
    assert len(set(trajectory[1, 2:].tolist())) == 1 and set(t0) & set(trajectory[1].tolist()) == {
        -1
    }


def test_offpolicy_terms():
    trajectory = torch.tensor([[-1, 0, 0, 0, -1, 1, 1]])
    reward = torch.tensor([[9.0, 1.0, 0.0, 2.0, 9.0, 1.0, 1.0]])
    log_ratio = torch.log(torch.tensor([[1.0, 2.0, 0.5, 3.0, 7.0, 0.25, 4.0]]))
    gamma = 0.5
    coef, ret, weight = offpolicy_terms(log_ratio, reward, trajectory, gamma, k=2, cap=10.0)
    # returns within each trajectory, nothing outside
    assert ret[0].tolist() == pytest.approx([0, 1 + 0.5 * 0 + 0.25 * 2, 0 + 0.5 * 2, 2, 0, 1.5, 1])
    # K = 2: the current and the next factor of the same trajectory
    assert weight[0].tolist() == pytest.approx([0, 2 * 0.5, 0.5 * 3, 3, 0, 0.25 * 4, 4])
    # gamma^s with s the index in the trajectory
    assert coef[0].tolist() == pytest.approx([0, 1.0, 0.5 * 1.5, 0.25 * 3, 0, 1.0, 0.5 * 4])
    capped, _, _ = offpolicy_terms(log_ratio, reward, trajectory, gamma, k=2, cap=0.5)
    assert capped[0].tolist() == pytest.approx([0, 0.5, 0.25, 0.125, 0, 0.5, 0.25])
    one, _, w1 = offpolicy_terms(log_ratio, reward, trajectory, gamma, k=1, cap=10.0)
    assert w1[0].tolist() == pytest.approx([0, 2, 0.5, 3, 0, 0.25, 4])  # Chen et al.: one factor
    none, _, w0 = offpolicy_terms(log_ratio, reward, trajectory, gamma, k=0, cap=10.0)
    assert w0[0].tolist() == [0, 1, 1, 1, 0, 1, 1]  # no correction


def test_offpolicy_loss():
    policy, ref, _ = tiny_models()
    behavior = copy.deepcopy(ref)
    batch = window_batch()
    targets = torch.zeros(*batch["item_idx"].shape, len(HEADS))
    targets[..., HEAD_INDEX["ratio"]] = torch.rand(
        batch["item_idx"].shape, generator=torch.Generator().manual_seed(0)
    )
    targets[0, 2, HEAD_INDEX["like"]] = 1.0
    cfg = RLConfig(method="offpolicy", kl_beta=0.1, offpolicy_cap=0.5)
    loss, stats = offpolicy_loss(policy, ref, behavior, batch, targets, cfg)
    assert stats["log_ratio"] == pytest.approx(0, abs=1e-6) and stats["weight"] == pytest.approx(1)
    assert stats["capped_share"] == 1.0 and stats["kl"] == pytest.approx(0, abs=1e-6)
    assert stats["n_actions"] == 11  # the actions of test_logged_actions
    loss.backward()
    assert all(p.grad is None for p in behavior.parameters()) and all(
        p.grad is None for p in ref.parameters()
    )
    # weights stay capped (pi_theta ~ pi_b), so the objective is a fixed return-weighted NLL of
    # the logged actions, and gradient steps lower it
    cfg = RLConfig(method="offpolicy", kl_beta=0.0, offpolicy_cap=0.5)
    first = offpolicy_loss(policy, ref, behavior, batch, targets, cfg)[1]["objective"]
    opt = torch.optim.SGD(policy.parameters(), lr=0.1)
    for _ in range(10):
        loss, stats = offpolicy_loss(policy, ref, behavior, batch, targets, cfg)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert stats["objective"] < first and stats["capped_share"] == 1.0
    with pytest.raises(ValueError):
        offpolicy_loss(policy, ref, behavior, batch, targets, RLConfig(offpolicy_reward="clicks"))


def test_behavior_model_and_offpolicy_run(tmp_path):
    prepared, device = train_base_and_rms(tmp_path)
    behavior_cfg = make_cfg(tmp_path, run_name="behavior")
    behavior_cfg.train.target, behavior_cfg.train.init_run = "recommended", "base"
    behavior_cfg.train.max_steps = 2
    base = load_model(tmp_path / "runs" / "base" / "best.pt")
    path = train(behavior_cfg, prepared, device)
    behavior = load_model(path)
    assert behavior.n_items == base.n_items
    cfg = make_cfg(tmp_path, run_name="offpolicy")
    cfg.rl = RLConfig(
        method="offpolicy", kl_beta=0.0, batch_size=4, steps=2, eval_every=2, warmup_steps=1,
        amp=False, lr=1e-2,
    )  # fmt: skip
    train_rl(cfg, prepared, device)
    log = [json.loads(line) for line in (cfg.run_dir / "log.jsonl").read_text().splitlines()]
    expected = {
        "train_weight",
        "train_capped_share",
        "train_ess_share",
        "train_return",
        "val_kl_ref",
    }
    assert expected <= set(log[-1]) and log[-1]["val_kl_ref"] > 0
    assert all(np.isfinite(v) for r in log for v in r.values())


def test_ope_terms():
    trajectory = torch.tensor([[-1, 0, 0, 0, -1, 1, 1]])
    log_ratio = torch.log(torch.tensor([[7.0, 2.0, 0.5, 3.0, 7.0, 0.25, 4.0]]))
    discount, cum, start = ope_terms(log_ratio, torch.zeros(1, 7), trajectory, gamma=0.5)
    assert discount[0].tolist() == pytest.approx([0, 1, 0.5, 0.25, 0, 1, 0.5])
    assert cum[0].exp().tolist()[1:4] == pytest.approx([2, 1, 3])  # cumulative from the start
    assert cum[0, 5:].exp().tolist() == pytest.approx([0.25, 1])
    assert start[0].tolist() == [False, True, False, False, False, True, False]


def test_ope_eval(tmp_path):
    prepared, device = train_base_and_rms(tmp_path)
    cfg = make_cfg(tmp_path, run_name="ope")
    cfg.rl = RLConfig(amp=False, behavior_run="base")  # pi_b = pi_ref here
    table = ope_eval(cfg, prepared, device, split="val", policies={"pi_ref": None}, gamma=0.9)
    assert table["policy"].to_list() == ["pi_b", "pi_ref"]
    for row in table.iter_rows(named=True):  # the same model as pi_b: every estimate = logs
        assert row["step_is"] == pytest.approx(row["behavior"])
        assert row["snips"] == pytest.approx(row["behavior"])
        assert row["log_ratio"] == pytest.approx(0, abs=1e-6) and row["ess_share"] == pytest.approx(
            1
        )
    assert table["n_actions"][0] > 0 and table["behavior"][0] > 0
    other = ope_eval(cfg, prepared, device, split="test", policies={"rl": None}, reward="ratio")
    assert other["reward"][0] == "ratio" and other["step_is"].is_finite().all(), other
