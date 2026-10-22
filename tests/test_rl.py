import copy
import json

import numpy as np
import polars as pl
import pytest
import torch
from test_reward_model import MODEL_CFG, make_cfg, write_events

from recsys_rl import evaluate, metrics
from recsys_rl.config import RewardModelConfig, RLConfig
from recsys_rl.data import N_SPECIAL_TOKENS, PAD_IDX, prepare
from recsys_rl.model import GPT2Recommender, load_model
from recsys_rl.reward_model import HEAD_INDEX, HEADS, RewardModel, train_reward_model
from recsys_rl.rewards import (
    REWARDS,
    Reward,
    item_mean_ratio,
    ratio_by_length_bin,
    score_recommendations,
)
from recsys_rl.rl import (
    advantages,
    behaviour_metrics,
    dpo_loss,
    policy_eval,
    policy_loss,
    sample_contexts,
    train_rl,
)
from recsys_rl.train import train

N_ITEMS = 12  # vocabulary incl. the 2 special tokens


def make_reward(**cfg) -> Reward:
    return Reward(
        RLConfig(**cfg),
        item_length=torch.arange(N_ITEMS, dtype=torch.float) * 60,  # item i lasts i minutes
        item_length_bin=torch.arange(N_ITEMS) % 3,
        length_bin_ratio=torch.tensor([0.5, 0.6, 0.7]),
        item_ratio=torch.full((N_ITEMS,), 0.4),
    )


def test_rewards():
    probs = torch.zeros(2, 1, 2, len(HEADS))  # [E, B, G, H]
    probs[..., HEAD_INDEX["ratio"]] = torch.tensor([[[0.8, 0.4]], [[0.6, 0.4]]])
    probs[..., HEAD_INDEX["like"]] = 0.1
    probs[..., HEAD_INDEX["dislike"]] = 0.2
    probs[..., HEAD_INDEX["continue"]] = 0.9
    cand, repeat = torch.tensor([[4, 5]]), torch.tensor([[0, 3]])

    def r(**cfg):
        return make_reward(**cfg)(probs, cand, repeat)[0].tolist()

    assert r() == pytest.approx([0.7, 0.4])
    assert r(reward="listen_time") == pytest.approx([0.7 * 4, 0.4 * 5])
    assert r(reward="like_ratio") == pytest.approx([0.17, 0.14])
    assert r(reward="debiased_ratio") == pytest.approx([0.7 - 0.6, 0.4 - 0.7])  # bins 1, 2
    assert r(reward="item_debiased_ratio") == pytest.approx([0.3, 0.0])
    assert r(reward="continue") == pytest.approx([0.9, 0.9])
    assert r(repeat_penalty=0.5, dislike_penalty=1.0) == pytest.approx([0.5, -0.3])
    std = probs[..., 0, 0].std().item()
    assert r(uncertainty_penalty=1.0) == pytest.approx([0.7 - std, 0.4])
    assert len(REWARDS) == 6
    with pytest.raises(ValueError):
        make_reward(reward="clicks")


def test_reward_statistics():
    items = pl.DataFrame({"idx": [2, 3], "count": [90, 0], "mean_ratio": [0.2, None]})
    out = item_mean_ratio(items, global_mean=0.6, prior=10)
    assert out.tolist() == pytest.approx([0.6, 0.6, (90 * 0.2 + 10 * 0.6) / 100, 0.6])

    class Store:
        columns = {
            "timestamp": np.array([0, 1, 2, 3, 100]),
            "played_ratio_pct": np.array([100, 150, 20, 60, 0]),
            "length_bin": np.array([0, 0, 2, 2, 1]),
        }

    by_bin = ratio_by_length_bin(Store(), train_end=50, n_bins=3)  # the last listen is val
    assert by_bin.tolist() == pytest.approx([1.0, 0.7, 0.4])  # empty bin -> global mean


def test_advantages():
    r = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    rloo = advantages(r, "rloo")
    assert rloo[0].tolist() == pytest.approx([1 - 2.5, 2 - 2.0, 3 - 1.5])
    assert rloo[1].abs().max() == 0
    grpo = advantages(r, "grpo")
    assert grpo[0].tolist() == pytest.approx([-1.0, 0.0, 1.0], abs=1e-5)
    with pytest.raises(ValueError):
        advantages(r[:, :1], "rloo")
    with pytest.raises(ValueError):
        advantages(r, "ppo")


def test_sample_contexts():
    class Store:
        columns = {"is_organic": np.array([1, 1, 0, 1, 1, 0, 1, 1, 1, 1])}

    starts, lengths = np.array([0, 6]), np.array([6, 4])  # window 2 has no recommendation
    ctx_starts, ctx_lengths = sample_contexts(
        Store(), starts, lengths, 200, np.random.default_rng(0)
    )
    first = ctx_starts == 0
    assert set(ctx_lengths[first]) == {2, 5}  # cut right before the recommendations
    assert set(ctx_lengths[~first]) == {1, 2, 3}  # any listen after the first one
    assert (ctx_lengths >= 1).all()


def tiny_models(seed: int = 0):
    torch.manual_seed(seed)
    policy = GPT2Recommender(MODEL_CFG, N_ITEMS, max_seq_len=8).eval()
    ref = copy.deepcopy(policy).requires_grad_(False)
    rm = RewardModel(
        RewardModelConfig(n_ensemble=2, head_hidden=8),
        MODEL_CFG,
        N_ITEMS,
        max_seq_len=8,
        item_length_bin=torch.zeros(N_ITEMS, dtype=torch.long),
    ).eval()
    return policy, ref, rm.requires_grad_(False)


def tiny_batch():
    item_idx = torch.tensor([[2, 3, 4, PAD_IDX], [5, 5, 6, 7], [8, PAD_IDX, PAD_IDX, PAD_IDX]])
    mask = item_idx != PAD_IDX
    zeros = torch.zeros_like(item_idx)
    return {
        "item_idx": item_idx,
        "mask": mask,
        **{c: zeros for c in ("ratio_bin", "length_bin", "is_organic", "liked", "disliked")},
    }


@torch.no_grad()
def expected_reward(policy, rm, reward, batch):
    """Exact E_{a ~ pi}[r(s, a)] over the whole vocabulary."""
    p = torch.softmax(policy.next_item_logits(batch)[:, N_SPECIAL_TOKENS:], -1)
    cand = torch.arange(N_SPECIAL_TOKENS, N_ITEMS)[None].expand(len(p), -1)
    repeat = torch.zeros_like(cand)
    return (p * reward(rm.score(batch, cand), cand, repeat)).sum(-1).mean().item()


@pytest.mark.parametrize("method", ["rloo", "grpo", "bon", "dpo"])
def test_policy_loss_improves_reward_and_kl(method):
    policy, ref, rm = tiny_models()
    reward = make_reward()
    batch = tiny_batch()
    next_item = torch.tensor([9, 1, 10])  # the second one is OOV: no SFT target
    # beta is the KL weight, except for dpo where it scales the preference margin
    cfg = RLConfig(method=method, group_size=8, kl_beta=1.0 if method == "dpo" else 0.0)
    cfg.sft_alpha = 0.1
    gen = torch.Generator().manual_seed(0)

    loss, stats = policy_loss(policy, ref, rm, reward, batch, next_item, cfg, gen)
    assert stats["kl"] == pytest.approx(0.0, abs=1e-6)  # policy == ref at the start
    assert {"reward", "objective", "entropy", "sft_loss", "repeat_share"} <= set(stats)
    if method == "dpo":
        assert stats["pair_share"] == 1.0 and stats["dpo_margin"] == pytest.approx(0, abs=1e-6)
    loss.backward()
    assert all(p.grad is None for p in ref.parameters())
    assert all(p.grad is None for p in rm.parameters())

    before = expected_reward(policy, rm, reward, batch)
    opt = torch.optim.SGD(policy.parameters(), lr=1.0)
    for _ in range(30):
        loss, stats = policy_loss(policy, ref, rm, reward, batch, next_item, cfg, gen)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert expected_reward(policy, rm, reward, batch) > before
    assert stats["kl"] > 0
    with pytest.raises(ValueError):
        policy_loss(policy, ref, rm, reward, batch, next_item, RLConfig(method="ppo"), gen)


def test_top_k_sampling():
    policy, ref, rm = tiny_models()
    batch = tiny_batch()
    cfg = RLConfig(group_size=64, top_k=2, temperature=0.5)
    logits = policy.next_item_logits(batch)[:, N_SPECIAL_TOKENS:]
    top = logits.topk(2, dim=-1).indices + N_SPECIAL_TOKENS
    drawn = []

    def spy(b, candidates):
        drawn.append(candidates)
        return RewardModel.score(rm, b, candidates)

    rm.score = spy
    gen = torch.Generator().manual_seed(0)
    policy_loss(policy, ref, rm, make_reward(), batch, torch.tensor([9, 9, 9]), cfg, gen)
    assert (drawn[0][:, :, None] == top[:, None, :]).any(-1).all()


def test_dpo_loss():
    logp = torch.log(torch.tensor([[0.5, 0.3, 0.2], [0.2, 0.2, 0.6]]))
    ref_logp = torch.log(torch.full((2, 3), 1 / 3))
    actions = torch.tensor([[0, 1, 2], [1, 1, 1]])
    rewards = torch.tensor([[0.1, 0.9, 0.5], [0.4, 0.4, 0.4]])  # row 2: no preference
    loss, stats = dpo_loss(logp, ref_logp, actions, rewards, beta=2.0)
    margin = np.log(0.3) - np.log(0.5)  # chosen = item 1, rejected = item 0; ref terms cancel
    assert loss.item() == pytest.approx(-np.log(1 / (1 + np.exp(-2 * margin))), rel=1e-5)
    assert stats == pytest.approx({"pair_share": 0.5, "dpo_margin": margin})
    loss, stats = dpo_loss(logp, ref_logp, actions, torch.zeros(2, 3), beta=2.0)
    assert loss.item() == 0 and stats["pair_share"] == 0


def test_policy_eval_and_behaviour():
    policy, ref, rm = tiny_models()

    class Set:
        n_users = 2
        cutoff = 10**9
        context_starts = np.array([0, 3])
        context_ends = np.array([3, 5])

    class Store:
        n = 5
        columns = {
            "item_idx": np.array([2, 3, 4, 5, 5]),
            "timestamp": np.arange(5),
            "next_gap": np.full(5, -1),
            "like_delay": np.full(5, -1),
            "dislike_delay": np.full(5, -1),
            **{c: np.zeros(5, dtype=np.int64) for c in ("ratio_bin", "length_bin", "is_organic")},
        }

        def gather(self, starts, lengths, seq_len, cutoff):
            from recsys_rl.data import EventStore

            return EventStore.gather(self, starts, lengths, seq_len, cutoff)

    recs, kl, ent = policy_eval(policy, ref, Store(), Set(), k=4, batch_size=1)
    assert recs.shape == (2, 4) and (recs >= N_SPECIAL_TOKENS).all()
    assert np.allclose(kl, 0, atol=1e-6) and (ent > 0).all()
    scores = score_recommendations(rm, make_reward(), Store(), Set(), recs, batch_size=1)
    assert scores["reward"].shape == (2, 4)
    assert np.allclose(scores["reward"], scores["ratio"])  # R1 without penalties

    histories = [np.array([2, 3]), np.array([9])]
    length = np.arange(N_ITEMS, dtype=float)
    artist = np.array([-1, -1, 7, 7, 7, 8, 8, 8, 8, 8, -1, -1])
    out = behaviour_metrics(
        np.array([[2, 4], [9, 10]]), histories, length, length / 10, N_ITEMS, artist, k=2
    )
    assert out["artists@2"] == pytest.approx(1.5)  # one artist, then artist 8 + unknown
    assert out["repeat@2"] == pytest.approx(0.5)
    assert out["length@2"] == pytest.approx((2 + 4 + 9 + 10) / 4)
    assert out["log_count@2"] == pytest.approx(0.625)


def test_train_rl_resume(tmp_path):
    write_events(tmp_path / "raw" / "flat" / "50m")
    base_cfg = make_cfg(tmp_path, run_name="base")
    prepared = prepare(base_cfg)
    device = torch.device("cpu")
    train(base_cfg, prepared, device)
    for name, seed in (("rm_proxy", 0), ("rm_gold", 7)):
        rm_cfg = make_cfg(tmp_path, run_name=name, base_run="base", max_steps=2, eval_every=2)
        rm_cfg.seed = seed
        train_reward_model(rm_cfg, prepared, device)
    item_ids = np.arange(100, 200)  # the items of write_events
    pl.DataFrame({"item_id": item_ids, "artist_id": item_ids % 7}).write_parquet(
        tmp_path / "raw" / "artist_item_mapping.parquet"
    )

    cfg = make_cfg(tmp_path, run_name="rl")
    cfg.rl = RLConfig(
        batch_size=4,
        group_size=4,
        steps=4,
        eval_every=2,
        warmup_steps=1,
        amp=False,
        lr=1e-2,
        sft_alpha=0.1,
        keep_checkpoints=True,
    )
    path = train_rl(cfg, prepared, device)
    log = [json.loads(line) for line in (cfg.run_dir / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == [0, 2, 4]
    assert log[0]["val_kl_ref"] == pytest.approx(0.0, abs=1e-6)  # step 0 is pi_ref
    assert log[2]["val_kl_ref"] > 0
    expected = {
        "val_ndcg@10", "val_recall@100", "val_proxy_reward@10", "val_gold_reward@10",
        "val_repeat@10", "val_length@10", "val_log_count@10", "val_gini@10", "val_entropy",
        "val_artists@10",
        "train_reward", "train_kl", "train_sft_loss", "train_grad_norm",
    }  # fmt: skip
    assert expected <= set(log[1])
    assert all(np.isfinite(v) for r in log for v in r.values())
    assert (cfg.run_dir / "step_2.pt").exists()

    policy = load_model(path)
    base = load_model(tmp_path / "runs" / "base" / "best.pt")
    assert any(
        not torch.equal(a, b) for a, b in zip(policy.parameters(), base.parameters(), strict=True)
    )
    cfg.rl.steps = 6
    train_rl(cfg, prepared, device)
    log = (cfg.run_dir / "log.jsonl").read_text().splitlines()
    assert [json.loads(r)["step"] for r in log] == [0, 2, 4, 6]

    cfg.run_name, cfg.rl.method, cfg.rl.steps = "rl_dpo", "dpo", 2
    train_rl(cfg, prepared, device)
    log = [json.loads(line) for line in (cfg.run_dir / "log.jsonl").read_text().splitlines()]
    assert {"train_pair_share", "train_dpo_margin", "train_objective"} <= set(log[-1])


def test_distinct_categories_and_item_artists(tmp_path):
    cats = np.array([0, 0, 1, -1, -1])
    recs = np.array([[0, 1, 2], [3, 4, 0], [1, 0, 3]])
    # [0, 0, 1] -> 2; two unknown items are distinct -> 3; [0, 0, unknown] -> 2
    assert metrics.distinct_categories(recs, cats, 3) == pytest.approx(7 / 3)
    assert metrics.distinct_categories(recs, cats, 2) == pytest.approx((1 + 2 + 1) / 3)
    assert evaluate.item_artists(tmp_path, pl.DataFrame({"idx": [2], "item_id": [5]})) is None
    pl.DataFrame({"item_id": [5, 5, 6], "artist_id": [10, 11, 12]}).write_parquet(
        tmp_path / "artist_item_mapping.parquet"
    )
    items = pl.DataFrame({"idx": [2, 3, 4], "item_id": [5, 6, 7]})
    assert evaluate.item_artists(tmp_path, items).tolist() == [-1, -1, 10, 12, -1]
