import itertools
import math

import numpy as np
import pytest
import torch
from test_reward_model import make_cfg
from test_rl import N_ITEMS, make_reward, tiny_batch, tiny_models, train_base_and_rms

from recsys_rl.config import RLConfig
from recsys_rl.data import N_SPECIAL_TOKENS
from recsys_rl.rerank import greedy_slate, rerank_eval, slate_eval, tilted_scores
from recsys_rl.rl import (
    distinct_count,
    plackett_luce_log_prob,
    policy_loss,
    sample_slates,
    train_rl,
)
from recsys_rl.significance import collect_per_user

ARTIST = torch.tensor([-1, -1, 0, 0, 0, 1, 1, 2, -1, -1, 3, 3])  # by vocabulary index


def test_greedy_slate():
    scores = torch.tensor([[5.0, 4.9, 4.8, 1.0, 0.5, 0.4]])
    categories = torch.tensor([[0, 0, 0, 1, 2, 2]])
    assert greedy_slate(scores, categories, 0.0, 3, 6).tolist() == [[0, 1, 2, 3, 4, 5]]
    # bonus 4: a new category (1 + 4) beats item 1 (4.9), but item 4 (0.5 + 4) does not
    assert greedy_slate(scores, categories, 4.0, 3, 6).tolist() == [[0, 3, 1, 2, 4, 5]]
    assert greedy_slate(scores, categories, 5.0, 3, 6).tolist() == [[0, 3, 4, 1, 2, 5]]
    # the slate is only the first 2 positions; the rest by score
    assert greedy_slate(scores, categories, 5.0, 2, 4).tolist() == [[0, 3, 1, 2]]
    # brute force: every greedy step maximises score + bonus * new category
    torch.manual_seed(0)
    scores, categories = torch.randn(20, 15), torch.randint(0, 4, (20, 15))
    top = greedy_slate(scores, categories, 0.7, 5, 8)
    for row in range(20):
        chosen = []
        for _ in range(5):
            gain = [
                -math.inf if j in chosen
                else scores[row, j].item()
                + 0.7 * (categories[row, j].item() not in categories[row, chosen].tolist())
                for j in range(15)
            ]  # fmt: skip
            chosen.append(int(np.argmax(gain)))
        assert top[row, :5].tolist() == chosen
        assert len(set(top[row].tolist())) == 8


def test_tilted_scores_rank_like_tilt():
    torch.manual_seed(0)
    logp = torch.log_softmax(torch.randn(4, 12, dtype=torch.float64), -1).sort(1, True).values
    reward = torch.rand(4, 5, dtype=torch.float64)
    assert torch.equal(tilted_scores(logp, reward, 0.0), reward)
    assert torch.equal(tilted_scores(logp, reward, math.inf), logp)
    scores = tilted_scores(logp, reward, 0.1)
    assert scores.shape == logp.shape and torch.equal(scores[:, 5:], logp[:, 5:])


def test_plackett_luce_log_prob():
    logp = torch.log_softmax(torch.randn(2, 5, dtype=torch.float64), -1)
    lists = torch.tensor(list(itertools.permutations(range(5), 3)))  # every ordered 3-list
    slates = lists[None].expand(2, -1, -1)
    total = plackett_luce_log_prob(logp, slates).exp().sum(1)
    assert torch.allclose(total, torch.ones(2, dtype=torch.float64))
    one = plackett_luce_log_prob(logp, slates[:, :, :1])  # a 1-list is one sample
    assert torch.allclose(one, logp.gather(1, lists[None, :, 0].expand(2, -1)))


def test_sample_slates_and_distinct_count():
    gen = torch.Generator().manual_seed(0)
    logp = torch.log_softmax(torch.randn(3, 20), -1)
    slates = sample_slates(logp, group_size=4, slate_size=6, generator=gen)
    assert slates.shape == (3, 4, 6)
    assert all(len(set(s.tolist())) == 6 for s in slates.flatten(0, 1))  # no repeats in a list
    with pytest.raises(ValueError):
        sample_slates(logp.masked_fill(logp < logp.topk(5).values[:, -1:], -math.inf), 2, 6)
    assert distinct_count(torch.tensor([[3, 1, 3, -2], [0, 0, 0, 0]])).tolist() == [3, 1]


@torch.no_grad()
def expected_artists(policy, batch, slate_size: int = 3, n: int = 400) -> float:
    gen = torch.Generator().manual_seed(1)
    logp = torch.log_softmax(policy.next_item_logits(batch)[:, N_SPECIAL_TOKENS:], -1)
    slates = sample_slates(logp, n, slate_size, gen) + N_SPECIAL_TOKENS
    cat = torch.where(ARTIST[slates] >= 0, ARTIST[slates], -1 - slates)
    return distinct_count(cat).float().mean().item()


def test_slate_policy_loss():
    assert len(ARTIST) == N_ITEMS
    policy, ref, rm = tiny_models()
    batch, next_item = tiny_batch(), torch.tensor([9, 1, 10])
    reward = make_reward(item_artist=ARTIST)
    cfg = RLConfig(method="slate", group_size=8, slate_size=3, kl_beta=0.0, slate_diversity=1.0)
    gen = torch.Generator().manual_seed(0)
    loss, stats = policy_loss(policy, ref, rm, reward, batch, next_item, cfg, gen)
    assert stats["kl"] == pytest.approx(0.0, abs=1e-6)
    assert {"reward", "item_reward", "slate_artists", "reward_std"} <= set(stats)
    assert stats["reward"] == pytest.approx(stats["item_reward"] + stats["slate_artists"] / 3)
    loss.backward()
    assert all(p.grad is None for p in ref.parameters())

    before = expected_artists(policy, batch)
    opt = torch.optim.SGD(policy.parameters(), lr=1.0)
    for _ in range(40):  # the diversity term dominates the small RM differences
        loss, stats = policy_loss(policy, ref, rm, reward, batch, next_item, cfg, gen)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert expected_artists(policy, batch) > before
    with pytest.raises(ValueError):  # the per-item artist penalty would count artists twice
        penalised = make_reward(item_artist=ARTIST, artist_penalty=1.0)
        policy_loss(policy, ref, rm, penalised, batch, next_item, cfg, gen)


def test_slate_eval_and_training(tmp_path):
    prepared, device = train_base_and_rms(tmp_path)
    cfg = make_cfg(tmp_path, run_name="slate")
    cfg.rl = RLConfig(amp=False)
    points = ((0.1, 0.0), (0.1, 1.0), (0.0, 0.0), (0.0, 1.0))
    table = slate_eval(cfg, prepared, device, points=points)  # all 100 items scored
    assert table["step"].to_list() == [0, 1, 2, 3]
    assert table["diversity"].to_list() == [0.0, 1.0, 0.0, 1.0]
    plain = rerank_eval(cfg, prepared, device, betas=(0.1,))
    for column in ("val_ndcg@10", "val_artists@10", "val_kl_ref", "val_proxy_reward@10"):
        assert table[column][0] == pytest.approx(plain[column][0])  # diversity 0 = reranker
    artists = table["val_artists@10"].to_list()
    assert artists[1] >= artists[0] and artists[3] >= artists[2]
    assert table["val_kl_ref"][1] == pytest.approx(table["val_kl_ref"][0])  # first position
    assert np.isnan(table["val_kl_ref"][3])
    with pytest.raises(ValueError):
        slate_eval(cfg, prepared, device, points=((math.inf, 1.0),))

    per_user = collect_per_user(
        cfg, prepared, device, rerank={"rr": ("ratio", 0.1)},
        slates={"sl0": ("ratio", 0.1, 0.0), "sl1": ("ratio", 0.1, 1.0)},
        k=20, exclude_seen=False,
    )  # fmt: skip
    assert {"rr", "sl0", "sl1"} <= set(per_user["method"])
    rr, sl0 = per_user.filter(method="rr"), per_user.filter(method="sl0")
    assert rr["ndcg@10"].to_list() == pytest.approx(sl0["ndcg@10"].to_list())
    assert per_user.filter(method="sl1")["artists@10"].mean() >= rr["artists@10"].mean()

    cfg = make_cfg(tmp_path, run_name="rl_slate")
    cfg.rl = RLConfig(
        method="slate", slate_size=4, slate_diversity=0.05, group_size=4, batch_size=4,
        steps=2, eval_every=2, warmup_steps=1, amp=False, lr=1e-2,
    )  # fmt: skip
    train_rl(cfg, prepared, device)
    log = (cfg.run_dir / "log.jsonl").read_text().splitlines()
    assert '"train_slate_artists"' in log[-1] and '"val_artists@10"' in log[-1]
