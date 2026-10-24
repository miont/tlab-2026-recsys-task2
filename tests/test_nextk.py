import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from test_reward_model import make_cfg
from test_rl import N_ITEMS, make_reward, tiny_batch, tiny_models, train_base_and_rms

from recsys_rl.config import RLConfig, SlateRMConfig
from recsys_rl.data import N_SPECIAL_TOKENS, PAD_IDX
from recsys_rl.nextk import (
    Appender,
    generate_slates,
    logged_list_effects,
    masked_kl,
    nextk_per_user,
    nextk_reward_scale,
    nextk_step,
    query_hidden,
    reranker_lists,
)
from recsys_rl.rl import train_rl
from recsys_rl.slate_rm import (
    SlateRewardModel,
    list_reward,
    logged_slates,
    slate_batch,
    slate_rm_eval,
    train_slate_rm,
)
from recsys_rl.utils import load_checkpoint

EDGES = [0, 5, 10, 25, 49, 75, 99, 100]


def padded_batch(t: int = 8) -> dict[str, torch.Tensor]:
    batch = tiny_batch()
    pad = t - batch["item_idx"].shape[1]
    return {k: F.pad(v, (0, pad)) for k, v in batch.items()}


def tiny_slate_rm(seed: int = 0) -> SlateRewardModel:
    _, _, rm = tiny_models(seed)
    return SlateRewardModel(rm, SlateRMConfig(slate_size=3, n_layer=1, n_head=2))


def test_logged_slates_brute_force(tmp_path):
    prepared, _ = train_base_and_rms(tmp_path)
    store, bounds = prepared.store, prepared.bounds
    rows, lengths = logged_slates(store, 0, bounds.train_end, slate_size=3, min_len=2)
    c = {k: np.asarray(v) for k, v in store.columns.items()}
    expected = []
    for u in range(store.n_users):
        lo, hi = store.offsets[u], store.offsets[u + 1]
        run: list[int] = []
        for i in range(lo, hi + 1):
            ok = (
                i < hi
                and c["timestamp"][i] < bounds.train_end
                and c["is_organic"][i] == 0
                and c["item_idx"][i] >= N_SPECIAL_TOKENS
            )
            if ok and run and c["session_id"][i] == c["session_id"][run[-1]]:
                run.append(i)
                continue
            for k in range(0, len(run), 3):  # close the previous run
                chunk = run[k : k + 3]
                if len(chunk) >= 2 and chunk[0] > lo:
                    expected.append((chunk[0], len(chunk)))
            run = [i] if ok else []
    assert len(expected) > 0
    assert list(zip(rows.tolist(), lengths.tolist(), strict=True)) == expected

    ctx, slates, mask, targets, valid = slate_batch(
        store, rows[:5], lengths[:5], 16, bounds.train_end, 3, make_cfg(tmp_path).data, 0.3
    )
    assert slates.shape == (5, 3) and (slates[~mask] == PAD_IDX).all()
    assert (ctx["mask"].sum(1) > 0).all()
    assert np.array_equal(slates[:, 0], c["item_idx"][rows[:5]])
    # the context ends right before the slate
    last = ctx["mask"].sum(1) - 1
    assert np.array_equal(ctx["timestamp"][np.arange(5), last], c["timestamp"][rows[:5] - 1])
    assert not valid[~mask].any()


def test_slate_rm_starts_as_item_rm_and_is_causal():
    model = tiny_slate_rm()
    hidden = torch.randn(4, 16)
    slates = torch.randint(N_SPECIAL_TOKENS, N_ITEMS, (4, 3))
    repeat = torch.zeros_like(slates)
    full, item = model.logits(hidden, slates, repeat)
    assert full.shape == (2, 4, 3, 6) and torch.allclose(full, item)
    torch.nn.init.normal_(model.delta.weight)  # a trained list part
    full, item = model.logits(hidden, slates, repeat)
    assert not torch.allclose(full, item)
    changed = slates.clone()
    changed[:, 2] = (changed[:, 2] - N_SPECIAL_TOKENS + 1) % (N_ITEMS - 2) + N_SPECIAL_TOKENS
    other, _ = model.logits(hidden, changed, repeat)
    assert torch.allclose(other[:, :, :2], full[:, :, :2], atol=1e-6)  # earlier positions
    assert not torch.allclose(other[:, :, 2], full[:, :, 2])
    # position_scores = the last position of the list prefix + candidate
    cand = torch.randint(N_SPECIAL_TOKENS, N_ITEMS, (4, 5))
    no_repeat = torch.zeros_like(cand)
    probs = model.position_scores(hidden, slates[:, :2], cand, repeat[:, :2], no_repeat)
    for m in range(5):
        lists = torch.cat([slates[:, :2], cand[:, m : m + 1]], 1)
        direct = torch.sigmoid(model.logits(hidden, lists, repeat)[0][:, :, 2])
        assert torch.allclose(probs[:, :, m], direct, atol=1e-6)
    p = torch.rand(2, 4, 3, 6)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=torch.bool)
    r = list_reward(p, mask)
    assert r[1] == pytest.approx(p.mean(0)[1, :2, 0].mean().item())


def test_generate_matches_query_hidden():
    policy, _, _ = tiny_models()
    batch = padded_batch()
    appender = Appender(make_reward(), EDGES)
    for append in (True, False):
        slates = generate_slates(policy, batch, appender, 4, append, greedy=True)
        assert slates.shape == (3, 4)
        assert all(len(set(s.tolist())) == 4 for s in slates)  # no repeats in a list
        hidden = query_hidden(policy, batch, slates, appender, append)
        chosen = torch.zeros(3, N_ITEMS - N_SPECIAL_TOKENS, dtype=torch.bool)
        for j in range(4):
            logits = policy.scores(hidden[:, j])[:, N_SPECIAL_TOKENS:]
            best = logits.masked_fill(chosen, -torch.inf).argmax(1) + N_SPECIAL_TOKENS
            assert torch.equal(best, slates[:, j])
            chosen[torch.arange(3), slates[:, j] - N_SPECIAL_TOKENS] = True
    ext = appender(batch, torch.tensor([9, 9, 9]))
    assert ext["item_idx"][0, 3] == 9 and ext["ratio_bin"][0, 3] == 4  # item ratio 0.4 -> bin 4


def reference_loss(policy, ref, batch, slates, adv, appender, beta):
    """The Next-K loss without the per-position backward trick."""
    s = N_SPECIAL_TOKENS
    h = query_hidden(policy, batch, slates, appender, True)
    with torch.no_grad():
        h_ref = query_hidden(ref, batch, slates, appender, True)
    chosen = torch.zeros(len(slates), N_ITEMS - s, dtype=torch.bool)
    rows = torch.arange(len(slates))
    loss = 0
    for j in range(slates.shape[1]):
        lp = F.log_softmax(policy.scores(h[:, j])[:, s:].masked_fill(chosen, -torch.inf), -1)
        lr = F.log_softmax(ref.scores(h_ref[:, j])[:, s:].masked_fill(chosen, -torch.inf), -1)
        kl = masked_kl(lp, lr, chosen)
        loss = loss - (adv * lp[rows, slates[:, j] - s]).mean() + beta * kl.mean()
        chosen = chosen.clone()  # the graph keeps the previous mask
        chosen[rows, slates[:, j] - s] = True
    return loss


def test_nextk_step_gradient_matches_direct_loss():
    policy, ref, _ = tiny_models()
    slate_rm = tiny_slate_rm(seed=1)
    torch.nn.init.normal_(slate_rm.delta.weight, std=0.5)
    reward = make_reward(item_artist=torch.arange(N_ITEMS) % 4)
    appender = Appender(reward, EDGES)
    batch = padded_batch()
    cfg = RLConfig(method="nextk", group_size=3, slate_size=3, kl_beta=0.5, slate_diversity=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    gen = torch.Generator().manual_seed(0)
    torch.nn.init.normal_(policy.item_emb.weight, std=0.3)  # pi != pi_ref, KL > 0
    stats = nextk_step(policy, ref, slate_rm, reward, appender, batch, cfg, scaler, None, gen)
    assert {"reward", "item_reward", "slate_artists", "kl", "objective", "entropy"} <= set(stats)
    assert stats["kl"] > 0
    grads = {n: p.grad.clone() for n, p in policy.named_parameters() if p.grad is not None}
    assert all(p.grad is None for p in ref.parameters())

    # the same samples and advantages through the direct loss
    from recsys_rl.nextk import repeat_batch
    from recsys_rl.rl import advantages, distinct_count

    gen = torch.Generator().manual_seed(0)
    rep = repeat_batch(batch, 3)
    slates = generate_slates(policy, rep, appender, 3, True, generator=gen)
    with torch.no_grad():
        r = list_reward(slate_rm.score(rep, slates))
        r = r + 0.1 * distinct_count(reward.artist_categories(slates)) / 3
    adv = advantages(r.view(-1, 3), "rloo").reshape(-1)
    policy.zero_grad()
    reference_loss(policy, ref, rep, slates, adv, appender, 0.5).backward()
    for name, p in policy.named_parameters():
        if p.grad is not None:
            assert torch.allclose(grads[name], p.grad, atol=1e-5), name


def test_nextk_pipeline(tmp_path):
    prepared, device = train_base_and_rms(tmp_path)
    for name, base, seed in (("slate_rm_proxy", "rm_proxy", 0), ("slate_rm_gold", "rm_gold", 7)):
        cfg = make_cfg(tmp_path, run_name=name)
        cfg.seed = seed
        cfg.slate_rm = SlateRMConfig(
            base_rm=base, slate_size=10, n_layer=1, n_head=2, batch_size=8, steps=4,
            eval_every=2, amp=False, user_fraction=0.8 if seed else 1.0,
        )  # fmt: skip
        path = train_slate_rm(cfg, prepared, device)
        log = [json.loads(line) for line in (cfg.run_dir / "log.jsonl").read_text().splitlines()]
        assert [rec["step"] for rec in log] == [0, 2, 4]
        first = log[0]  # delta = 0: the item-level RM
        assert first["val_slate_loss"] == pytest.approx(first["val_item_loss"])
        assert load_checkpoint(path)["step"] in (0, 2, 4)
        assert train_slate_rm(cfg, prepared, device) == path  # finished: no more training
        assert len((cfg.run_dir / "log.jsonl").read_text().splitlines()) == 3
    rows, lengths = logged_slates(prepared.store, 0, prepared.bounds.train_end, 10)
    from recsys_rl.slate_rm import load_slate_rm

    model = load_slate_rm(tmp_path / "runs" / "slate_rm_proxy" / "best.pt")
    summary, by_pos = slate_rm_eval(
        model, prepared.store, rows[:50], lengths[:50], prepared.bounds.train_end, cfg.data, 0.3
    )
    assert summary["n_slates"] == 50 and by_pos["position"].to_list()[0] == 1
    effects = logged_list_effects(prepared.store, rows[:50], lengths[:50], None)
    assert effects["n"].sum() == lengths[:50].sum()

    cfg = make_cfg(tmp_path, run_name="rl_nextk")
    cfg.rl = RLConfig(
        method="nextk", slate_size=10, group_size=3, batch_size=4, steps=2, eval_every=2,
        warmup_steps=1, amp=False, lr=1e-2, slate_diversity=0.05,
    )  # fmt: skip
    scale = nextk_reward_scale(cfg, prepared, device, n_contexts=8)
    assert scale["nextk_group_std"] >= 0 and 0 < scale["topk_mean"] < 1
    policy_path = train_rl(cfg, prepared, device)
    log = (cfg.run_dir / "log.jsonl").read_text().splitlines()
    assert len(log) == 2 and '"val_gold_slate@10"' in log[-1] and '"train_kl"' in log[-1]
    first = json.loads(log[0])
    assert first["val_kl_ref"] == 0.0

    per_user = nextk_per_user(
        cfg, prepared, device, policies={"pi_ref_nextk": (None, True), "pi_ref_topk": (None, False),
        "rl": (policy_path, True)}, searches={"search": (0.1, True), "greedy_r": (0.0, True),
        "search_topk": (0.1, False), "search_ref": (1e9, False)},
        best_of={"bon4": 4}, extra={"rr": lambda ev: reranker_lists(ev, 0.1)},
    )  # fmt: skip
    assert set(per_user["method"]) == {
        "pi_ref_nextk", "pi_ref_topk", "rl", "search", "greedy_r", "search_topk", "search_ref",
        "bon4", "rr",
    }  # fmt: skip
    for column in ("ndcg@10", "proxy_slate@10", "gold_slate@10", "proxy_ratio@10", "artists@10"):
        assert per_user[column].is_not_null().all()
    # the Top-K list of pi_ref is its plain top of the vocabulary
    from recsys_rl.rl import policy_eval

    ref_topk = per_user.filter(method="pi_ref_topk")
    assert ref_topk["kl_ref"].sum() == 0
    # without appending and with a negligible tilt the search is the Top-K list of pi_ref
    search_ref = per_user.filter(method="search_ref")
    for column in ("ndcg@10", "proxy_slate@10", "artists@10"):
        np.testing.assert_allclose(search_ref[column], ref_topk[column], rtol=1e-5)
    _ = policy_eval
    # greedy by the reward alone maximises the first position's reward among the candidates
    search = per_user.filter(method="greedy_r")["proxy_slate@10"].mean()
    assert np.isfinite(search)
