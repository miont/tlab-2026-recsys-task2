"""RL fine-tuning of the base recommender against the proxy reward model (stage 4).

One-step setting (contextual bandit, plan §6.1, §6.3). The policy ``pi_theta`` starts from the
base model ``runs/<rl.base_run>/best.pt``; a frozen copy of it is ``pi_ref``. One step:

- contexts: ``batch_size`` train windows, each cut right before a random recommendation listen
  (``is_organic = 0``, any listen if the window has none), so the states are the ones where the
  service recommended something; like flags are leak-free with respect to ``train_end``;
- actions: ``group_size`` items per context sampled from ``softmax(logits / temperature)``,
  optionally restricted to the policy's ``top_k``;
- reward: ``rewards.Reward`` of the proxy RM (``runs/<rl.reward_run>/best.pt``) scoring the
  candidates as recommendations (``is_organic = 0``);
- objective (``rl.method``), all on the same ``G`` samples and rewards:

  - ``rloo`` / ``grpo``: policy gradient ``-mean(A * log pi_sample(a|s))`` with the RLOO
    advantage (reward minus the mean of the other samples of the group) or the GRPO one (reward
    minus the group mean, divided by the std of the group or, with ``grpo_std=batch``, by the
    std of these centred rewards over the whole batch as in REINFORCE++);
  - ``bon``: Best-of-N fine-tuning, cross-entropy on the sample with the highest reward
    (the training-time counterpart of reranking the policy's candidates by the RM);
  - ``dpo``: online DPO (Rafailov et al., 2023; pairs from RM-ranked policy samples as in
    OneRec): the best sample of the group is chosen, the worst is rejected,
    ``-log sigmoid(kl_beta * margin)`` of their log-ratios to ``pi_ref``; groups whose samples
    all get the same reward give no pair;

- loss: objective ``+ kl_beta * KL(pi_theta || pi_ref) + sft_alpha * CE``, where the KL is exact
  over the vocabulary (not added for ``dpo``, whose ``kl_beta`` is the DPO temperature that
  already keeps it close to ``pi_ref``) and CE is the next-item cross-entropy on the real listen
  after the context (an optional anchor to the data).

Dropout is off (the policy stays in eval mode) so that sampling, log-probs and the KL use the
same distribution.

Every ``eval_every`` steps (every ``eval_early_every`` steps up to ``eval_early_until``, and at
step 0, which is ``pi_ref`` itself) the policy is evaluated
on the val day with the protocol of ``evaluate``: relevance of the top-100, proxy and gold
reward of the top-10, behaviour of the top-10 (share of tracks from the history, mean length,
popularity, coverage, Gini, distinct artists) and the mean KL / entropy over the val contexts.
Records go to ``runs/<run_name>/log.jsonl``; ``last.pt`` holds the full training state
(sampling is seeded by the step, so a resumed run continues the same way) and ``policy.pt``
the final policy.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recsys_rl import evaluate, metrics
from recsys_rl.config import Config, RLConfig, save_config
from recsys_rl.data import N_SPECIAL_TOKENS, EventStore, PreparedData, prepare
from recsys_rl.evaluate import EvalSet
from recsys_rl.model import GPT2Recommender, load_model, save_model, to_tensors
from recsys_rl.reward_model import RewardModel, context_repeat_counts, load_reward_model
from recsys_rl.rewards import Reward, score_recommendations
from recsys_rl.train import amp_dtype, make_optimizer
from recsys_rl.utils import get_device, load_checkpoint, save_checkpoint, set_seed

METHODS = ("rloo", "grpo", "bon", "dpo")


def sample_contexts(
    store: EventStore,
    starts: np.ndarray,
    lengths: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """``(starts, lengths)`` of ``n`` contexts: random windows cut right before a random
    recommendation listen of the window (position >= 1); the listen at ``start + length`` is
    the real next item."""
    sel = rng.integers(0, len(starts), n)
    organic = store.columns["is_organic"]
    cuts = np.empty(n, dtype=np.int64)
    for i, w in enumerate(sel):
        s, length = starts[w], lengths[w]
        rec = np.flatnonzero(np.asarray(organic[s + 1 : s + length]) == 0) + 1
        cuts[i] = rng.choice(rec) if len(rec) else rng.integers(1, length)
    return starts[sel], cuts


def advantages(rewards: torch.Tensor, method: str, grpo_std: str = "group") -> torch.Tensor:
    """``[B, G]`` advantages of ``G`` samples per context (``G >= 2``).

    The GRPO divisor is the std of each group (``grpo_std="group"``) or one std of the centred
    rewards of the whole batch (``"batch"``): per-group stds make groups of near-equal rewards,
    where the differences are mostly RM noise, as loud as the informative ones."""
    g = rewards.shape[1]
    if g < 2:
        raise ValueError("group-baseline methods need group_size >= 2")
    if method == "rloo":
        return rewards - (rewards.sum(1, keepdim=True) - rewards) / (g - 1)
    if method == "grpo":
        centred = rewards - rewards.mean(1, keepdim=True)
        if grpo_std == "group":
            return centred / (rewards.std(1, keepdim=True) + 1e-6)
        if grpo_std == "batch":
            return centred / (centred.std() + 1e-6)
        raise ValueError(f"unknown grpo_std {grpo_std!r}, expected group or batch")
    raise ValueError(f"unknown advantage {method!r}, expected rloo or grpo")


def dpo_loss(
    logp: torch.Tensor,
    ref_logp: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """DPO loss on the best (chosen) vs worst (rejected) of the ``G`` sampled ``actions`` per
    context; ``logp`` / ``ref_logp`` are ``[B, n]`` log-probabilities over the actions.

    Contexts whose samples all get the same reward have no preference and are skipped."""
    rows = torch.arange(len(actions), device=actions.device)
    chosen = actions[rows, rewards.argmax(1)]
    rejected = actions[rows, rewards.argmin(1)]
    pairs = rewards.max(1).values > rewards.min(1).values
    margin = (logp[rows, chosen] - ref_logp[rows, chosen]) - (
        logp[rows, rejected] - ref_logp[rows, rejected]
    )
    stats = {"pair_share": pairs.float().mean().item()}
    if not pairs.any():
        return logp.sum() * 0, stats | {"dpo_margin": 0.0}
    stats["dpo_margin"] = margin[pairs].mean().item()
    return -F.logsigmoid(beta * margin[pairs]).mean(), stats


def policy_loss(
    policy: GPT2Recommender,
    ref: GPT2Recommender,
    rm: RewardModel,
    reward: Reward,
    batch: Mapping[str, torch.Tensor],
    next_item: torch.Tensor,
    cfg: RLConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Loss of one RL step on the contexts of ``batch`` and its statistics; ``next_item``
    (``[B]`` vocabulary indices) is the real listen after each context (for the SFT anchor)."""
    s = N_SPECIAL_TOKENS
    logits = policy.next_item_logits(batch)[:, s:].float()
    with torch.no_grad():
        ref_logp = F.log_softmax(ref.next_item_logits(batch)[:, s:].float(), -1)
    logp = F.log_softmax(logits, -1)

    sample_logits = logits / cfg.temperature
    if cfg.top_k:
        kth = sample_logits.topk(cfg.top_k, dim=-1).values[:, -1:]
        sample_logits = sample_logits.masked_fill(sample_logits < kth, float("-inf"))
    sample_logp = F.log_softmax(sample_logits, -1)
    actions = torch.multinomial(
        sample_logp.detach().exp(), cfg.group_size, replacement=True, generator=generator
    )
    candidates = actions + s
    with torch.no_grad():
        probs = rm.score(batch, candidates).float()
        repeat = context_repeat_counts(batch["item_idx"], batch["mask"], candidates)
        r = reward(probs, candidates, repeat)

    p = logp.exp()
    kl = (p * (logp - ref_logp)).sum(-1).mean()
    stats = {
        "reward": r.mean().item(),
        "reward_std": r.std(1).mean().item(),
        "kl": kl.item(),
        "entropy": -(p * logp).sum(-1).mean().item(),
        "repeat_share": (repeat > 0).float().mean().item(),
    }
    if cfg.method in ("rloo", "grpo"):
        adv = advantages(r, cfg.method, cfg.grpo_std)
        objective = -(adv * sample_logp.gather(1, actions)).mean()
    elif cfg.method == "bon":
        best = actions.gather(1, r.argmax(1, keepdim=True))
        objective = -logp.gather(1, best).mean()
    elif cfg.method == "dpo":
        objective, dpo_stats = dpo_loss(logp, ref_logp, actions, r, cfg.kl_beta)
        stats |= dpo_stats
    else:
        raise ValueError(f"unknown method {cfg.method!r}, expected one of {METHODS}")
    loss = objective if cfg.method == "dpo" else objective + cfg.kl_beta * kl
    stats["objective"] = objective.item()
    if cfg.sft_alpha:
        valid = next_item >= s
        ce = (
            F.cross_entropy(logits[valid], next_item[valid] - s)
            if valid.any()
            else logits.sum() * 0
        )
        loss = loss + cfg.sft_alpha * ce
        stats["sft_loss"] = ce.item()
    return loss, stats


@torch.no_grad()
def policy_eval(
    policy: GPT2Recommender,
    ref: GPT2Recommender,
    store: EventStore,
    eval_set: EvalSet,
    k: int = 100,
    batch_size: int = 128,
    amp_dtype: torch.dtype | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(recs, kl, entropy)``: top-``k`` vocabulary indices ``[n_users, k]`` of the policy
    after the frozen contexts of ``eval_set``, ``KL(pi || pi_ref)`` and the policy entropy
    per user."""
    device = next(policy.parameters()).device
    seq_len = policy.backbone.config.n_positions
    lengths = eval_set.context_ends - eval_set.context_starts
    recs, kls, ents = [], [], []
    for b in range(0, eval_set.n_users, batch_size):
        sl = slice(b, b + batch_size)
        batch = to_tensors(
            store.gather(eval_set.context_starts[sl], lengths[sl], seq_len, eval_set.cutoff),
            device,
        )
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = policy.next_item_logits(batch)[:, N_SPECIAL_TOKENS:].float()
            ref_logits = ref.next_item_logits(batch)[:, N_SPECIAL_TOKENS:].float()
        logp, ref_logp = F.log_softmax(logits, -1), F.log_softmax(ref_logits, -1)
        p = logp.exp()
        kls.append((p * (logp - ref_logp)).sum(-1).cpu().numpy())
        ents.append(-(p * logp).sum(-1).cpu().numpy())
        recs.append((logits.topk(k, dim=1).indices + N_SPECIAL_TOKENS).cpu().numpy())
    return np.concatenate(recs), np.concatenate(kls), np.concatenate(ents)


def behaviour_metrics(
    recs: np.ndarray,
    histories: list[np.ndarray],
    item_length: np.ndarray,
    item_log_count: np.ndarray,
    n_items: int,
    item_artist: np.ndarray | None = None,
    k: int = 10,
) -> dict[str, float]:
    """Composition of the top-``k`` (vocabulary indices): share of tracks from the user's
    history, mean track length (s) and log10 train count, Gini of exposure over ``n_items`` and,
    if ``item_artist`` is given, the mean number of distinct artists."""
    out = {
        f"repeat@{k}": metrics.repeat_rate(recs, histories, k),
        f"length@{k}": metrics.mean_item_value(recs, item_length, k),
        f"log_count@{k}": metrics.mean_item_value(recs, item_log_count, k),
        f"gini@{k}": metrics.gini(recs, k, n_items),
    }
    if item_artist is not None:
        out[f"artists@{k}"] = metrics.distinct_categories(recs, item_artist, k)
    return out


def load_frozen_rm(cfg: Config, run: str, device: torch.device) -> RewardModel:
    rm = load_reward_model(Path(cfg.data_root) / "runs" / run / "best.pt", device)
    return rm.eval().requires_grad_(False)


def train_rl(
    cfg: Config, prepared: PreparedData | None = None, device: torch.device | None = None
) -> Path:
    """Train (or resume) the RL policy of ``cfg.run_name``; returns ``policy.pt``. A run whose
    log already reaches ``rl.steps`` returns at once."""
    r = cfg.rl
    if r.method not in METHODS:
        raise ValueError(f"unknown method {r.method!r}, expected one of {METHODS}")
    run_dir = cfg.run_dir
    last_path, policy_path = run_dir / "last.pt", run_dir / "policy.pt"
    log_path = run_dir / "log.jsonl"
    if policy_path.exists() and log_path.exists():
        last = json.loads(log_path.read_text().splitlines()[-1])
        if last["step"] >= r.steps:  # finished: skip loading the models and the data
            print(f"{cfg.run_name} is done at step {last['step']}")
            return policy_path
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    set_seed(cfg.seed)
    device = device or get_device()
    prepared = prepared or prepare(cfg)
    store, items, bounds = prepared.store.in_memory(), prepared.items, prepared.bounds

    base_path = Path(cfg.data_root) / "runs" / r.base_run / "best.pt"
    policy = load_model(base_path, device).eval()
    ref = load_model(base_path, device).eval().requires_grad_(False)
    seq_len = policy.backbone.config.n_positions
    rm = load_frozen_rm(cfg, r.reward_run, device)
    rms = {"proxy": rm}
    if r.gold_run:
        rms["gold"] = load_frozen_rm(cfg, r.gold_run, device)
    for name, model in rms.items():
        if model.max_seq_len != seq_len or model.encoder.n_items != policy.n_items:
            raise ValueError(f"{name} RM does not match the base model (vocabulary / seq len)")
    reward = Reward.build(r, items, store, bounds.train_end, cfg.data).to(device)

    optimizer = make_optimizer(policy, r.lr, r.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(1, r.warmup_steps))
    )
    dtype = amp_dtype(device, r.amp)
    scaler = torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)
    generator = torch.Generator(device=device)

    state = {"step": 0}
    if last_path.exists():
        ckpt = load_checkpoint(last_path, map_location=device)
        policy.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        state = ckpt["state"]
        print(f"resumed from {last_path} at step {state['step']}")

    val = evaluate.build_eval_set(store, bounds, "val", seq_len, cfg.data.listen_plus_threshold)
    idx_to_item = evaluate.idx_to_item_array(items)
    val_histories = val.histories(store, "item_idx")
    item_length = np.full(policy.n_items, np.nan)
    item_length[items["idx"].to_numpy()] = items["track_length"].to_numpy()
    item_log_count = np.full(policy.n_items, np.nan)
    item_log_count[items["idx"].to_numpy()] = np.log10(items["count"].to_numpy())
    item_artist = evaluate.item_artists(cfg.raw_dir, items)  # None if not downloaded
    print(
        f"device {device}, amp {dtype}, {r.method} {r.reward}, beta {r.kl_beta}, "
        f"G {r.group_size}, {r.steps} steps, {val.n_users} val users"
    )

    def run_eval(train_stats: dict[str, float]) -> None:
        recs, kl, ent = policy_eval(policy, ref, store, val, amp_dtype=dtype)
        record = {"step": state["step"], "time": time.time()}
        record |= {f"train_{k}": v for k, v in train_stats.items()}
        val_metrics = evaluate.evaluate_recs(idx_to_item[recs], val, len(items))
        val_metrics |= {"kl_ref": float(kl.mean()), "entropy": float(ent.mean())}
        val_metrics |= behaviour_metrics(
            recs, val_histories, item_length, item_log_count, len(items), item_artist
        )
        for name, model in rms.items():
            scores = score_recommendations(model, reward, store, val, recs[:, :10], amp_dtype=dtype)
            val_metrics[f"{name}_reward@10"] = float(scores["reward"].mean())
            val_metrics[f"{name}_ratio@10"] = float(scores["ratio"].mean())
            val_metrics[f"{name}_ratio_std@10"] = float(scores["ratio_std"].mean())
        record |= {f"val_{k}": v for k, v in val_metrics.items()}
        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(
            f"step {state['step']} val ndcg@10 {val_metrics['ndcg@10']:.4f} "
            f"proxy {val_metrics['proxy_reward@10']:.4f}"
            + (f" gold {val_metrics['gold_reward@10']:.4f}" if "gold" in rms else "")
            + f" kl {val_metrics['kl_ref']:.4f} repeat {val_metrics['repeat@10']:.3f}"
            f" length {val_metrics['length@10']:.0f}"
        )
        if r.keep_checkpoints:
            save_model(run_dir / f"step_{state['step']}.pt", policy, step=state["step"])
        save_checkpoint(
            last_path,
            model=policy.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),
            state=state,
        )

    if state["step"] == 0 and not log_path.exists():
        run_eval({})  # pi_ref: the starting point of every trajectory
    stats: list[dict[str, float]] = []
    starts, lengths = prepared.window_starts, prepared.window_lengths
    while state["step"] < r.steps:
        rng = np.random.default_rng([cfg.seed, state["step"]])
        generator.manual_seed(cfg.seed * 1_000_003 + state["step"])
        ctx_starts, ctx_lengths = sample_contexts(store, starts, lengths, r.batch_size, rng)
        batch = to_tensors(store.gather(ctx_starts, ctx_lengths, seq_len, bounds.train_end), device)
        next_item = torch.from_numpy(
            np.asarray(store.columns["item_idx"])[ctx_starts + ctx_lengths].astype(np.int64)
        ).to(device)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            loss, step_stats = policy_loss(policy, ref, rm, reward, batch, next_item, r, generator)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        step_stats["grad_norm"] = float(nn.utils.clip_grad_norm_(policy.parameters(), r.grad_clip))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        stats.append(step_stats)
        state["step"] += 1
        early = r.eval_early_every > 0 and state["step"] <= r.eval_early_until
        every = r.eval_early_every if early else r.eval_every
        if state["step"] % every == 0 or state["step"] == r.steps:
            run_eval({k: float(np.mean([s[k] for s in stats])) for k in stats[0]})
            stats = []
    save_model(policy_path, policy, step=state["step"], rl=asdict(cfg.rl))
    return policy_path
