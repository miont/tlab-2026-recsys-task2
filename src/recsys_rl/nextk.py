"""Next-K list generation with a list-level reward (level 2 of the slate study).

GPTRec (Petrov & Macdonald, 2024) builds a list item by item, each position conditioned on the
earlier ones, and tunes it with RL against a list-level metric. Here:

- **Policy** (``generate_slates``): the GPT-2 recommender picks position ``j`` from
  ``pi(. | context, a_1..a_{j-1})``: the earlier tracks of the list are appended to the context
  as recommendations (``is_organic = 0``) with the track's train-mean played ratio as its
  feedback bin (the model has no "unknown feedback" input) and the track's length bin; tracks
  already in the list are masked. ``rl.nextk_append = False`` keeps one distribution for all
  positions (Top-K / Plackett-Luce lists of one softmax, the level-1 policy). Contexts are cut to
  ``seq_len - slate_size`` listens, so the appended list always fits.
- **Reward**: the proxy slate RM (``slate_rm.SlateRewardModel``, ``rl.slate_reward_run``) —
  the mean expected played ratio over the positions, each conditioned on the list before it —
  plus ``rl.slate_diversity`` * distinct artists / ``slate_size``.
- **Objective** (``nextk_step``): RLOO over ``group_size`` lists per context with the exact
  log-probability of each generated list, plus ``kl_beta`` times the KL of the list
  distributions estimated on the generated list, ``sum_j KL(pi(.|s_j) || pi_ref(.|s_j))`` with
  both distributions renormalised over the items not yet in the list (``pi_ref`` generates in
  the same way, so ``pi_ref`` itself is the start of every run).
- **Inference-time baselines** (no training): ``search`` — per-position tilted greedy decoding,
  ``a_j = argmax log pi_ref(a | s_j) + (r_j(a | a_<j) - mean) / beta`` over the top-``N`` of
  ``pi_ref(. | s_j)``, with ``r_j`` the slate RM's expected ratio of the candidate at position
  ``j`` after the chosen prefix: the per-position reranker. It ignores the effect of ``a_j`` on
  the later positions, which the list reward and RL take into account; ``best_of_n`` — ``n``
  lists sampled from ``pi_ref`` (Next-K), the best by the proxy list reward.

Evaluation (``NextKEvaluator``) uses the deterministic list of each policy (greedy decoding,
masked repeats) as its top-10: relevance against the period's Listen+ (as every top-10 of the
study), composition, the item-level proxy / gold RM view of the 10 tracks (each scored alone)
and the list-level proxy / gold slate RM reward of the list.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import nn

from recsys_rl import evaluate, metrics
from recsys_rl.config import Config, save_config
from recsys_rl.data import N_SPECIAL_TOKENS, EventStore, PreparedData, prepare
from recsys_rl.generate import append_listen, ratio_bins
from recsys_rl.model import GPT2Recommender, load_model, save_model, to_tensors
from recsys_rl.reward_model import RewardModel, context_repeat_counts
from recsys_rl.rewards import NO_PENALTIES, Reward
from recsys_rl.rl import advantages, distinct_count, item_features, load_frozen_rm, sample_contexts
from recsys_rl.slate_rm import SlateRewardModel, finite, list_reward, load_frozen_slate_rm
from recsys_rl.train import amp_dtype, make_optimizer
from recsys_rl.utils import get_device, load_checkpoint, save_checkpoint, set_seed


class Appender:
    """Appends list items to a context batch as recommendations with the track's train-mean
    played ratio bin and length bin (see the module docstring)."""

    def __init__(self, reward: Reward, ratio_bin_edges: list[int]):
        pct = (reward.item_ratio * 100).round().long()
        self.ratio_bin = ratio_bins(pct, ratio_bin_edges)
        self.length_bin = reward.item_length_bin

    def __call__(self, batch: Mapping[str, torch.Tensor], items: torch.Tensor) -> dict:
        zeros = torch.zeros_like(items)
        return append_listen(
            batch,
            {
                "item_idx": items,
                "ratio_bin": self.ratio_bin[items],
                "length_bin": self.length_bin[items],
                "is_organic": zeros,
                "liked": zeros,
                "disliked": zeros,
            },
        )


def cut_contexts(starts: np.ndarray, lengths: np.ndarray, max_len: int):
    """Keep the last ``max_len`` listens of each context."""
    cut = np.maximum(lengths - max_len, 0)
    return starts + cut, lengths - cut


def repeat_batch(batch: Mapping[str, torch.Tensor], n: int) -> dict[str, torch.Tensor]:
    return {k: v.repeat_interleave(n, 0) for k, v in batch.items()}


def masked_log_softmax(logits: torch.Tensor, chosen: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.masked_fill(chosen, -math.inf), -1)


def masked_kl(lp: torch.Tensor, lr: torch.Tensor, chosen: torch.Tensor) -> torch.Tensor:
    """``[B]`` KL of two distributions renormalised over the items not ``chosen``; masked
    entries are zeroed before the product (``torch.where`` would pass NaN gradients from
    ``-inf - -inf``)."""
    lp, lr = lp.masked_fill(chosen, 0.0), lr.masked_fill(chosen, 0.0)
    return (lp.exp().masked_fill(chosen, 0.0) * (lp - lr)).sum(-1)


def _last(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return batch["mask"].sum(1) - 1


@torch.no_grad()
def generate_slates(
    model: GPT2Recommender,
    batch: Mapping[str, torch.Tensor],
    appender: Appender,
    slate_size: int,
    append: bool = True,
    greedy: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """``[B, L]`` lists (vocabulary indices) generated item by item, repeats masked."""
    s = N_SPECIAL_TOKENS
    cur = batch
    logits = None
    chosen = None
    out = []
    for j in range(slate_size):
        if logits is None or append:
            logits = model.next_item_logits(cur)[:, s:].float()
            if chosen is None:
                chosen = torch.zeros_like(logits, dtype=torch.bool)
        lp = masked_log_softmax(logits, chosen)
        if greedy:
            a = lp.argmax(1)
        else:
            a = torch.multinomial(lp.exp(), 1, generator=generator)[:, 0]
        out.append(a + s)
        chosen[torch.arange(len(a), device=a.device), a] = True
        if append and j < slate_size - 1:
            cur = appender(cur, a + s)
    return torch.stack(out, 1)


def query_hidden(
    model: GPT2Recommender,
    batch: Mapping[str, torch.Tensor],
    slates: torch.Tensor,
    appender: Appender,
    append: bool,
) -> torch.Tensor:
    """``[B, L, d]`` hidden states that choose each position of the given lists (one forward
    pass over the context with the first ``L - 1`` items appended)."""
    last = _last(batch)
    rows = torch.arange(len(last), device=last.device)
    length = slates.shape[1]
    if not append:
        return model(batch)[rows, last][:, None].expand(-1, length, -1)
    ext = batch
    for j in range(length - 1):
        ext = appender(ext, slates[:, j])
    hidden = model(ext)
    pos = last[:, None] + torch.arange(length, device=last.device)[None, :]
    return hidden[rows[:, None], pos]


def prefix_masks(slates: torch.Tensor, n_items: int) -> torch.Tensor:
    """``[B, L, V]``: at position ``j`` the items of positions ``< j`` (softmax indices)."""
    b, length = slates.shape
    masks = torch.zeros(b, length, n_items, dtype=torch.bool, device=slates.device)
    idx = slates - N_SPECIAL_TOKENS
    for j in range(1, length):
        masks[:, j] = masks[:, j - 1]
        masks[torch.arange(b, device=slates.device), j, idx[:, j - 1]] = True
    return masks


def nextk_step(
    policy: GPT2Recommender,
    ref: GPT2Recommender,
    slate_rm: SlateRewardModel,
    reward: Reward,
    appender: Appender,
    batch: Mapping[str, torch.Tensor],
    cfg,
    scaler: torch.amp.GradScaler,
    dtype: torch.dtype | None,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """One RL step (module docstring): generates, scores, backpropagates the loss (the
    per-position softmax terms one position at a time to bound memory) and returns statistics;
    the optimizer step is the caller's."""
    s, g, length = N_SPECIAL_TOKENS, cfg.group_size, cfg.slate_size
    device = batch["mask"].device
    rep = repeat_batch(batch, g)
    autocast = torch.autocast(device.type, dtype=dtype, enabled=dtype is not None)
    with autocast:
        slates = generate_slates(
            policy, rep, appender, length, cfg.nextk_append, generator=generator
        )
        with torch.no_grad():
            probs = slate_rm.score(rep, slates)
    item_r = list_reward(probs)
    slate_r = item_r
    stats: dict[str, float] = {"item_reward": item_r.mean().item()}
    if cfg.slate_diversity or reward.item_artist is not None:
        artists = distinct_count(reward.artist_categories(slates))
        slate_r = slate_r + cfg.slate_diversity * artists / length
        stats["slate_artists"] = artists.float().mean().item()
    r = slate_r.view(-1, g)
    adv = advantages(r, "rloo").reshape(-1)
    stats |= {"reward": r.mean().item(), "reward_std": r.std(1).mean().item()}

    with autocast:
        hidden = query_hidden(policy, rep, slates, appender, cfg.nextk_append)
        with torch.no_grad():
            ref_hidden = query_hidden(ref, rep, slates, appender, cfg.nextk_append)
    leaf = hidden.detach().requires_grad_(True)
    n_items = policy.n_items - s
    chosen = torch.zeros(len(slates), n_items, dtype=torch.bool, device=device)
    rows = torch.arange(len(slates), device=device)
    total_kl, total_logp, total_ent, objective = 0.0, 0.0, 0.0, 0.0
    for j in range(length):
        with autocast:
            logits = policy.scores(leaf[:, j])[:, s:].float()
            with torch.no_grad():
                ref_logits = ref.scores(ref_hidden[:, j])[:, s:].float()
        lp = masked_log_softmax(logits, chosen)
        lr = masked_log_softmax(ref_logits, chosen)
        p = lp.exp()
        kl = masked_kl(lp, lr, chosen)
        logp_a = lp[rows, slates[:, j] - s]
        pg = -(adv * logp_a).mean()
        loss = pg + cfg.kl_beta * kl.mean()
        scaler.scale(loss).backward()
        objective += pg.item()
        total_kl += kl.mean().item()
        total_logp += logp_a.mean().item()
        total_ent += -(p * lp.masked_fill(chosen, 0.0)).sum(-1).mean().item()
        chosen[rows, slates[:, j] - s] = True
    hidden.backward(leaf.grad)
    ctx_repeat = context_repeat_counts(rep["item_idx"], rep["mask"], slates)
    stats |= {
        "objective": objective,
        "kl": total_kl,
        "logp": total_logp,
        "entropy": total_ent / length,
        "repeat_share": (ctx_repeat > 0).float().mean().item(),
    }
    return stats


class NextKEvaluator:
    """Lists of policies and baselines for the users of an eval set, with per-user metrics."""

    def __init__(
        self,
        cfg: Config,
        prepared: PreparedData,
        device: torch.device,
        split: str = "val",
        max_users: int = 0,
        ref: GPT2Recommender | None = None,
        batch_size: int = 128,
    ):
        r = cfg.rl
        self.cfg, self.device, self.split, self.batch_size = cfg, device, split, batch_size
        self.store, self.items, bounds = prepared.store.in_memory(), prepared.items, prepared.bounds
        base = Path(cfg.data_root) / "runs" / r.base_run / "best.pt"
        self.ref = ref if ref is not None else load_model(base, device).eval()
        self.seq_len = self.ref.backbone.config.n_positions
        self.length = r.slate_size
        self.dtype = amp_dtype(device, r.amp)
        es = evaluate.build_eval_set(
            self.store, bounds, split, self.seq_len, cfg.data.listen_plus_threshold
        )
        if 0 < max_users < es.n_users:
            es = es.subset(
                np.sort(np.random.default_rng(cfg.seed).choice(es.n_users, max_users, False))
            )
        self.es = es
        lengths = es.context_ends - es.context_starts
        self.ctx_starts, self.ctx_lengths = cut_contexts(
            es.context_starts, lengths, self.seq_len - self.length
        )
        self.reward = Reward.build(
            r, self.items, self.store, bounds.train_end, cfg.data, cfg.raw_dir
        ).to(device)
        self.profile_reward = self.reward.variant(reward="debiased_ratio", **NO_PENALTIES)
        self.appender = Appender(self.reward, cfg.data.ratio_bin_edges)
        self.rms: dict[str, RewardModel] = {"proxy": load_frozen_rm(cfg, r.reward_run, device)}
        if r.gold_run:
            self.rms["gold"] = load_frozen_rm(cfg, r.gold_run, device)
        self.slate_rms = {"proxy": load_frozen_slate_rm(cfg, r.slate_reward_run, device)}
        if r.slate_gold_run:
            self.slate_rms["gold"] = load_frozen_slate_rm(cfg, r.slate_gold_run, device)
        self.idx_to_item = evaluate.idx_to_item_array(self.items)
        self.histories = es.histories(self.store, "item_idx")
        self.item_length, self.item_log_count, self.item_artist = item_features(
            self.items, self.ref.n_items, cfg.raw_dir
        )
        self.max_pcount = float(np.sort(self.items["count"].to_numpy())[-10:].mean())

    def batches(self):
        for b in range(0, self.es.n_users, self.batch_size):
            sl = slice(b, b + self.batch_size)
            raw = self.store.gather(
                self.ctx_starts[sl], self.ctx_lengths[sl], self.seq_len, self.es.cutoff
            )
            yield sl, to_tensors(raw, self.device)

    def autocast(self):
        dtype = self.dtype
        return torch.autocast(self.device.type, dtype=dtype, enabled=dtype is not None)

    @torch.no_grad()
    def policy_slates(
        self, model: GPT2Recommender | None, append: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(slates, kl)``: greedy lists of ``model`` (None = ``pi_ref``) and the list KL to
        ``pi_ref`` on them (``sum_j`` of the per-position KL)."""
        model = model or self.ref
        slates, kls = [], []
        for _, batch in self.batches():
            with self.autocast():
                sl = generate_slates(model, batch, self.appender, self.length, append, greedy=True)
                kls.append(self._list_kl(model, batch, sl, append).cpu())
            slates.append(sl.cpu())
        return torch.cat(slates).numpy(), torch.cat(kls).numpy()

    def _list_kl(self, model, batch, slates, append) -> torch.Tensor:
        if model is self.ref:
            return torch.zeros(len(slates))
        s = N_SPECIAL_TOKENS
        h = query_hidden(model, batch, slates, self.appender, append)
        h_ref = query_hidden(self.ref, batch, slates, self.appender, append)
        chosen = torch.zeros(len(slates), model.n_items - s, dtype=torch.bool, device=h.device)
        rows = torch.arange(len(slates), device=h.device)
        total = torch.zeros(len(slates), device=h.device)
        for j in range(slates.shape[1]):
            lp = masked_log_softmax(model.scores(h[:, j])[:, s:].float(), chosen)
            lr = masked_log_softmax(self.ref.scores(h_ref[:, j])[:, s:].float(), chosen)
            total += masked_kl(lp, lr, chosen)
            chosen[rows, slates[:, j] - s] = True
        return total.float()

    @torch.no_grad()
    def search(
        self, beta: float, n_candidates: int = 100, append: bool = True, diversity: float = 0.0
    ) -> np.ndarray:
        """Per-position tilted greedy lists by the proxy slate RM (module docstring)."""
        s = N_SPECIAL_TOKENS
        rm = self.slate_rms["proxy"]
        out = []
        for _, batch in self.batches():
            with self.autocast():
                hidden = rm.context_state(batch)
                cur, logits = batch, None
                chosen = None
                prefix = torch.zeros(len(hidden), 0, dtype=torch.long, device=self.device)
                for j in range(self.length):
                    if logits is None or append:
                        logits = self.ref.next_item_logits(cur)[:, s:].float()
                        if chosen is None:
                            chosen = torch.zeros_like(logits, dtype=torch.bool)
                    lp = masked_log_softmax(logits, chosen)
                    top = lp.topk(n_candidates, dim=1)
                    cand = top.indices + s
                    rep_p = context_repeat_counts(batch["item_idx"], batch["mask"], prefix)
                    rep_c = context_repeat_counts(batch["item_idx"], batch["mask"], cand)
                    probs = rm.position_scores(hidden, prefix, cand, rep_p, rep_c)
                    r = list_reward(probs[..., None, :])  # [B, M]
                    if diversity:
                        cats = self.reward.artist_categories(cand)
                        seen = self.reward.artist_categories(prefix)
                        new = ~(cats[:, :, None] == seen[:, None, :]).any(-1)
                        r = r + diversity * new.float()
                    score = _tilted(top.values, r, beta)
                    # small vocabularies: the top-N may reach items already in the list
                    score = score.masked_fill(torch.isinf(top.values), -math.inf)
                    a = cand.gather(1, score.argmax(1, keepdim=True))[:, 0]
                    prefix = torch.cat([prefix, a[:, None]], 1)
                    chosen[torch.arange(len(a), device=a.device), a - s] = True
                    if append and j < self.length - 1:
                        cur = self.appender(cur, a)
            out.append(prefix.cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def best_of_n(self, n: int, append: bool = True, seed: int = 0) -> np.ndarray:
        """The best of ``n`` lists sampled from ``pi_ref`` by the proxy list reward."""
        rm = self.slate_rms["proxy"]
        generator = torch.Generator(device=self.device).manual_seed(seed)
        out = []
        for _, batch in self.batches():
            rep = repeat_batch(batch, n)
            with self.autocast():
                sl = generate_slates(
                    self.ref, rep, self.appender, self.length, append, generator=generator
                )
                r = list_reward(rm.score(rep, sl)).view(-1, n)
            best = r.argmax(1)
            out.append(sl.view(-1, n, self.length)[torch.arange(len(best)), best].cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def list_rewards(self, slates: np.ndarray) -> dict[str, np.ndarray]:
        """Per user: the proxy / gold slate RM list reward of the lists."""
        out: dict[str, list[np.ndarray]] = {name: [] for name in self.slate_rms}
        for sl, batch in self.batches():
            lists = torch.from_numpy(slates[sl]).to(self.device)
            for name, rm in self.slate_rms.items():
                with self.autocast():
                    out[name].append(list_reward(rm.score(batch, lists)).float().cpu().numpy())
        return {f"{name}_slate@10": np.concatenate(v) for name, v in out.items()}

    def per_user(self, method: str, slates: np.ndarray, kl: np.ndarray | None = None):
        """One row per user (``significance.KEYS`` + metrics) of the lists ``slates``."""
        from recsys_rl.significance import _top10_profile

        values = metrics.relevance_per_user(self.idx_to_item[slates], self.es.targets, ks=(10,))
        values |= _top10_profile(
            slates, self.histories, self.item_length, self.item_log_count, self.item_artist,
            self.max_pcount, self.rms, self.profile_reward, self.store, self.es, self.dtype,
        )  # fmt: skip
        values |= self.list_rewards(slates)
        values["kl_ref"] = np.full(len(slates), np.nan) if kl is None else kl
        keys = {"split": self.split, "method": method, "exclude_seen": False, "user": self.es.users}
        return pl.DataFrame(keys | values)


@torch.no_grad()
def reranker_lists(ev: NextKEvaluator, beta: float, n_candidates: int = 100) -> np.ndarray:
    """Top-``slate_size`` lists of the item-level reranker ``pi_ref * exp(r / beta)`` (R1, the
    proxy item-level RM; ``rerank.py``) for the users of ``ev``."""
    from recsys_rl.rerank import candidate_scores, tilt

    reward = ev.reward.variant(reward="ratio", **NO_PENALTIES)
    cands = candidate_scores(
        ev.ref, ev.rms["proxy"], reward, ev.store, ev.es, n_candidates, ev.length,
        amp_dtype=ev.dtype,
    )  # fmt: skip
    top, _, _ = tilt(cands["logp"], cands["reward"], cands["h_out"], beta, ev.length)
    return cands["idx"].gather(1, top).numpy()


def _tilted(logp: torch.Tensor, reward: torch.Tensor, beta: float) -> torch.Tensor:
    if beta == 0:
        return reward
    if not math.isfinite(beta):
        return logp
    p = logp.exp()
    mean = (p * reward).sum(1, keepdim=True) / p.sum(1, keepdim=True)
    return logp + (reward - mean) / beta


def summarise(frame: pl.DataFrame) -> dict[str, float]:
    skip = {"split", "method", "exclude_seen", "user"}
    return {c: float(frame[c].mean()) for c in frame.columns if c not in skip}


def train_nextk(
    cfg: Config, prepared: PreparedData | None = None, device: torch.device | None = None
) -> Path:
    """Train (or resume) the Next-K policy of ``cfg.run_name`` (``rl.method = nextk``);
    returns ``policy.pt``. Evaluations go to ``log.jsonl`` with the ``val_`` prefix."""
    r = cfg.rl
    run_dir = cfg.run_dir
    last_path, policy_path = run_dir / "last.pt", run_dir / "policy.pt"
    log_path = run_dir / "log.jsonl"
    if policy_path.exists() and log_path.exists():
        last = json.loads(log_path.read_text().splitlines()[-1])
        if last["step"] >= r.steps:
            print(f"{cfg.run_name} is done at step {last['step']}")
            return policy_path
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    set_seed(cfg.seed)
    device = device or get_device()
    prepared = prepared or prepare(cfg)
    base_path = Path(cfg.data_root) / "runs" / r.base_run / "best.pt"
    policy = load_model(base_path, device).eval()
    ref = load_model(base_path, device).eval().requires_grad_(False)
    ev = NextKEvaluator(cfg, prepared, device, "val", r.eval_users, ref=ref)
    store, bounds = ev.store, prepared.bounds
    seq_len, length = ev.seq_len, r.slate_size
    slate_rm = ev.slate_rms["proxy"]
    optimizer = make_optimizer(policy, r.lr, r.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(1, r.warmup_steps))
    )
    dtype = ev.dtype
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
    print(
        f"device {device}, nextk append={r.nextk_append}, beta {r.kl_beta}, G {r.group_size}, "
        f"L {length}, diversity {r.slate_diversity}, {r.steps} steps, {ev.es.n_users} val users"
    )

    def run_eval(train_stats: Mapping[str, float]) -> None:
        slates, kl = ev.policy_slates(policy, r.nextk_append)
        record = {"step": state["step"], "time": time.time()}
        record |= {f"train_{k}": v for k, v in train_stats.items()}
        record |= {f"val_{k}": v for k, v in summarise(ev.per_user("policy", slates, kl)).items()}
        with log_path.open("a") as f:
            f.write(json.dumps(finite(record)) + "\n")
        print(
            f"step {state['step']} val ndcg@10 {record['val_ndcg@10']:.4f} "
            f"proxy slate {record['val_proxy_slate@10']:.4f}"
            + (f" gold slate {record['val_gold_slate@10']:.4f}" if "gold" in ev.slate_rms else "")
            + f" kl {record['val_kl_ref']:.3f} artists {record.get('val_artists@10', math.nan):.2f}"
        )
        save_checkpoint(
            last_path,
            model=policy.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),
            state=state,
        )

    if state["step"] == 0 and not log_path.exists():
        run_eval({})
    stats: list[dict[str, float]] = []
    starts, lengths = prepared.window_starts, prepared.window_lengths
    while state["step"] < r.steps:
        rng = np.random.default_rng([cfg.seed, state["step"]])
        generator.manual_seed(cfg.seed * 1_000_003 + state["step"])
        ctx_starts, ctx_lengths = sample_contexts(store, starts, lengths, r.batch_size, rng)
        ctx_starts, ctx_lengths = cut_contexts(ctx_starts, ctx_lengths, seq_len - length)
        batch = to_tensors(store.gather(ctx_starts, ctx_lengths, seq_len, bounds.train_end), device)
        optimizer.zero_grad(set_to_none=True)
        step_stats = nextk_step(
            policy, ref, slate_rm, ev.reward, ev.appender, batch, r, scaler, dtype, generator
        )
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


@torch.no_grad()
def nextk_reward_scale(
    cfg: Config,
    prepared: PreparedData | None = None,
    device: torch.device | None = None,
    n_contexts: int = 1024,
) -> dict[str, float]:
    """Mean and in-group std of the proxy list reward of ``group_size`` lists sampled from
    ``pi_ref`` per training context, for Next-K and Top-K (Plackett-Luce) lists; ``kl_beta`` is
    set in units of this std, as ``rl.reward_scales`` does for the item-level rewards."""
    r = cfg.rl
    device = device or get_device()
    prepared = prepared or prepare(cfg)
    store, bounds = prepared.store.in_memory(), prepared.bounds
    ref = load_model(Path(cfg.data_root) / "runs" / r.base_run / "best.pt", device).eval()
    seq_len = ref.backbone.config.n_positions
    rm = load_frozen_slate_rm(cfg, r.slate_reward_run, device)
    reward = Reward.build(r, prepared.items, store, bounds.train_end, cfg.data, cfg.raw_dir)
    appender = Appender(reward.to(device), cfg.data.ratio_bin_edges)
    dtype = amp_dtype(device, r.amp)
    rng = np.random.default_rng([cfg.seed, 1_000_000_007])
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    out: dict[str, list[torch.Tensor]] = {"nextk": [], "topk": []}
    for b in range(0, n_contexts, r.batch_size):
        n = min(r.batch_size, n_contexts - b)
        starts, lengths = sample_contexts(
            store, prepared.window_starts, prepared.window_lengths, n, rng
        )
        starts, lengths = cut_contexts(starts, lengths, seq_len - r.slate_size)
        batch = to_tensors(store.gather(starts, lengths, seq_len, bounds.train_end), device)
        rep = repeat_batch(batch, r.group_size)
        for name, append in (("nextk", True), ("topk", False)):
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                sl = generate_slates(ref, rep, appender, r.slate_size, append, generator=generator)
                out[name].append(list_reward(rm.score(rep, sl)).float().view(-1, r.group_size))
    res = {}
    for name, parts in out.items():
        v = torch.cat(parts).double()
        res[f"{name}_mean"] = v.mean().item()
        res[f"{name}_group_std"] = v.std(1).mean().item()
    return res


def nextk_per_user(
    cfg: Config,
    prepared: PreparedData | None = None,
    device: torch.device | None = None,
    split: str = "val",
    policies: Mapping[str, tuple[str | Path | None, bool]] | None = None,
    searches: Mapping[str, tuple[float, bool]] | None = None,
    best_of: Mapping[str, int] | None = None,
    extra: Mapping[str, Callable[[NextKEvaluator], np.ndarray]] | None = None,
    max_users: int = 0,
) -> pl.DataFrame:
    """Per-user metrics (``significance.KEYS`` + ``NextKEvaluator.per_user``) of the greedy
    lists of ``policies`` (name -> (checkpoint, None = ``pi_ref``; append)), the ``searches``
    (name -> (beta, append)), Best-of-N (name -> n) and ``extra`` (name -> function of the
    evaluator returning lists), on ``split``."""
    device = device or get_device()
    prepared = prepared or prepare(cfg)
    ev = NextKEvaluator(cfg, prepared, device, split, max_users)
    frames = []
    for name, (path, append) in (policies or {}).items():
        model = None if path is None else load_model(path, device).eval()
        slates, kl = ev.policy_slates(model, append)
        frames.append(ev.per_user(name, slates, kl))
        del model
    for name, (beta, append) in (searches or {}).items():
        frames.append(ev.per_user(name, ev.search(beta, append=append)))
    for name, n in (best_of or {}).items():
        frames.append(ev.per_user(name, ev.best_of_n(n, seed=cfg.seed)))
    for name, fn in (extra or {}).items():
        frames.append(ev.per_user(name, fn(ev)))
    return pl.concat(frames, how="diagonal_relaxed")


def logged_list_effects(
    store: EventStore, rows: np.ndarray, lengths: np.ndarray, item_artist: np.ndarray | None
) -> pl.DataFrame:
    """Descriptive check of list effects in the logs: mean played ratio by position and by
    whether the track's artist already occurred earlier in the same logged slate."""
    ratio = np.minimum(np.asarray(store.columns["played_ratio_pct"]), 100) / 100
    idx = np.asarray(store.columns["item_idx"])
    out = []
    for row, n in zip(rows, lengths, strict=True):
        items = idx[row : row + n]
        art = item_artist[items] if item_artist is not None else -1 - items
        art = np.where(art >= 0, art, -1 - items)
        for j in range(n):
            out.append((j + 1, bool((art[:j] == art[j]).any()), ratio[row + j]))
    frame = pl.DataFrame(out, schema=["position", "artist_seen", "ratio"], orient="row")
    return (
        frame.group_by("position", "artist_seen")
        .agg(n=pl.len(), mean_ratio=pl.col("ratio").mean())
        .sort("position", "artist_seen")
    )
