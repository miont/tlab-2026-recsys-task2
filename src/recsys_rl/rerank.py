"""Reranking baseline (plan §4.3, §8): the proxy RM applied at inference, without training.

The KL-regularised RL objective ``E_pi[r] - beta * KL(pi || pi_ref)`` has the closed-form
optimum ``pi*(a|s) ∝ pi_ref(a|s) * exp(r(s, a) / beta)``. RL amortises it into one model;
here it is computed directly for the frozen val / test contexts:

- the proxy RM scores the top-``n_candidates`` items of ``pi_ref`` with the reward of
  ``cfg.rl`` (R1-R6, R5 penalties), as in RL;
- items outside the top-N are not scored and get the ``pi_ref``-weighted mean reward of the
  top-N, so ``pi_beta`` tilts the top-N relative to that mean and leaves the rest of the
  distribution proportional to ``pi_ref``. ``KL(pi_beta || pi_ref)`` and the entropy are exact
  for this ``pi_beta`` over the whole vocabulary, so the points go on the same KL axis as the
  RL runs;
- the top-K of ``pi_beta`` merges the tilted top-N with the next ``K`` items of ``pi_ref``
  (untilted), which is the exact top-K of ``pi_beta``;
- ``beta = inf`` is ``pi_ref`` itself; ``beta = 0`` is the greedy rerank of the top-N by the
  reward alone (Petrov & Macdonald, 2024), a deterministic policy without a KL.

Every point is evaluated like an RL checkpoint (``rl.train_rl``): relevance of the top-100,
proxy / gold reward of the top-10, composition of the top-10, KL and entropy. If the reranker
reaches the RL trade-off curve, RL adds nothing beyond serving one model instead of the
policy + RM pair.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F

from recsys_rl import evaluate
from recsys_rl.config import Config
from recsys_rl.data import N_SPECIAL_TOKENS, EventStore, PreparedData, prepare
from recsys_rl.evaluate import EvalSet
from recsys_rl.model import GPT2Recommender, load_model, to_tensors
from recsys_rl.reward_model import RewardModel, context_repeat_counts
from recsys_rl.rewards import Reward, score_recommendations
from recsys_rl.rl import behaviour_metrics, item_features, load_frozen_rm
from recsys_rl.train import amp_dtype
from recsys_rl.utils import get_device

BETAS = (math.inf, 1.0, 0.3, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0)


@torch.no_grad()
def candidate_scores(
    ref: GPT2Recommender,
    rm: RewardModel,
    reward: Reward,
    store: EventStore,
    eval_set: EvalSet,
    n_candidates: int,
    k: int,
    batch_size: int = 128,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Per user of ``eval_set`` (CPU, float64): ``logp`` ``[U, N + k]`` and ``idx`` (vocabulary
    indices) of the top-``N + k`` items of ``pi_ref`` (fewer if the vocabulary is smaller),
    ``reward`` ``[U, N]`` of the top-N and ``h_out`` ``[U]``, the part ``-sum p log p`` of the
    ``pi_ref`` entropy outside the top-N."""
    device = next(ref.parameters()).device
    seq_len = ref.backbone.config.n_positions
    lengths = eval_set.context_ends - eval_set.context_starts
    out: dict[str, list[torch.Tensor]] = {"logp": [], "idx": [], "reward": [], "h_out": []}
    for b in range(0, eval_set.n_users, batch_size):
        sl = slice(b, b + batch_size)
        batch = to_tensors(
            store.gather(eval_set.context_starts[sl], lengths[sl], seq_len, eval_set.cutoff),
            device,
        )
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = ref.next_item_logits(batch)[:, N_SPECIAL_TOKENS:].float()
        logp = F.log_softmax(logits, -1)
        top = logp.topk(min(n_candidates + k, logp.shape[1]), dim=1)  # the tail may be short
        cand = top.indices[:, :n_candidates] + N_SPECIAL_TOKENS
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            probs = rm.score(batch, cand).float()
        r = reward(probs, cand, context_repeat_counts(batch["item_idx"], batch["mask"], cand))
        top_in = top.values[:, :n_candidates]
        h_all = -(logp.exp() * logp).sum(1)
        h_in = -(top_in.exp() * top_in).sum(1)
        out["logp"].append(top.values.double().cpu())
        out["idx"].append((top.indices + N_SPECIAL_TOKENS).cpu())
        out["reward"].append(r.double().cpu())
        out["h_out"].append((h_all - h_in).double().cpu())
    return {name: torch.cat(parts) for name, parts in out.items()}


def tilt(
    logp: torch.Tensor, reward: torch.Tensor, h_out: torch.Tensor, beta: float, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(top, kl, entropy)`` of ``pi_beta`` (see the module docstring) from
    ``candidate_scores``: ``top`` ``[U, k]`` positions into the ``N + k`` candidates, ``kl`` and
    ``entropy`` ``[U]`` (NaN for the greedy ``beta = 0``)."""
    n = reward.shape[1]
    lp_in = logp[:, :n]
    if beta == 0:
        nan = torch.full((len(logp),), math.nan, dtype=logp.dtype)
        return reward.topk(k, dim=1).indices, nan, nan
    p_in = lp_in.exp()
    mass_in = p_in.sum(1, keepdim=True)
    mean = (p_in * reward).sum(1, keepdim=True) / mass_in
    tau = (reward - mean) / beta if math.isfinite(beta) else torch.zeros_like(reward)
    log_out = torch.log1p(-mass_in.clamp(max=1 - 1e-12))  # pi_ref mass outside the top-N
    log_z = torch.logsumexp(torch.cat([log_out, lp_in + tau], 1), 1)  # [U]
    lq_in = lp_in + tau - log_z[:, None]  # log pi_beta of the top-N
    q_in = lq_in.exp()
    kl = (q_in * tau).sum(1) - log_z
    # outside: pi_beta = pi_ref / Z, so -sum q log q = (h_out + P_out * log Z) / Z
    entropy = -(q_in * lq_in).sum(1) + (h_out + log_out[:, 0].exp() * log_z) / log_z.exp()
    scores = torch.cat([lp_in + tau, logp[:, n:]], 1)
    return scores.topk(k, dim=1).indices, kl, entropy


def rerank_eval(
    cfg: Config,
    prepared: PreparedData | None = None,
    device: torch.device | None = None,
    betas: Sequence[float] = BETAS,
    n_candidates: int = 100,
    split: str = "val",
    name: str = "rerank",
    k: int = 100,
) -> pl.DataFrame:
    """One row per ``beta``: the metrics of ``rl.train_rl``'s log with the ``{split}_``
    prefix, ``run = name`` and ``step`` = position in ``betas`` (``pi_ref`` first), so the rows
    plot as one more trajectory. The reward and the RMs come from ``cfg.rl``."""
    r = cfg.rl
    device = device or get_device()
    prepared = prepared or prepare(cfg)
    store, items, bounds = prepared.store.in_memory(), prepared.items, prepared.bounds
    ref = load_model(Path(cfg.data_root) / "runs" / r.base_run / "best.pt", device).eval()
    seq_len = ref.backbone.config.n_positions
    rms = {"proxy": load_frozen_rm(cfg, r.reward_run, device)}
    if r.gold_run:
        rms["gold"] = load_frozen_rm(cfg, r.gold_run, device)
    reward = Reward.build(r, items, store, bounds.train_end, cfg.data).to(device)
    dtype = amp_dtype(device, r.amp)
    if 0 in betas and n_candidates < k:
        raise ValueError(f"the greedy rerank (beta = 0) needs n_candidates >= k = {k}")

    es = evaluate.build_eval_set(store, bounds, split, seq_len, cfg.data.listen_plus_threshold)
    idx_to_item = evaluate.idx_to_item_array(items)
    histories = es.histories(store, "item_idx")
    item_length, item_log_count, item_artist = item_features(items, ref.n_items, cfg.raw_dir)
    cands = candidate_scores(ref, rms["proxy"], reward, store, es, n_candidates, k, amp_dtype=dtype)
    rows = []
    for step, beta in enumerate(betas):
        top, kl, entropy = tilt(cands["logp"], cands["reward"], cands["h_out"], beta, k)
        recs = cands["idx"].gather(1, top).numpy()
        m = evaluate.evaluate_recs(idx_to_item[recs], es, len(items))
        m |= {"kl_ref": float(kl.mean()), "entropy": float(entropy.mean())}
        m |= behaviour_metrics(
            recs, histories, item_length, item_log_count, len(items), item_artist
        )
        for rm_name, model in rms.items():
            scores = score_recommendations(model, reward, store, es, recs[:, :10], amp_dtype=dtype)
            m[f"{rm_name}_reward@10"] = float(scores["reward"].mean())
            m[f"{rm_name}_ratio@10"] = float(scores["ratio"].mean())
            m[f"{rm_name}_ratio_std@10"] = float(scores["ratio_std"].mean())
        row = {"run": name, "step": step, "beta": beta, "n_candidates": n_candidates}
        rows.append(row | {f"{split}_{key}": value for key, value in m.items()})
    return pl.DataFrame(rows)
