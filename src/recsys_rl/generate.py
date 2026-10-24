"""Multi-step autoregressive generation (stage 6.1, plan §8, task question 4).

For every user of a val / test eval set the policy recommends ``steps`` tracks one after
another: the recommended track is appended to the context as if the user had listened to it,
and the next recommendation is made from the extended context. The generated slate drifts from
the real history (exposure bias): the user never heard these tracks, so the context and the
reward model's inputs leave the support of the logs.

The model reads the feedback of every listen (played-ratio bin, like / dislike flags, organic
flag, track-length bin) and has no "unknown feedback" token, so the appended listen needs
feedback. Modes (``MODES``):

- ``accept``: the track was played to the end, no like / dislike — naive generation that takes
  every recommendation as consumed (plan mode (a), the item alone);
- ``expected``: the played ratio is the expected ratio of the feedback RM (its bin), no like /
  dislike (their probabilities are far below 1/2) — plan mode (b);
- ``sampled``: the outcome is drawn from the feedback RM: skip (ratio < 0.3) with ``p(skip)``,
  Listen+ (>= 0.5) with ``p(listen_plus)``, a partial play otherwise, each mapped to a typical
  ratio (``OUTCOME_PCT``); like and dislike are Bernoulli draws — plan mode (c), a stochastic
  user simulator.

Every appended listen is a recommendation (``is_organic = 0``) with the track's train-median
length bin. The feedback RM is the proxy RM (what the system itself knows); rewards along the
rollout are scored by the proxy and the gold RM.

Policies (``Policy``): a ``GPT2Recommender`` (``pi_ref`` or an RL checkpoint) or the reranker
``pi_beta ∝ pi_ref * exp(r / beta)`` over the top-N of ``pi_ref`` (``rerank``), recomputed on
the extended context at every step. Decoding: ``greedy`` (the most likely track; repeats of a
track inside the slate are not masked, so loops show up) or ``sample``.

Per user and step ``t = 1..steps`` (``rollout``) — step 1 is the ordinary evaluation:

- relevance of the step's top-100 against the period's real Listen+ targets (``ndcg@10``,
  ``recall@100``, ...) and ``hit``: the generated track is among the targets;
- distribution: ``kl_ref`` = ``KL(pi(.|s_t) || pi_ref(.|s_t))`` on the same extended context
  and ``entropy``;
- the generated track: ``in_history`` (heard before the cutoff), ``in_slate`` (already
  generated at an earlier step), ``length``, ``log_count`` (train popularity), cumulative
  ``slate_distinct`` / ``slate_artists`` (distinct tracks / artists generated so far);
- RM view of the generated track: ``{proxy,gold}_ratio``, ``{proxy,gold}_ratio_std``
  (ensemble spread: does the RM extrapolate more with ``t``?), ``{proxy,gold}_like``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from recsys_rl import evaluate, metrics
from recsys_rl.config import Config
from recsys_rl.data import N_SPECIAL_TOKENS, EventStore, PreparedData, prepare
from recsys_rl.evaluate import EvalSet
from recsys_rl.model import INPUT_COLUMNS, GPT2Recommender, load_model, to_tensors
from recsys_rl.reward_model import (
    HEAD_INDEX,
    RewardModel,
    context_repeat_counts,
    item_length_bins,
)
from recsys_rl.rewards import Reward
from recsys_rl.rl import item_features, load_frozen_rm, tilted_target
from recsys_rl.train import amp_dtype
from recsys_rl.utils import get_device

MODES = ("accept", "expected", "sampled")
DECODINGS = ("greedy", "sample")
OUTCOME_PCT = {"skip": 3, "partial": 40, "listen_plus": 100}  # typical played ratio, %


def ratio_bins(pct: torch.Tensor, edges: Sequence[int]) -> torch.Tensor:
    """Played-ratio bin of ``pct`` (0..100+), as ``data.bin_index``."""
    return (pct[..., None] > torch.as_tensor(edges, device=pct.device)).sum(-1)


def append_listen(
    batch: Mapping[str, torch.Tensor], features: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """The right-padded ``batch`` (``INPUT_COLUMNS`` + ``mask``) with one more listen per row
    (``features``: ``[B]`` per input column). A full row drops its oldest listen, as a training
    window would."""
    lengths = batch["mask"].sum(1)
    full = lengths == batch["mask"].shape[1]
    pos = torch.where(full, lengths - 1, lengths)
    rows = torch.arange(len(lengths), device=lengths.device)
    out = {}
    for name in (*INPUT_COLUMNS, "mask"):
        values = batch[name].clone()
        values[full] = values[full].roll(-1, dims=1)
        values[rows, pos] = True if name == "mask" else features[name].to(values.dtype)
        out[name] = values
    return out


class Policy:
    """Next-item log-probabilities ``[B, n_items - N_SPECIAL_TOKENS]`` of a model or the
    reranker; ``ref_logp`` is ``pi_ref``'s on the same batch (computed once per step)."""

    def __init__(
        self,
        model: GPT2Recommender | None = None,
        rm: RewardModel | None = None,
        reward: Reward | None = None,
        beta: float = math.inf,
        n_candidates: int = 100,
    ):
        """``model`` is a policy network (None = ``pi_ref``); the reranker needs ``rm``,
        ``reward`` and ``beta``."""
        if (rm is None) != (reward is None) or (rm is not None and model is not None):
            raise ValueError("a policy is a model or a reranker (rm + reward), not both")
        self.model, self.rm, self.reward = model, rm, reward
        self.beta, self.n_candidates = beta, n_candidates

    def log_probs(self, batch: Mapping[str, torch.Tensor], ref_logp: torch.Tensor) -> torch.Tensor:
        if self.model is not None:
            logits = self.model.next_item_logits(batch)[:, N_SPECIAL_TOKENS:].float()
            return F.log_softmax(logits, -1)
        if self.rm is None or math.isinf(self.beta):
            return ref_logp
        top = ref_logp.topk(self.n_candidates, dim=-1).indices
        cand = top + N_SPECIAL_TOKENS
        probs = self.rm.score(batch, cand).float()
        repeat = context_repeat_counts(batch["item_idx"], batch["mask"], cand)
        r = self.reward(probs, cand, repeat, weights=ref_logp.gather(1, top).exp())
        return tilted_target(ref_logp, top, r, self.beta)


def simulated_feedback(
    probs: torch.Tensor,
    items: torch.Tensor,
    mode: str,
    item_length_bin: torch.Tensor,
    ratio_bin_edges: Sequence[int],
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Input features ``[B]`` of the appended listens ``items`` (vocabulary indices);
    ``probs`` ``[E, B, 1, H]`` is the feedback RM's view of them (see the module docstring)."""
    mean = probs.mean(0)[:, 0]  # [B, H]
    zeros = torch.zeros_like(items)
    if mode == "accept":
        pct = torch.full_like(items, OUTCOME_PCT["listen_plus"])
        liked, disliked = zeros, zeros
    elif mode == "expected":
        pct = (mean[:, HEAD_INDEX["ratio"]] * 100).round().long()
        liked, disliked = zeros, zeros
    elif mode == "sampled":
        p_skip, p_plus = mean[:, HEAD_INDEX["skip"]], mean[:, HEAD_INDEX["listen_plus"]]
        p = torch.stack([p_skip, (1 - p_skip - p_plus).clamp_min(0), p_plus], 1)
        outcome = torch.multinomial(p / p.sum(1, keepdim=True), 1, generator=generator)[:, 0]
        pct = torch.as_tensor(list(OUTCOME_PCT.values()), device=items.device)[outcome]
        u = torch.rand(len(items), 2, device=items.device, generator=generator)
        liked = (u[:, 0] < mean[:, HEAD_INDEX["like"]]).long()
        disliked = (u[:, 1] < mean[:, HEAD_INDEX["dislike"]]).long()
    else:
        raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
    return {
        "item_idx": items,
        "ratio_bin": ratio_bins(pct, ratio_bin_edges),
        "length_bin": item_length_bin[items],
        "is_organic": zeros,
        "liked": liked,
        "disliked": disliked,
    }


@torch.no_grad()
def rollout(
    policy: Policy,
    ref: GPT2Recommender,
    feedback_rm: RewardModel,
    rms: Mapping[str, RewardModel],
    store: EventStore,
    eval_set: EvalSet,
    idx_to_item: np.ndarray,
    item_length_bin: torch.Tensor,
    ratio_bin_edges: Sequence[int],
    item_length: np.ndarray,
    item_log_count: np.ndarray,
    item_artist: np.ndarray | None = None,
    mode: str = "accept",
    decoding: str = "greedy",
    steps: int = 5,
    k: int = 100,
    batch_size: int = 128,
    seed: int = 0,
    amp_dtype: torch.dtype | None = None,
) -> pl.DataFrame:
    """One row per (step, user) with the metrics of the module docstring; ``user`` is the
    store index and ``step`` starts at 1."""
    if decoding not in DECODINGS:
        raise ValueError(f"unknown decoding {decoding!r}, expected one of {DECODINGS}")
    device = next(ref.parameters()).device
    seq_len = ref.backbone.config.n_positions
    generator = torch.Generator(device=device).manual_seed(seed)
    lengths = eval_set.context_ends - eval_set.context_starts
    histories = eval_set.histories(store, "item_idx")
    artist = None if item_artist is None else torch.from_numpy(item_artist).to(device)
    item_length_bin = item_length_bin.to(device)
    frames = []
    for b in range(0, eval_set.n_users, batch_size):
        sl = slice(b, b + batch_size)
        raw = to_tensors(
            store.gather(eval_set.context_starts[sl], lengths[sl], seq_len, eval_set.cutoff),
            device,
        )
        batch = {name: raw[name] for name in (*INPUT_COLUMNS, "mask")}
        targets = eval_set.targets[sl]
        target_sets = [set(t.tolist()) for t in targets]
        hist = histories[sl]
        generated: list[torch.Tensor] = []
        for step in range(1, steps + 1):
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                ref_logits = ref.next_item_logits(batch)[:, N_SPECIAL_TOKENS:].float()
                ref_logp = F.log_softmax(ref_logits, -1)
                logp = policy.log_probs(batch, ref_logp).float()
            p = logp.exp()
            recs = (logp.topk(k, dim=1).indices + N_SPECIAL_TOKENS).cpu().numpy()
            if decoding == "greedy":
                action = logp.argmax(1)
            else:
                action = torch.multinomial(p, 1, generator=generator)[:, 0]
            item = action + N_SPECIAL_TOKENS
            item_np = item.cpu().numpy()
            values = metrics.relevance_per_user(idx_to_item[recs], targets, ks=(10, k))
            values |= {
                "kl_ref": (p * (logp - ref_logp)).sum(1).cpu().numpy(),
                "entropy": -(p * logp).sum(1).cpu().numpy(),
                "hit": np.array(
                    [idx_to_item[i] in t for i, t in zip(item_np, target_sets, strict=True)]
                ),
                "in_history": np.array([i in h for i, h in zip(item_np, hist, strict=True)]),
                "in_slate": (
                    torch.stack(generated, 1).eq(item[:, None]).any(1).cpu().numpy()
                    if generated
                    else np.zeros(len(item_np), dtype=bool)
                ),
                "length": item_length[item_np],
                "log_count": item_log_count[item_np],
            }
            generated.append(item)
            slate = torch.stack(generated, 1)
            values["slate_distinct"] = _distinct(slate).cpu().numpy()
            if artist is not None:
                known = torch.where(artist[slate] >= 0, artist[slate], -1 - slate)
                values["slate_artists"] = _distinct(known).cpu().numpy()
            feedback_probs = None
            for name, rm in rms.items():
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    probs = rm.score(batch, item[:, None]).float()  # [E, B, 1, H]
                ratio = probs[..., HEAD_INDEX["ratio"]][..., 0]
                values[f"{name}_ratio"] = ratio.mean(0).cpu().numpy()
                values[f"{name}_ratio_std"] = (
                    (ratio.std(0) if len(ratio) > 1 else torch.zeros_like(ratio[0])).cpu().numpy()
                )
                values[f"{name}_like"] = (
                    probs[..., HEAD_INDEX["like"]][..., 0].mean(0).cpu().numpy()
                )
                if rm is feedback_rm:
                    feedback_probs = probs
            if feedback_probs is None:
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    feedback_probs = feedback_rm.score(batch, item[:, None]).float()
            frames.append(pl.DataFrame({"step": step, "user": eval_set.users[sl]} | values))
            features = simulated_feedback(
                feedback_probs, item, mode, item_length_bin, ratio_bin_edges, generator
            )
            batch = append_listen(batch, features)
    return pl.concat(frames).sort("step", "user")


def _distinct(values: torch.Tensor) -> torch.Tensor:
    """Number of distinct values per row of ``[B, n]``."""
    srt = values.sort(1).values
    return 1 + (srt[:, 1:] != srt[:, :-1]).sum(1)


def rollout_eval(
    cfg: Config,
    prepared: PreparedData | None = None,
    device: torch.device | None = None,
    split: str = "test",
    policies: Mapping[str, str | Path | None] | None = None,
    rerank: Mapping[str, float] | None = None,
    modes: Sequence[str] = MODES,
    decodings: Sequence[str] = DECODINGS,
    steps: int = 5,
    n_candidates: int = 100,
    n_users: int | None = None,
    k: int = 100,
) -> pl.DataFrame:
    """``rollout`` of every policy (name -> checkpoint, None = ``pi_ref``) and reranker point
    (name -> beta, with the reward of ``cfg.rl``) in every mode and decoding, with columns
    ``policy``, ``mode``, ``decoding``. ``n_users`` takes a seeded subset of the eval set;
    relevance is at 10 and ``k``."""
    r = cfg.rl
    device = device or get_device()
    prepared = prepared or prepare(cfg)
    store, items, bounds = prepared.store.in_memory(), prepared.items, prepared.bounds
    ref = load_model(Path(cfg.data_root) / "runs" / r.base_run / "best.pt", device).eval()
    seq_len = ref.backbone.config.n_positions
    rms = {"proxy": load_frozen_rm(cfg, r.reward_run, device)}
    if r.gold_run:
        rms["gold"] = load_frozen_rm(cfg, r.gold_run, device)
    reward = Reward.build(r, items, store, bounds.train_end, cfg.data, cfg.raw_dir).to(device)
    dtype = amp_dtype(device, r.amp)
    es = evaluate.build_eval_set(store, bounds, split, seq_len, cfg.data.listen_plus_threshold)
    if n_users is not None and n_users < es.n_users:
        rng = np.random.default_rng(cfg.seed)
        es = es.subset(np.sort(rng.choice(es.n_users, n_users, replace=False)))
    item_length, item_log_count, item_artist = item_features(items, ref.n_items, cfg.raw_dir)
    common = {
        "ref": ref,
        "feedback_rm": rms["proxy"],
        "rms": rms,
        "store": store,
        "eval_set": es,
        "idx_to_item": evaluate.idx_to_item_array(items),
        "item_length_bin": item_length_bins(items, cfg.data),
        "ratio_bin_edges": cfg.data.ratio_bin_edges,
        "item_length": item_length,
        "item_log_count": item_log_count,
        "item_artist": item_artist,
        "steps": steps,
        "k": k,
        "seed": cfg.seed,
        "amp_dtype": dtype,
    }
    specs: dict[str, tuple[str | Path | None, float | None]] = {
        **{name: (path, None) for name, path in (policies or {}).items()},
        **{name: (None, beta) for name, beta in (rerank or {}).items()},
    }
    frames = []
    for name, (path, beta) in specs.items():
        if beta is not None:
            policy = Policy(rm=rms["proxy"], reward=reward, beta=beta, n_candidates=n_candidates)
        elif path is None:
            policy = Policy()
        else:
            policy = Policy(load_model(path, device).eval())
        for mode in modes:
            for decoding in decodings:
                rows = rollout(policy, mode=mode, decoding=decoding, **common)
                frames.append(
                    rows.with_columns(
                        policy=pl.lit(name), mode=pl.lit(mode), decoding=pl.lit(decoding)
                    )
                )
        del policy
    return pl.concat(frames).select(
        "policy", "mode", "decoding", pl.exclude("policy", "mode", "decoding")
    )
