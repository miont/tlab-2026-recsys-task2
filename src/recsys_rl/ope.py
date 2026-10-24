"""Off-policy evaluation on the logged recommendations of the val / test day (stage 5.3).

The real reward of a policy is only observed for the tracks the production recommender
showed, so the value of another policy is estimated by importance sampling against the
behaviour model ``pi_b`` (``runs/<rl.behavior_run>/best.pt``, Matveev et al., 2026, §OPE):

- data: for every user of the eval set, the last ``seq_len`` listens up to the end of the
  period; the actions are the period's logged recommendations (``rl.logged_actions``: non-organic,
  known track, a trajectory is a run of consecutive recommendations within a session), the
  reward is ``like + ratio / 10`` or the played ratio of the logs (likes given before the end
  of the period). A user with more than ``seq_len`` period listens keeps only the last ones;
- ``behavior``: the logged discounted return ``mean_i sum_t gamma^s r_t`` per trajectory
  (``s`` the index of the action in its trajectory);
- ``step_is``: ``mean_i sum_t gamma^s min(clip, w_{0:s}) r_t`` with the cumulative weight
  ``w_{0:s} = prod_{tau <= s} pi(a_tau|x_tau) / pi_b(a_tau|x_tau)`` (the paper's Step-IS,
  clip 10);
- ``snips``: the self-normalised version, ``sum gamma^s w r / sum gamma^s w`` times the mean
  discount mass ``sum gamma^s`` per trajectory, so it equals ``behavior`` when ``pi = pi_b``;
- diagnostics: mean log-ratio, share of clipped weights and effective sample size of the
  weights (as a share of the actions).

The direct and doubly robust estimators of the paper need Monte-Carlo roll-outs of a feedback
model and are not computed. Policies are models (the reranker would need the RM at every logged
state). The weights are only as good as ``pi_b``, an estimate of the production recommender.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from recsys_rl import evaluate
from recsys_rl.config import Config
from recsys_rl.data import N_SPECIAL_TOKENS, PreparedData, prepare
from recsys_rl.model import GPT2Recommender, load_model, to_tensors
from recsys_rl.reward_model import HEAD_INDEX, feedback_targets
from recsys_rl.rl import logged_actions
from recsys_rl.train import amp_dtype
from recsys_rl.utils import get_device


@torch.no_grad()
def action_log_probs(
    model: GPT2Recommender, batch: dict[str, torch.Tensor], action: torch.Tensor, chunk: int = 2048
) -> torch.Tensor:
    """``[N]`` log-probability of the logged item at every action position (row-major); the
    full softmax is computed in chunks of states."""
    rows, cols = action.nonzero(as_tuple=True)
    hidden = model(batch)[rows, cols - 1]
    items = batch["item_idx"][rows, cols] - N_SPECIAL_TOKENS
    out = []
    for start in range(0, len(items), chunk):
        sl = slice(start, start + chunk)
        logp = F.log_softmax(model.scores(hidden[sl])[:, N_SPECIAL_TOKENS:].float(), -1)
        out.append(logp.gather(1, items[sl, None])[:, 0])
    return torch.cat(out)


def ope_terms(
    log_ratio: torch.Tensor, reward: torch.Tensor, trajectory: torch.Tensor, gamma: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(discount, cum_log_weight, start)`` ``[B, T]``: ``gamma^s``, ``log w_{0:s}`` of every
    action and the mask of trajectory starts (zeros outside actions)."""
    action = trajectory >= 0
    same_prev = torch.zeros_like(action)
    same_prev[:, 1:] = action[:, 1:] & (trajectory[:, 1:] == trajectory[:, :-1])
    step = torch.zeros_like(reward)
    cum = torch.zeros_like(reward)
    log_ratio = torch.where(action, log_ratio, 0)
    cum[:, 0] = log_ratio[:, 0]
    for t in range(1, trajectory.shape[1]):
        step[:, t] = torch.where(same_prev[:, t], step[:, t - 1] + 1, 0)
        cum[:, t] = log_ratio[:, t] + torch.where(same_prev[:, t], cum[:, t - 1], 0)
    discount = torch.where(action, gamma**step, 0)
    return discount, torch.where(action, cum, 0), action & ~same_prev


def ope_eval(
    cfg: Config,
    prepared: PreparedData | None = None,
    device: torch.device | None = None,
    split: str = "val",
    policies: Mapping[str, str | Path | None] | None = None,
    reward: str = "like_ratio",
    gamma: float = 0.9,
    clip: float = 10.0,
    batch_size: int = 64,
) -> pl.DataFrame:
    """One row per policy (name -> checkpoint, None = ``pi_ref``): the estimates and
    diagnostics of the module docstring, plus the behaviour model itself as ``pi_b``."""
    r = cfg.rl
    device = device or get_device()
    prepared = prepared or prepare(cfg)
    store, bounds = prepared.store.in_memory(), prepared.bounds
    runs = Path(cfg.data_root) / "runs"
    behavior = load_model(runs / r.behavior_run / "best.pt", device).eval()
    seq_len = behavior.backbone.config.n_positions
    dtype = amp_dtype(device, r.amp)
    es = evaluate.build_eval_set(store, bounds, split, seq_len, cfg.data.listen_plus_threshold)
    start, end = (bounds.val_start, bounds.val_end) if split == "val" else (bounds.test_start, None)
    cutoff = int(end) if end is not None else int(np.iinfo(np.int64).max)
    ends = store.cutoff_index(cutoff)[es.users]
    starts = np.maximum(es.history_starts, ends - seq_len)

    loaded = {"pi_b": behavior}  # the models are small enough to stay on the device together
    for name, path in (policies or {}).items():
        loaded[name] = load_model(path or runs / r.base_run / "best.pt", device).eval()
    keys = ("is", "sn_num", "sn_den", "log_ratio", "clipped", "w", "w2")
    sums = {name: dict.fromkeys(keys, 0.0) for name in loaded}
    totals = {"behavior": 0.0, "mass": 0.0, "n_traj": 0.0, "n_actions": 0.0}
    for b in range(0, es.n_users, batch_size):
        sl = slice(b, b + batch_size)
        raw = store.gather(starts[sl], ends[sl] - starts[sl], seq_len, cutoff)
        targets, _ = feedback_targets(raw, cutoff, cfg.data, cfg.rm.skip_threshold)
        batch = to_tensors(raw, device)
        action, trajectory = logged_actions(batch)
        in_period = batch["timestamp"] >= start
        action &= in_period
        trajectory = torch.where(action, trajectory, -1)
        if not action.any():
            continue
        t = torch.from_numpy(targets).to(device)
        rew = t[..., HEAD_INDEX["ratio"]]
        if reward == "like_ratio":
            rew = t[..., HEAD_INDEX["like"]] + rew / 10
        elif reward != "ratio":
            raise ValueError(f"unknown reward {reward!r}")
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            logp_b = action_log_probs(behavior, batch, action)
        zeros = torch.zeros_like(rew)
        discount, _, first = ope_terms(zeros, rew, trajectory, gamma)
        d, rr = discount[action], rew[action]
        totals["behavior"] += (d * rr).sum().item()
        totals["mass"] += d.sum().item()
        totals["n_traj"] += first.sum().item()
        totals["n_actions"] += float(action.sum().item())
        for name, model in loaded.items():
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                logp = logp_b if model is behavior else action_log_probs(model, batch, action)
            log_ratio = torch.zeros_like(rew)
            log_ratio[action] = logp - logp_b
            _, cum, _ = ope_terms(log_ratio, rew, trajectory, gamma)
            w = cum[action].exp()
            wc = w.clamp(max=clip)
            acc = sums[name]
            acc["is"] += (d * wc * rr).sum().item()
            acc["sn_num"] += (d * w * rr).sum().item()
            acc["sn_den"] += (d * w).sum().item()
            acc["log_ratio"] += log_ratio[action].sum().item()
            acc["clipped"] += (w > clip).sum().item()
            acc["w"] += w.sum().item()
            acc["w2"] += (w**2).sum().item()
    n_traj, n = max(totals["n_traj"], 1.0), max(totals["n_actions"], 1.0)
    rows = []
    for name, acc in sums.items():
        rows.append(
            {
                "split": split,
                "policy": name,
                "reward": reward,
                "behavior": totals["behavior"] / n_traj,
                "step_is": acc["is"] / n_traj,
                "snips": acc["sn_num"] / max(acc["sn_den"], 1e-12) * totals["mass"] / n_traj,
                "log_ratio": acc["log_ratio"] / n,
                "clipped_share": acc["clipped"] / n,
                "ess_share": acc["w"] ** 2 / max(acc["w2"], 1e-12) / n,
                "n_trajectories": int(totals["n_traj"]),
                "n_actions": int(totals["n_actions"]),
            }
        )
    return pl.DataFrame(rows)
