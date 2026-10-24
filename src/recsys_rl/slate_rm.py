"""List-level (slate) reward model on runs of logged recommendations (level 2 of the slate study).

The item-level RM (``reward_model.RewardModel``) scores one candidate from the context state;
in a list every candidate is scored as if it were the only one. In Yambda recommendations come
as a stream (the run of consecutive ``is_organic = 0`` listens of a session), so the logs hold
real lists with the user's response to every position. A *slate* here is a chunk of at most
``slate_size`` consecutive recommendations of one run (``logged_slates``); its context is every
listen before its first track, with feedback.

Model (``SlateRewardModel``): the response to position ``j`` is predicted from the context state
``h`` (the item-level RM's encoder) and the tracks at positions ``1..j`` *without* the feedback
of positions ``< j`` — the information available when the whole list is chosen at once:

    logits_j = item_heads(h, c_j) + delta(z_j),   z = causal transformer([h, c_1, ..., c_L])

``c_j`` is the item-level RM's candidate embedding (item, length bin, ``is_organic = 0``, repeat
count in the context) and ``item_heads`` its ensemble of heads, so with the zero-initialised
``delta`` the model starts exactly as the item-level RM applied to every position; the
transformer learns only what the list adds (position, earlier tracks of the list: fatigue,
artist repeats, transitions). By default the item-level RM is frozen
(``slate_rm.freeze_rm``). ``delta`` has one output per ensemble member and the slates get
Poisson(1) bootstrap weights per member, so the ensemble spread stays meaningful.

The list reward (``list_reward``) is the mean expected played ratio over the positions: R1 of
the item-level study, now with every position conditioned on the list before it.

Limits: the logged runs were chosen adaptively by the production recommender (a skip changes
the next track), so the model learns the response to the lists that were shown, not to an
arbitrary list; lists far from the logs are extrapolated (as for the item-level RM, measured by
the ensemble spread). Proxy and gold slate RMs extend the proxy and gold item-level RMs and use
different seeds / user subsamples (``slate_rm.user_fraction``).
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy.stats import spearmanr
from torch import nn
from transformers import GPT2Config, GPT2Model

from recsys_rl import metrics
from recsys_rl.config import Config, DataConfig, SlateRMConfig, save_config
from recsys_rl.data import N_SPECIAL_TOKENS, PAD_IDX, EventStore, PreparedData, prepare
from recsys_rl.model import to_tensors
from recsys_rl.reward_model import (
    HEAD_INDEX,
    HEADS,
    RewardModel,
    context_repeat_counts,
    feedback_loss,
    feedback_targets,
    load_reward_model,
)
from recsys_rl.train import amp_dtype, lr_lambda, make_optimizer
from recsys_rl.utils import get_device, load_checkpoint, save_checkpoint, set_seed


def finite(record: Mapping) -> dict:
    """``record`` with non-finite floats as None (``NaN`` is not valid JSON for polars)."""
    return {
        k: None if isinstance(v, float) and not np.isfinite(v) else v for k, v in record.items()
    }


def logged_slates(
    store: EventStore, start: int, end: int, slate_size: int, min_len: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """``(rows, lengths)`` of the logged slates with ``start <= timestamp < end``: runs of
    consecutive recommendations (non-organic, known item) within one session of one user, cut
    into chunks of ``slate_size`` from the start of the run; chunks shorter than ``min_len``
    and chunks without an earlier listen of the user (no context) are dropped."""
    cols = store.columns
    n = len(cols["timestamp"])
    ts = np.asarray(cols["timestamp"])
    user = np.repeat(np.arange(store.n_users), np.diff(store.offsets))
    session = np.asarray(cols["session_id"])
    action = (
        (np.asarray(cols["is_organic"]) == 0)
        & (np.asarray(cols["item_idx"]) >= N_SPECIAL_TOKENS)
        & (ts >= start)
        & (ts < end)
    )
    chained = np.zeros(n, dtype=bool)
    chained[1:] = action[1:] & action[:-1] & (session[1:] == session[:-1]) & (user[1:] == user[:-1])
    run_start = action & ~chained
    rows = np.arange(n)
    first_row = np.maximum.accumulate(np.where(run_start, rows, 0))  # start of the current run
    pos = rows - first_row
    run_id = np.cumsum(run_start) - 1
    run_len = np.bincount(run_id[action], minlength=max(int(run_start.sum()), 1))
    chunk = np.flatnonzero(action & (pos % slate_size == 0))
    lengths = np.minimum(slate_size, run_len[run_id[chunk]] - pos[chunk])
    keep = (lengths >= min_len) & (chunk > store.offsets[user[chunk]])
    return chunk[keep].astype(np.int64), lengths[keep].astype(np.int64)


def slate_users(store: EventStore, rows: np.ndarray) -> np.ndarray:
    return np.searchsorted(store.offsets, rows, side="right") - 1


def slate_batch(
    store: EventStore,
    rows: np.ndarray,
    lengths: np.ndarray,
    seq_len: int,
    cutoff: int,
    slate_size: int,
    data_cfg: DataConfig,
    skip_threshold: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(context, slates, mask, targets, valid)`` of logged slates: the context batch
    (``EventStore.gather``: the last ``seq_len`` listens before the slate, leak-free with
    respect to ``cutoff``), slate items ``[B, L]`` (``PAD_IDX`` after the end), their mask and
    the feedback labels ``[B, L, H]`` of ``feedback_targets``."""
    lo = store.offsets[slate_users(store, rows)]
    ctx_starts = np.maximum(lo, rows - seq_len)
    context = store.gather(ctx_starts, rows - ctx_starts, seq_len, cutoff)
    pos = np.arange(slate_size)
    mask = pos[None, :] < lengths[:, None]
    idx = np.where(mask, rows[:, None] + pos[None, :], 0)
    names = ("item_idx", "played_ratio_pct", "timestamp", "next_gap", "like_delay", "dislike_delay")
    cols = {name: np.where(mask, np.asarray(store.columns[name])[idx], 0) for name in names}
    cols["mask"] = mask
    targets, valid = feedback_targets(cols, cutoff, data_cfg, skip_threshold)
    slates = np.where(mask, cols["item_idx"], PAD_IDX).astype(np.int64)
    return context, slates, mask, targets, valid


class SlateRewardModel(nn.Module):
    def __init__(self, rm: RewardModel, cfg: SlateRMConfig):
        super().__init__()
        self.rm = rm
        self.cfg = cfg
        d = rm.encoder.cfg.n_embd
        self.slate_size = cfg.slate_size
        self.state_proj = nn.Linear(d, d)
        self.backbone = GPT2Model(
            GPT2Config(
                vocab_size=1,
                bos_token_id=None,
                eos_token_id=None,
                n_positions=cfg.slate_size + 1,
                n_embd=d,
                n_layer=cfg.n_layer,
                n_head=cfg.n_head,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
            )
        )
        self.delta = nn.Linear(d, rm.cfg.n_ensemble * len(HEADS))
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        if cfg.freeze_rm:
            self.rm.requires_grad_(False)

    @property
    def max_seq_len(self) -> int:
        return self.rm.max_seq_len

    def context_state(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """``[B, d]`` item-level RM state after the context."""
        if self.cfg.freeze_rm:
            with torch.no_grad():
                return self.rm.last_hidden(batch)
        return self.rm.last_hidden(batch)

    def logits(
        self, hidden: torch.Tensor, slates: torch.Tensor, repeat_count: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(logits, item_logits)`` ``[E, B, L, H]`` of the slates ``[B, L]`` (vocabulary
        indices, ``PAD_IDX`` after the end) after the context states ``hidden`` ``[B, d]``;
        ``item_logits`` is the item-level RM alone (``delta = 0``)."""
        b, length = slates.shape
        c = self.rm.candidate_embedding(slates, torch.zeros_like(slates), repeat_count)
        item = self.rm.heads_on(hidden[:, None], c)
        x = torch.cat([self.state_proj(hidden)[:, None], c], 1)
        z = self.backbone(inputs_embeds=x).last_hidden_state[:, 1:]
        delta = self.delta(z).view(b, length, len(item), len(HEADS)).permute(2, 0, 1, 3)
        return item + delta, item

    def score(self, batch: Mapping[str, torch.Tensor], slates: torch.Tensor) -> torch.Tensor:
        """``[E, B, L, H]`` predicted probabilities of the slates after each context."""
        hidden = self.context_state(batch)
        repeat = context_repeat_counts(batch["item_idx"], batch["mask"], slates)
        return torch.sigmoid(self.logits(hidden, slates, repeat)[0].float())

    def position_scores(
        self,
        hidden: torch.Tensor,
        prefix: torch.Tensor,
        candidates: torch.Tensor,
        prefix_repeat: torch.Tensor,
        cand_repeat: torch.Tensor,
    ) -> torch.Tensor:
        """``[E, B, M, H]`` probabilities of the ``M`` candidates ``[B, M]`` placed after the
        ``prefix`` ``[B, j]`` of each list (for greedy list construction)."""
        b, m = candidates.shape
        j = prefix.shape[1]
        slates = torch.cat([prefix[:, None].expand(b, m, j), candidates[..., None]], -1)
        repeat = torch.cat([prefix_repeat[:, None].expand(b, m, j), cand_repeat[..., None]], -1)
        h = hidden[:, None].expand(b, m, hidden.shape[-1]).reshape(b * m, -1)
        logits = self.logits(h, slates.reshape(b * m, j + 1), repeat.reshape(b * m, j + 1))[0]
        return torch.sigmoid(logits[:, :, -1].float()).view(len(logits), b, m, len(HEADS))


def list_reward(probs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """``[B]`` mean over the (masked) positions of the ensemble-mean expected played ratio;
    ``probs`` is ``[E, B, L, H]``."""
    ratio = probs.mean(0)[..., HEAD_INDEX["ratio"]]
    if mask is None:
        return ratio.mean(-1)
    m = mask.float()
    return (ratio * m).sum(-1) / m.sum(-1).clamp_min(1)


def save_slate_rm(path: Path, model: SlateRewardModel, **extra) -> None:
    rm = model.rm
    save_checkpoint(
        path,
        model=model.state_dict(),
        slate_cfg=asdict(model.cfg),
        rm_cfg=asdict(rm.cfg),
        model_cfg=asdict(rm.encoder.cfg),
        n_items=rm.encoder.n_items,
        max_seq_len=rm.max_seq_len,
        audio_dim=None if rm.audio_proj is None else rm.audio_proj.in_features,
        **extra,
    )


def load_slate_rm(path: str | Path, device: torch.device | str = "cpu") -> SlateRewardModel:
    from recsys_rl.config import ModelConfig, RewardModelConfig

    ckpt = load_checkpoint(path)
    n_items = ckpt["n_items"]
    audio = None if ckpt["audio_dim"] is None else torch.zeros(n_items, ckpt["audio_dim"])
    rm = RewardModel(
        RewardModelConfig(**ckpt["rm_cfg"]),
        ModelConfig(**ckpt["model_cfg"]),
        n_items,
        ckpt["max_seq_len"],
        torch.zeros(n_items, dtype=torch.long),
        audio,
    )
    model = SlateRewardModel(rm, SlateRMConfig(**ckpt["slate_cfg"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device)


def load_frozen_slate_rm(cfg: Config, run: str, device: torch.device) -> SlateRewardModel:
    model = load_slate_rm(Path(cfg.data_root) / "runs" / run / "best.pt", device)
    return model.eval().requires_grad_(False)


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    return metrics.roc_auc(y, p) if 0 < y.mean() < 1 else float("nan")


@torch.no_grad()
def slate_rm_eval(
    model: SlateRewardModel,
    store: EventStore,
    rows: np.ndarray,
    lengths: np.ndarray,
    cutoff: int,
    data_cfg: DataConfig,
    skip_threshold: float,
    batch_size: int = 256,
    amp: torch.dtype | None = None,
) -> tuple[dict[str, float], pl.DataFrame]:
    """``(summary, by_position)`` on logged slates for the slate RM (``slate_*``) and the
    item-level RM it extends (``item_*``, the same context state, every position scored alone):
    BCE ``loss`` over the heads, ratio MAE / Spearman, Listen+ and skip AUC, the Spearman of the
    list reward with the real mean ratio of the slate; ``by_position`` has the ratio MAE and
    Listen+ AUC of both per position."""
    model.eval()
    device = next(model.parameters()).device
    s = model.cfg.slate_size
    out: dict[str, list[np.ndarray]] = {k: [] for k in ("slate", "item", "y", "valid", "pos")}
    lists: dict[str, list[np.ndarray]] = {"slate": [], "item": [], "y": []}
    losses = {"slate": [], "item": []}
    for b in range(0, len(rows), batch_size):
        sl = slice(b, b + batch_size)
        ctx, slates, mask, targets, valid = slate_batch(
            store, rows[sl], lengths[sl], model.max_seq_len, cutoff, s, data_cfg, skip_threshold
        )
        batch = to_tensors(ctx, device)
        slates_t = torch.from_numpy(slates).to(device)
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            hidden = model.context_state(batch)
            repeat = context_repeat_counts(batch["item_idx"], batch["mask"], slates_t)
            full, item = model.logits(hidden, slates_t, repeat)
        t, v = torch.from_numpy(targets).to(device), torch.from_numpy(valid).to(device)
        m = torch.from_numpy(mask).to(device)
        for name, logits in (("slate", full), ("item", item)):
            losses[name].append(feedback_loss(logits[:, m], t[m], v[m])[0].item() * m.sum().item())
            probs = torch.sigmoid(logits.float()).mean(0)
            out[name].append(probs[m].cpu().numpy())
            lists[name].append(list_reward(torch.sigmoid(logits.float()), m).cpu().numpy())
        out["y"].append(targets[mask])
        out["valid"].append(valid[mask])
        out["pos"].append(np.broadcast_to(np.arange(s), mask.shape)[mask])
        ratio = targets[..., HEAD_INDEX["ratio"]]
        lists["y"].append((ratio * mask).sum(1) / mask.sum(1))
    cat = {k: np.concatenate(v) for k, v in out.items()}
    y, valid, pos = cat["y"], cat["valid"], cat["pos"]
    n = len(y)
    summary: dict[str, float] = {"n_slates": float(len(rows)), "n_listens": float(n)}
    r, lp, sk = HEAD_INDEX["ratio"], HEAD_INDEX["listen_plus"], HEAD_INDEX["skip"]
    y_list = np.concatenate(lists["y"])
    for name in ("slate", "item"):
        p = cat[name]
        summary[f"{name}_loss"] = float(np.sum(losses[name]) / n)
        summary[f"{name}_ratio_mae"] = float(np.abs(p[:, r] - y[:, r]).mean())
        summary[f"{name}_ratio_spearman"] = float(spearmanr(p[:, r], y[:, r]).statistic)
        summary[f"{name}_listen_plus_auc"] = _auc(y[:, lp], p[:, lp])
        summary[f"{name}_skip_auc"] = _auc(y[:, sk], p[:, sk])
        c = HEAD_INDEX["continue"]
        ok = valid[:, c]
        summary[f"{name}_continue_auc"] = _auc(y[ok, c], p[ok, c])
        summary[f"{name}_list_spearman"] = float(
            spearmanr(np.concatenate(lists[name]), y_list).statistic
        )
    rows_out = []
    for j in range(s):
        at = pos == j
        if at.sum() < 2:
            continue
        row = {"position": j + 1, "n": int(at.sum()), "mean_ratio": float(y[at, r].mean())}
        for name in ("slate", "item"):
            p = cat[name][at]
            row[f"{name}_ratio_mae"] = float(np.abs(p[:, r] - y[at, r]).mean())
            row[f"{name}_listen_plus_auc"] = _auc(y[at, lp], p[:, lp])
        rows_out.append(row)
    return summary, pl.DataFrame(rows_out)


def val_slates(
    store: EventStore, start: int, end: int, train_end: int, cfg: SlateRMConfig, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Logged slates of an evaluation period by users with train history (a seeded subset of
    at most ``cfg.max_val_slates``)."""
    rows, lengths = logged_slates(store, start, end, cfg.slate_size, cfg.min_len)
    users = slate_users(store, rows)
    has_train = store.cutoff_index(train_end) > store.offsets[:-1]
    keep = has_train[users]
    rows, lengths = rows[keep], lengths[keep]
    if 0 < cfg.max_val_slates < len(rows):
        sel = np.sort(np.random.default_rng(seed).choice(len(rows), cfg.max_val_slates, False))
        rows, lengths = rows[sel], lengths[sel]
    return rows, lengths


def train_slate_rm(
    cfg: Config, prepared: PreparedData | None = None, device: torch.device | None = None
) -> Path:
    """Train (or resume) the slate RM of ``cfg.run_name`` on top of ``runs/<slate_rm.base_rm>``;
    returns the best checkpoint (lowest val loss of the slate RM)."""
    c = cfg.slate_rm
    run_dir = cfg.run_dir
    last_path, best_path, log_path = run_dir / "last.pt", run_dir / "best.pt", run_dir / "log.jsonl"
    if best_path.exists() and last_path.exists():
        done = load_checkpoint(last_path)["state"]
        if done.get("done") or done["step"] >= c.steps:  # finished or stopped early
            print(f"{cfg.run_name} is done at step {done['step']}")
            return best_path
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    set_seed(cfg.seed)
    device = device or get_device()
    prepared = prepared or prepare(cfg)
    store, bounds = prepared.store.in_memory(), prepared.bounds
    rm = load_reward_model(Path(cfg.data_root) / "runs" / c.base_rm / "best.pt", device)
    model = SlateRewardModel(rm, c).to(device)
    if c.freeze_rm:
        model.rm.eval()
    seq_len = model.max_seq_len
    skip = cfg.rm.skip_threshold

    rows, lengths = logged_slates(store, 0, bounds.train_end, c.slate_size, c.min_len)
    rng = np.random.default_rng(cfg.seed)
    if c.user_fraction < 1.0:
        users = slate_users(store, rows)
        unique = np.unique(users)
        chosen = rng.choice(unique, max(1, round(c.user_fraction * len(unique))), replace=False)
        keep = np.isin(users, chosen)
        rows, lengths = rows[keep], lengths[keep]
    n_ens = rm.cfg.n_ensemble
    weights = (
        rng.poisson(1.0, size=(len(rows), n_ens)).astype(np.float32)
        if c.bootstrap
        else np.ones((len(rows), n_ens), dtype=np.float32)
    )
    v_rows, v_lengths = val_slates(
        store, bounds.val_start, bounds.val_end, bounds.train_end, c, cfg.seed
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(model, c.lr, c.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(c.warmup_steps, c.steps))
    dtype = amp_dtype(device, c.amp)
    scaler = torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)
    state = {"step": 0, "best": float("inf"), "bad_evals": 0, "done": False}
    if last_path.exists():
        ckpt = load_checkpoint(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        state = ckpt["state"]
        print(f"resumed from {last_path} at step {state['step']}")
    print(
        f"{sum(p.numel() for p in params) / 1e6:.2f}M trainable params, device {device}, "
        f"{len(rows)} train slates (mean length {lengths.mean():.1f}), {len(v_rows)} val slates"
    )

    def run_eval(train_loss: float) -> bool:
        summary, _ = slate_rm_eval(
            model, store, v_rows, v_lengths, bounds.val_end, cfg.data, skip, amp=dtype
        )
        improved = summary["slate_loss"] < state["best"]
        if improved:
            state["best"], state["bad_evals"] = summary["slate_loss"], 0
            save_slate_rm(best_path, model, step=state["step"], val=summary)
        else:
            state["bad_evals"] += 1
        state["done"] = state["bad_evals"] >= c.patience or state["step"] >= c.steps
        record = {"step": state["step"], "loss": train_loss, "time": time.time()}
        record |= {f"val_{k}": v for k, v in summary.items()}
        with log_path.open("a") as f:
            f.write(json.dumps(finite(record)) + "\n")
        print(
            f"step {state['step']} loss {train_loss:.4f} val loss {summary['slate_loss']:.4f} "
            f"(item {summary['item_loss']:.4f}) list spearman {summary['slate_list_spearman']:.4f} "
            f"(item {summary['item_list_spearman']:.4f})" + (" *" if improved else "")
        )
        save_checkpoint(
            last_path,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),
            state=state,
        )
        model.train()
        if c.freeze_rm:
            model.rm.eval()
        return state["done"]

    if state["step"] == 0 and not log_path.exists():
        run_eval(float("nan"))  # the item-level RM itself (delta = 0)
    model.train()
    if c.freeze_rm:
        model.rm.eval()
    losses: list[float] = []
    while state["step"] < c.steps:
        sel = np.random.default_rng([cfg.seed, state["step"]]).integers(0, len(rows), c.batch_size)
        ctx, slates, mask, targets, valid = slate_batch(
            store, rows[sel], lengths[sel], seq_len, bounds.train_end, c.slate_size, cfg.data, skip
        )
        batch = to_tensors(ctx, device)
        slates_t = torch.from_numpy(slates).to(device)
        m = torch.from_numpy(mask).to(device)
        w = torch.from_numpy(weights[sel]).to(device)  # [B, E]
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            hidden = model.context_state(batch)
            repeat = context_repeat_counts(batch["item_idx"], batch["mask"], slates_t)
            logits, _ = model.logits(hidden, slates_t, repeat)
        w = w.T[:, :, None].expand(-1, -1, m.shape[1])[:, m]
        t, v = torch.from_numpy(targets).to(device), torch.from_numpy(valid).to(device)
        loss, _ = feedback_loss(logits[:, m], t[m], v[m], w)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(params, c.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        losses.append(loss.item())
        state["step"] += 1
        if state["step"] % c.eval_every == 0 or state["step"] >= c.steps:
            stop = run_eval(float(np.mean(losses)))
            losses = []
            if stop:
                break
    return best_path
