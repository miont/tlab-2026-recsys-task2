"""Base recommender training (stage 2).

Next-item prediction on the cached train windows (``data.prepare``) with sampled softmax:
``n_negatives`` items per step are shared by the whole batch and drawn from the train
popularity distribution ``q`` (logQ-corrected, see ``model.sampled_softmax_loss``). The model
is evaluated on the val day every ``eval_every`` steps with the full softmax; the best val
NDCG@10 is kept as ``runs/<run_name>/best.pt``, and ``last.pt`` holds the full training state
so an interrupted Colab session resumes where it stopped.
"""

from __future__ import annotations

import json
import math
import queue
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

from recsys_rl import evaluate
from recsys_rl.config import Config, save_config
from recsys_rl.data import N_SPECIAL_TOKENS, PreparedData, prepare
from recsys_rl.model import (
    GPT2Recommender,
    load_model,
    sampled_softmax_loss,
    save_model,
    to_tensors,
)
from recsys_rl.utils import get_device, load_checkpoint, save_checkpoint, set_seed


def amp_dtype(device: torch.device, enabled: bool) -> torch.dtype | None:
    """bf16 on Ampere+ GPUs, fp16 (with a grad scaler) on older ones, None off CUDA."""
    if not enabled or device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def proposal_log_q(items: pl.DataFrame, n_items: int) -> torch.Tensor:
    """``log q`` of the negative-sampling proposal: train listen counts, ``-inf`` for specials."""
    counts = torch.zeros(n_items, dtype=torch.float64)
    counts[torch.from_numpy(items["idx"].to_numpy().astype(np.int64))] = torch.from_numpy(
        items["count"].to_numpy().astype(np.float64)
    )
    return (counts / counts.sum()).log().float()


def build_model(cfg: Config, n_items: int) -> GPT2Recommender:
    return GPT2Recommender(cfg.model, n_items, cfg.data.max_seq_len)


def make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW; weight decay only on transformer matrices (not embeddings, biases, LayerNorms):
    decaying the item table every step would shrink rarely seen items."""
    embeddings = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.ndim >= 2 and id(p) not in embeddings else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


def lr_lambda(warmup: int, total: int):
    def f(step: int) -> float:
        warm = min(1.0, (step + 1) / max(1, warmup))
        return warm * 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, total))))

    return f


def _prefetch(it: Iterator, size: int = 4) -> Iterator:
    """Run ``it`` in a background thread (gathering from memmaps overlaps with GPU work)."""
    q: queue.Queue = queue.Queue(size)
    done = object()

    def worker():
        for x in it:
            q.put(x)
        q.put(done)

    threading.Thread(target=worker, daemon=True).start()
    while (x := q.get()) is not done:
        yield x


def next_item_loss(
    model: GPT2Recommender,
    batch: dict[str, torch.Tensor],
    log_q: torch.Tensor,
    n_negatives: int,
    target: str = "all",
    min_ratio: int = 50,
) -> torch.Tensor:
    """Sampled-softmax loss of predicting every listen of the windows from its prefix.

    Padding and OOV targets are skipped; ``target="listen_plus"`` also skips targets with
    ``played_ratio_pct < min_ratio`` and ``target="recommended"`` organic targets, which gives
    the behaviour model of the recommender (``rl`` off-policy branch); the skipped listens
    still feed the context.
    """
    hidden = model(batch)[:, :-1]
    targets = batch["item_idx"][:, 1:]
    valid = batch["mask"][:, 1:] & (targets >= N_SPECIAL_TOKENS)
    if target == "listen_plus":
        valid &= batch["played_ratio_pct"][:, 1:] >= min_ratio
    elif target == "recommended":
        valid &= batch["is_organic"][:, 1:] == 0
    elif target != "all":
        raise ValueError(f"unknown target {target!r}")
    negatives = torch.multinomial(log_q.exp(), n_negatives, replacement=True)
    return sampled_softmax_loss(
        hidden[valid], targets[valid], model.item_emb.weight, negatives, log_q
    )


def train(
    cfg: Config, prepared: PreparedData | None = None, device: torch.device | None = None
) -> Path:
    """Train (or resume) the base model of ``cfg.run_name``; returns the best checkpoint path."""
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    set_seed(cfg.seed)
    device = device or get_device()
    prepared = prepared or prepare(cfg)
    store, items, bounds = prepared.store.in_memory(), prepared.items, prepared.bounds
    n_items = N_SPECIAL_TOKENS + len(items)
    t = cfg.train
    min_ratio = round(cfg.data.listen_plus_threshold * 100)

    model = build_model(cfg, n_items).to(device)
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    if t.init_run and not last_path.exists():  # fine-tuning; a resumed run has its own weights
        init = load_model(Path(cfg.data_root) / "runs" / t.init_run / "best.pt")
        if init.n_items != n_items:
            raise ValueError(f"{t.init_run} has another vocabulary ({init.n_items} items)")
        model.load_state_dict(init.state_dict())
    optimizer = make_optimizer(model, t.lr, t.weight_decay)
    n_windows = len(prepared.window_starts)
    batches_per_epoch = n_windows // t.batch_size
    if batches_per_epoch == 0:
        raise ValueError(f"{n_windows} train windows < batch size {t.batch_size}")
    total_steps = t.epochs * batches_per_epoch
    if t.max_steps > 0:
        total_steps = min(total_steps, t.max_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(t.warmup_steps, total_steps))
    dtype = amp_dtype(device, t.amp)
    scaler = torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)
    log_q = proposal_log_q(items, n_items).to(device)

    state = {"step": 0, "epoch": 0, "batch": 0, "best": -1.0, "bad_evals": 0}
    if last_path.exists():
        ckpt = load_checkpoint(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        state = ckpt["state"]
        print(f"resumed from {last_path} at step {state['step']}")

    val = evaluate.build_eval_set(
        store, bounds, "val", cfg.data.max_seq_len, cfg.data.listen_plus_threshold
    )
    idx_to_item = evaluate.idx_to_item_array(items)
    log_path = run_dir / "log.jsonl"
    print(
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, device {device}, "
        f"amp {dtype}, {batches_per_epoch} steps/epoch, {total_steps} steps, "
        f"{val.n_users} val users"
    )

    def run_eval(train_loss: float) -> bool:
        """Evaluate on val, checkpoint; True if training should stop early."""
        recs = evaluate.recommend_model(model, store, val, idx_to_item, amp_dtype=dtype)
        val_metrics = evaluate.evaluate_recs(recs, val, len(items))
        improved = val_metrics["ndcg@10"] > state["best"]
        if improved:
            state["best"], state["bad_evals"] = val_metrics["ndcg@10"], 0
            save_model(best_path, model, step=state["step"], val=val_metrics)
        else:
            state["bad_evals"] += 1
        record = {"step": state["step"], "epoch": state["epoch"], "loss": train_loss}
        record |= {"lr": scheduler.get_last_lr()[0], "time": time.time()}
        record |= {f"val_{k}": v for k, v in val_metrics.items()}
        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(
            f"step {state['step']} loss {train_loss:.4f} val ndcg@10 "
            f"{val_metrics['ndcg@10']:.4f} recall@100 {val_metrics['recall@100']:.4f}"
            + (" *" if improved else "")
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
        return state["bad_evals"] >= t.patience

    def batches(epoch: int, first: int) -> Iterator[dict[str, np.ndarray]]:
        order = np.random.default_rng(cfg.seed + epoch).permutation(n_windows)
        for b in range(first, batches_per_epoch):
            sel = order[b * t.batch_size : (b + 1) * t.batch_size]
            yield store.gather(
                prepared.window_starts[sel],
                prepared.window_lengths[sel],
                cfg.data.max_seq_len,
                bounds.train_end,
            )

    model.train()
    losses: list[float] = []
    stop = state["step"] >= total_steps
    while not stop and state["epoch"] < t.epochs:
        for batch in _prefetch(batches(state["epoch"], state["batch"])):
            batch = to_tensors(batch, device)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                loss = next_item_loss(model, batch, log_q, t.n_negatives, t.target, min_ratio)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
            state["step"] += 1
            state["batch"] += 1
            if state["batch"] == batches_per_epoch:
                state["epoch"], state["batch"] = state["epoch"] + 1, 0
            if state["step"] % t.eval_every == 0 or state["step"] >= total_steps:
                stop = run_eval(float(np.mean(losses))) or state["step"] >= total_steps
                losses = []
            if stop:
                break
    if not best_path.exists():  # resumed after the last evaluation of a finished run
        run_eval(float(np.mean(losses)) if losses else float("nan"))
    return best_path
