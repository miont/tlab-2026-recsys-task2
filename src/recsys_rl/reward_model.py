"""User feedback (reward) model (stage 3).

Predicts the user's response to a candidate track given the listening history:

====================  ==========================================================  ======
head                  target of listen ``t``                                      loss
====================  ==========================================================  ======
``ratio``             ``min(played_ratio_pct, 100) / 100``                        soft BCE
``listen_plus``       ``played_ratio_pct >= listen_plus_threshold * 100``          BCE
``skip``              ``played_ratio_pct < skip_threshold * 100``                  BCE
``like``              a like attributed to the listen (``data.attach_feedback``)   BCE
``dislike``           the same for dislikes                                       BCE
``continue``          the user's next listen starts within ``session_gap_minutes``  BCE
====================  ==========================================================  ======

Labels are restricted to what is observed before the period ``cutoff`` (a like given after
``train_end`` is not a train label; ``continue`` is unknown if ``t + gap >= cutoff``). BCE is
unweighted so that the probabilities stay calibrated (they are used as rewards).

Architecture (late fusion). The encoder is a ``GPT2Recommender`` initialised from the base
model, fed with the same inputs (item + feedback of every listen, leak-free like flags from
``EventStore.gather``). Its hidden state ``h_{t-1}`` summarises listens ``< t`` with their
feedback; the candidate ``item_t`` is represented by its (shared) item embedding plus
embeddings of features known at recommendation time: track-length bin, ``is_organic`` (0 =
recommendation, the RL setting) and how many times the item occurs in the context (repeats are
the strongest feedback signal in the EDA). Each ensemble member is an MLP on
``[h, c, h * c]``. Compared with feeding the candidate as an extra token (plan §5.2) this scores
``G`` candidates of a context with a single transformer pass, which RL (G samples per context,
reranking of the top-100, catalogue-wide analysis) needs; the price is that the candidate does
not attend to the history directly.

Training is teacher forcing over the cached train windows: one pass predicts the feedback of
every listen ``t >= 1`` of the window from ``h_{t-1}``, so the prediction for listen ``t`` never
sees its own feedback or anything later. The model can only learn the response to items that
were actually played; for never-played candidates it extrapolates (the main limitation, its
size is measured by the ensemble spread).

Uncertainty and over-optimisation: ``n_ensemble`` heads on a shared encoder, each trained with
its own Poisson(1) bootstrap weights per window. The proxy RM (used by RL) and the gold RM (used
only for evaluation) are separate runs that differ in seed, train-user subsample and features
(gold: random init + audio embeddings), see ``notebooks/03_reward_model.ipynb``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch import nn

from recsys_rl import metrics
from recsys_rl.config import Config, DataConfig, ModelConfig, RewardModelConfig, save_config
from recsys_rl.data import N_SPECIAL_TOKENS, EventStore, PreparedData, SplitBounds, prepare
from recsys_rl.model import GPT2Recommender, load_model, to_tensors
from recsys_rl.train import _prefetch, amp_dtype, lr_lambda, make_optimizer
from recsys_rl.utils import get_device, load_checkpoint, save_checkpoint, set_seed

HEADS = ("ratio", "listen_plus", "skip", "like", "dislike", "continue")
HEAD_INDEX = {name: i for i, name in enumerate(HEADS)}
# Bins of "occurrences of the candidate among earlier context listens": 0, 1, 2, 3-4, 5-9, 10+.
REPEAT_BIN_EDGES = (0, 1, 2, 4, 9)
N_REPEAT_BINS = len(REPEAT_BIN_EDGES) + 1


def feedback_targets(
    cols: Mapping[str, np.ndarray], cutoff: int, data_cfg: DataConfig, skip_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(targets, valid)`` of shape ``[..., len(HEADS)]`` for listens given as event-store
    columns (a gathered batch or store rows). OOV / padding listens are invalid for every head.
    """
    ratio = np.asarray(cols["played_ratio_pct"]).astype(np.float32)
    ts = np.asarray(cols["timestamp"]).astype(np.int64)
    next_gap = np.asarray(cols["next_gap"]).astype(np.int64)
    gap = data_cfg.session_gap_minutes * 60

    def given(name: str) -> np.ndarray:
        delay = np.asarray(cols[f"{name}_delay"]).astype(np.int64)
        return (delay >= 0) & (ts + delay < cutoff)

    targets = np.stack(
        [
            np.minimum(ratio, 100) / 100,
            ratio >= round(data_cfg.listen_plus_threshold * 100),
            ratio < round(skip_threshold * 100),
            given("like"),
            given("dislike"),
            (next_gap >= 0) & (next_gap <= gap),
        ],
        axis=-1,
    ).astype(np.float32)
    known = np.asarray(cols["item_idx"]) >= N_SPECIAL_TOKENS
    if "mask" in cols:
        known = known & np.asarray(cols["mask"])
    valid = np.repeat(known[..., None], len(HEADS), axis=-1)
    valid[..., HEAD_INDEX["continue"]] &= ts + gap < cutoff
    return targets, valid


def repeat_bins(counts: torch.Tensor) -> torch.Tensor:
    edges = torch.tensor(REPEAT_BIN_EDGES, device=counts.device)
    return (counts[..., None] > edges).sum(-1)


def prefix_repeat_counts(item_idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``[B, T]``: occurrences of the item at ``t`` among positions ``< t`` of its row."""
    t = item_idx.shape[1]
    earlier = torch.ones(t, t, dtype=torch.bool, device=item_idx.device).tril(-1)
    same = item_idx[:, :, None] == item_idx[:, None, :]
    return (same & earlier & mask[:, None, :]).sum(-1)


def context_repeat_counts(
    item_idx: torch.Tensor, mask: torch.Tensor, candidates: torch.Tensor
) -> torch.Tensor:
    """``[B, G]``: occurrences of each candidate among the context listens."""
    return ((candidates[:, :, None] == item_idx[:, None, :]) & mask[:, None, :]).sum(-1)


def item_length_bins(items: pl.DataFrame, data_cfg: DataConfig) -> torch.Tensor:
    """``[n_items]`` length bin of every vocabulary item (median train length), 0 for specials."""
    out = torch.zeros(N_SPECIAL_TOKENS + len(items), dtype=torch.long)
    lengths = items["track_length"].fill_null(0).to_numpy().astype(np.float64)
    bins = (lengths[:, None] > np.asarray(data_cfg.length_bin_edges)[None, :]).sum(1)
    out[torch.from_numpy(items["idx"].to_numpy().astype(np.int64))] = torch.from_numpy(bins)
    return out


def load_audio_embeddings(cfg: Config, items: pl.DataFrame) -> np.ndarray:
    """``[n_items, d]`` float16 Yambda audio embeddings by vocabulary index (zero rows for
    specials and items without an embedding), cached under ``processed_dir``."""
    cache = cfg.processed_dir / "audio_embeddings.npy"
    if cache.exists():
        return np.load(cache)
    lf = pl.scan_parquet(cfg.raw_dir / "embeddings.parquet")
    col = "normalized_embed" if "normalized_embed" in lf.collect_schema().names() else "embed"
    vocab_ids = items["item_id"].cast(pl.Int64).to_list()
    found = (
        lf.select(pl.col("item_id").cast(pl.Int64), col)
        # filtered while scanning: the file has all 7.7M tracks, the vocabulary ~200k
        .filter(pl.col("item_id").is_in(vocab_ids))
        .join(
            items.lazy().select(pl.col("item_id").cast(pl.Int64), "idx"),
            on="item_id",
            how="inner",
        )
        .collect()
    )
    if found.height == 0:
        raise ValueError("no vocabulary item has an audio embedding")
    flat = found[col].explode().to_numpy().astype(np.float32)
    emb = flat.reshape(found.height, -1)
    out = np.zeros((N_SPECIAL_TOKENS + len(items), emb.shape[1]), dtype=np.float16)
    out[found["idx"].to_numpy()] = emb
    print(f"audio embeddings for {found.height}/{len(items)} vocabulary items")
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp.npy")
    np.save(tmp, out)
    tmp.replace(cache)
    return out


class RewardModel(nn.Module):
    def __init__(
        self,
        cfg: RewardModelConfig,
        model_cfg: ModelConfig,
        n_items: int,
        max_seq_len: int,
        item_length_bin: torch.Tensor,
        audio: torch.Tensor | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.encoder = GPT2Recommender(model_cfg, n_items, max_seq_len)
        d = model_cfg.n_embd
        self.register_buffer("item_length_bin", item_length_bin.long())
        self.audio_proj: nn.Linear | None = None
        if audio is not None:
            self.register_buffer("audio", audio.half())
            self.audio_proj = nn.Linear(audio.shape[1], d)
        self.cand_emb = nn.ModuleDict(
            {
                "length_bin": nn.Embedding(model_cfg.n_length_bins, d),
                "is_organic": nn.Embedding(2, d),
                "repeat_bin": nn.Embedding(N_REPEAT_BINS, d),
            }
        )
        for emb in self.cand_emb.values():
            nn.init.normal_(emb.weight, std=0.02)
        self.cand_norm = nn.LayerNorm(d)
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(3 * d, cfg.head_hidden), nn.GELU(), nn.Linear(cfg.head_hidden, len(HEADS))
            )
            for _ in range(cfg.n_ensemble)
        )

    @property
    def max_seq_len(self) -> int:
        return self.encoder.backbone.config.n_positions

    def _audio(self, idx: torch.Tensor) -> torch.Tensor:
        return self.audio_proj(self.audio[idx].float())

    def encode(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Hidden states ``[B, T, d]``; position ``t`` summarises listens ``<= t``."""
        extra = self._audio(batch["item_idx"]) if self.audio_proj is not None else None
        return self.encoder(batch, extra)

    def head_logits(
        self,
        hidden: torch.Tensor,
        candidates: torch.Tensor,
        is_organic: torch.Tensor,
        repeat_count: torch.Tensor,
        length_bin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``[E, *candidates.shape, len(HEADS)]`` logits; ``hidden`` (``[..., d]``) broadcasts
        against the candidates. ``length_bin`` defaults to the item's own bin."""
        c = self.encoder.item_emb(candidates)
        if self.audio_proj is not None:
            c = c + self._audio(candidates)
        if length_bin is None:
            length_bin = self.item_length_bin[candidates]
        c = c + self.cand_emb["length_bin"](length_bin)
        c = c + self.cand_emb["is_organic"](is_organic)
        c = c + self.cand_emb["repeat_bin"](repeat_bins(repeat_count))
        c = self.cand_norm(c)
        h = hidden.expand_as(c)
        x = torch.cat([h, c, h * c], dim=-1)
        return torch.stack([head(x) for head in self.heads])

    def teacher_forcing(
        self, batch: Mapping[str, torch.Tensor], overrides: Mapping[str, int] | None = None
    ) -> torch.Tensor:
        """``[E, B, T - 1, H]`` logits for the listens at positions ``1..T-1`` of the windows.

        ``overrides`` replaces candidate features by a constant (``length_bin``, ``is_organic``,
        ``repeat_count``) for the sensitivity analysis."""
        hidden = self.encode(batch)[:, :-1]
        items = batch["item_idx"][:, 1:]
        feats = {
            "is_organic": batch["is_organic"][:, 1:],
            "repeat_count": prefix_repeat_counts(batch["item_idx"], batch["mask"])[:, 1:],
            "length_bin": None,
        }
        for name, value in (overrides or {}).items():
            if name not in feats:
                raise KeyError(f"unknown candidate feature {name!r}")
            feats[name] = torch.full_like(items, value)
        return self.head_logits(hidden, items, **feats)

    def last_hidden(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """``[B, d]`` state after the last listen of each context."""
        hidden = self.encode(batch)
        last = batch["mask"].sum(1) - 1
        return hidden[torch.arange(len(hidden), device=hidden.device), last]

    def score(
        self, batch: Mapping[str, torch.Tensor], candidates: torch.Tensor, is_organic: int = 0
    ) -> torch.Tensor:
        """``[E, B, G, H]`` predicted probabilities for ``candidates`` (``[B, G]`` vocabulary
        indices) recommended after each context; ``ratio`` is the expected played ratio."""
        hidden = self.last_hidden(batch)[:, None]
        repeat = context_repeat_counts(batch["item_idx"], batch["mask"], candidates)
        organic = torch.full_like(candidates, is_organic)
        return torch.sigmoid(self.head_logits(hidden, candidates, organic, repeat).float())


def feedback_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(total, per_head)``: BCE per head, averaged over valid examples (weighted by the
    bootstrap ``weights`` ``[E, N]``) and over ensemble members, summed over heads.

    ``logits`` is ``[E, N, H]``, ``targets`` and ``valid`` ``[N, H]``."""
    bce = F.binary_cross_entropy_with_logits(
        logits.float(), targets.expand_as(logits), reduction="none"
    )
    w = valid.float()[None]
    if weights is not None:
        w = w * weights[..., None]
    per_head = ((bce * w).sum(1) / w.sum(1).clamp(min=1e-6)).mean(0)
    return per_head.sum(), per_head


def feedback_metrics(pred: np.ndarray, targets: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    """Quality of ``[N, H]`` predicted probabilities: ratio MAE / RMSE / Spearman, ROC-AUC /
    PR-AUC / ECE for the binary heads, per-head log loss and their sum ``loss``."""
    out: dict[str, float] = {}
    total = 0.0
    for name, h in HEAD_INDEX.items():
        v = valid[:, h]
        y, p = (
            targets[v, h].astype(np.float64),
            np.clip(pred[v, h].astype(np.float64), 1e-6, 1 - 1e-6),
        )
        if len(y) == 0:
            continue
        bce = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        out[f"{name}_logloss"] = bce
        total += bce
        out[f"{name}_mean_true"] = float(y.mean())
        out[f"{name}_mean_pred"] = float(p.mean())
        if name == "ratio":
            out["ratio_mae"] = float(np.abs(p - y).mean())
            out["ratio_rmse"] = float(np.sqrt(((p - y) ** 2).mean()))
            varies = len(y) > 1 and np.ptp(p) > 0 and np.ptp(y) > 0
            out["ratio_spearman"] = float(spearmanr(p, y).statistic) if varies else np.nan
        else:
            out[f"{name}_auc"] = metrics.roc_auc(y, p)
            out[f"{name}_pr_auc"] = metrics.average_precision(y, p)
        out[f"{name}_ece"] = metrics.expected_calibration_error(y, p)
    out["loss"] = total
    return out


@dataclass
class FeedbackSet:
    """Listens of an evaluation period with their labels, arranged in context windows.

    Window ``i`` covers rows ``starts[i]:starts[i] + lengths[i]``; its last ``n_scored[i]``
    listens are predicted (each period listen exactly once, after at least ``seq_len // 2``
    earlier listens or the user's whole history). Earlier listens of the same day are part of
    the context with their feedback: the RM predicts the response to listen ``t`` from
    everything before ``t``.
    """

    name: str
    cutoff: int  # inputs and labels are observed before it
    rows: np.ndarray  # store rows in prediction order
    starts: np.ndarray
    lengths: np.ndarray
    n_scored: np.ndarray
    targets: np.ndarray  # [N, H]
    valid: np.ndarray  # [N, H]

    @property
    def n_events(self) -> int:
        return len(self.rows)


def period_bounds(store: EventStore, bounds: SplitBounds, split: str) -> tuple[int, int, int]:
    """``(start, end, cutoff)`` of ``"val"`` / ``"test"``; test ends with the data."""
    if split == "val":
        return bounds.val_start, bounds.val_end, bounds.val_end
    if split == "test":
        end = int(np.asarray(store.columns["timestamp"]).max()) + 1
        return bounds.test_start, end, end
    raise ValueError(f"unknown split {split!r}")


def build_feedback_set(
    store: EventStore,
    bounds: SplitBounds,
    split: str,
    seq_len: int,
    data_cfg: DataConfig,
    skip_threshold: float,
    max_users: int = 0,
    seed: int = 0,
) -> FeedbackSet:
    """Listens of the val / test day by users with train history (``max_users`` > 0 keeps a
    seeded random subset of them)."""
    start, end, cutoff = period_bounds(store, bounds, split)
    lo = store.offsets[:-1]
    users = np.flatnonzero(store.cutoff_index(bounds.train_end) > lo)
    if 0 < max_users < len(users):
        users = np.sort(np.random.default_rng(seed).choice(users, max_users, replace=False))
    first, stop = store.cutoff_index(start)[users], store.cutoff_index(end)[users]
    chunk = max(1, seq_len // 2)
    starts, lengths, n_scored, rows = [], [], [], []
    for u_lo, a, b in zip(lo[users], first, stop, strict=True):
        for c in range(a, b, chunk):
            c_end = min(c + chunk, b)
            w = max(u_lo, c_end - seq_len)
            starts.append(w)
            lengths.append(c_end - w)
            n_scored.append(c_end - c)
            rows.append(np.arange(c, c_end))
    rows_arr = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    cols = {name: np.asarray(values[rows_arr]) for name, values in store.columns.items()}
    targets, valid = feedback_targets(cols, cutoff, data_cfg, skip_threshold)
    return FeedbackSet(
        name=split,
        cutoff=cutoff,
        rows=rows_arr.astype(np.int64),
        starts=np.array(starts, dtype=np.int64),
        lengths=np.array(lengths, dtype=np.int64),
        n_scored=np.array(n_scored, dtype=np.int64),
        targets=targets,
        valid=valid,
    )


@torch.no_grad()
def predict(
    rm: RewardModel,
    store: EventStore,
    fs: FeedbackSet,
    batch_size: int = 128,
    amp_dtype: torch.dtype | None = None,
    overrides: Mapping[str, int] | None = None,
) -> np.ndarray:
    """``[E, N, H]`` probabilities for the listens of ``fs`` (in ``fs.rows`` order)."""
    rm.eval()
    device = next(rm.parameters()).device
    out = []
    for b in range(0, len(fs.starts), batch_size):
        sl = slice(b, b + batch_size)
        lengths = fs.lengths[sl]
        batch = to_tensors(store.gather(fs.starts[sl], lengths, rm.max_seq_len, fs.cutoff), device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = rm.teacher_forcing(batch, overrides)
        pos = np.arange(1, rm.max_seq_len)[None, :]  # position of the predicted listen
        sel = (pos >= (lengths - fs.n_scored[sl])[:, None]) & (pos < lengths[:, None])
        out.append(torch.sigmoid(logits[:, torch.from_numpy(sel).to(device)].float()).cpu())
    if not out:
        return np.zeros((rm.cfg.n_ensemble, 0, len(HEADS)), dtype=np.float32)
    return torch.cat(out, dim=1).numpy()


def build_reward_model(cfg: Config, items: pl.DataFrame, audio: np.ndarray | None = None):
    """A new RM; the encoder comes from ``runs/<rm.base_run>/best.pt`` if ``base_run`` is set."""
    n_items = N_SPECIAL_TOKENS + len(items)
    base = None
    model_cfg, seq_len = cfg.model, cfg.data.max_seq_len
    if cfg.rm.base_run:
        base = load_model(Path(cfg.data_root) / "runs" / cfg.rm.base_run / "best.pt")
        model_cfg, seq_len = base.cfg, base.backbone.config.n_positions
        if base.n_items != n_items:
            raise ValueError(f"base model has {base.n_items} items, vocabulary {n_items}")
    rm = RewardModel(
        cfg.rm,
        model_cfg,
        n_items,
        seq_len,
        item_length_bins(items, cfg.data),
        None if audio is None else torch.from_numpy(audio),
    )
    if base is not None:
        rm.encoder.load_state_dict(base.state_dict())
    return rm


def save_reward_model(path: Path, rm: RewardModel, **extra) -> None:
    save_checkpoint(
        path,
        model=rm.state_dict(),
        rm_cfg=asdict(rm.cfg),
        model_cfg=asdict(rm.encoder.cfg),
        n_items=rm.encoder.n_items,
        max_seq_len=rm.max_seq_len,
        audio_dim=None if rm.audio_proj is None else rm.audio_proj.in_features,
        **extra,
    )


def load_reward_model(path: str | Path, device: torch.device | str = "cpu") -> RewardModel:
    ckpt = load_checkpoint(path)
    n_items = ckpt["n_items"]
    audio = None
    if ckpt["audio_dim"] is not None:
        audio = torch.zeros(n_items, ckpt["audio_dim"])  # overwritten by the state dict
    rm = RewardModel(
        RewardModelConfig(**ckpt["rm_cfg"]),
        ModelConfig(**ckpt["model_cfg"]),
        n_items,
        ckpt["max_seq_len"],
        torch.zeros(n_items, dtype=torch.long),
        audio,
    )
    rm.load_state_dict(ckpt["model"])
    return rm.to(device)


def window_users(store: EventStore, starts: np.ndarray) -> np.ndarray:
    return np.searchsorted(store.offsets, starts, side="right") - 1


def train_reward_model(
    cfg: Config, prepared: PreparedData | None = None, device: torch.device | None = None
) -> Path:
    """Train (or resume) the RM of ``cfg.run_name``; returns the best checkpoint path
    (lowest val ``loss`` of ``feedback_metrics``)."""
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    set_seed(cfg.seed)
    device = device or get_device()
    prepared = prepared or prepare(cfg)
    store, items, bounds = prepared.store.in_memory(), prepared.items, prepared.bounds
    r = cfg.rm

    audio = load_audio_embeddings(cfg, items) if r.audio_embeddings else None
    rm = build_reward_model(cfg, items, audio).to(device)
    seq_len = rm.max_seq_len

    starts, lengths = prepared.window_starts, prepared.window_lengths
    rng = np.random.default_rng(cfg.seed)
    if r.user_fraction < 1.0:
        users = np.unique(window_users(store, starts))
        chosen = rng.choice(users, max(1, round(r.user_fraction * len(users))), replace=False)
        keep = np.isin(window_users(store, starts), chosen)
        starts, lengths = starts[keep], lengths[keep]
    n_windows = len(starts)
    weights = (
        rng.poisson(1.0, size=(n_windows, r.n_ensemble)).astype(np.float32)
        if r.bootstrap
        else np.ones((n_windows, r.n_ensemble), dtype=np.float32)
    )
    batches_per_epoch = n_windows // r.batch_size
    if batches_per_epoch == 0:
        raise ValueError(f"{n_windows} train windows < batch size {r.batch_size}")
    total_steps = r.epochs * batches_per_epoch
    if r.max_steps > 0:
        total_steps = min(total_steps, r.max_steps)

    optimizer = make_optimizer(rm, r.lr, r.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(r.warmup_steps, total_steps))
    dtype = amp_dtype(device, r.amp)
    scaler = torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)

    state = {"step": 0, "epoch": 0, "batch": 0, "best": float("inf"), "bad_evals": 0}
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    if last_path.exists():
        ckpt = load_checkpoint(last_path, map_location=device)
        rm.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        state = ckpt["state"]
        print(f"resumed from {last_path} at step {state['step']}")

    val = build_feedback_set(store, bounds, "val", seq_len, cfg.data, r.skip_threshold)
    log_path = run_dir / "log.jsonl"
    n_params = sum(p.numel() for p in rm.parameters()) / 1e6
    print(
        f"{n_params:.1f}M params, device {device}, amp {dtype}, {n_windows} windows, "
        f"{batches_per_epoch} steps/epoch, {total_steps} steps, {val.n_events} val listens"
    )

    def run_eval(train_loss: float) -> bool:
        probs = predict(rm, store, val, amp_dtype=dtype)
        val_metrics = feedback_metrics(probs.mean(0), val.targets, val.valid)
        improved = val_metrics["loss"] < state["best"]
        if improved:
            state["best"], state["bad_evals"] = val_metrics["loss"], 0
            save_reward_model(best_path, rm, step=state["step"], val=val_metrics)
        else:
            state["bad_evals"] += 1
        record = {"step": state["step"], "epoch": state["epoch"], "loss": train_loss}
        record |= {"lr": scheduler.get_last_lr()[0], "time": time.time()}
        record |= {f"val_{k}": v for k, v in val_metrics.items()}
        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(
            f"step {state['step']} loss {train_loss:.4f} val loss {val_metrics['loss']:.4f} "
            f"ratio mae {val_metrics['ratio_mae']:.4f} like auc {val_metrics['like_auc']:.4f}"
            + (" *" if improved else "")
        )
        save_checkpoint(
            last_path,
            model=rm.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),
            state=state,
        )
        rm.train()
        return state["bad_evals"] >= r.patience

    def batches(
        epoch: int, first: int
    ) -> Iterator[tuple[dict, np.ndarray, np.ndarray, np.ndarray]]:
        order = np.random.default_rng(cfg.seed + epoch).permutation(n_windows)
        for b in range(first, batches_per_epoch):
            sel = order[b * r.batch_size : (b + 1) * r.batch_size]
            batch = store.gather(starts[sel], lengths[sel], seq_len, bounds.train_end)
            targets, valid = feedback_targets(batch, bounds.train_end, cfg.data, r.skip_threshold)
            yield batch, targets[:, 1:], valid[:, 1:], weights[sel]

    rm.train()
    losses: list[float] = []
    stop = state["step"] >= total_steps
    while not stop and state["epoch"] < r.epochs:
        for batch, targets, valid, w in _prefetch(batches(state["epoch"], state["batch"])):
            batch = to_tensors(batch, device)
            targets = torch.from_numpy(targets).to(device)
            valid = torch.from_numpy(valid).to(device)
            w = torch.from_numpy(w).to(device)
            pos = valid.any(-1)  # [B, T - 1]
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                logits = rm.teacher_forcing(batch)
            w = w.T[:, :, None].expand(-1, -1, pos.shape[1])[:, pos]
            loss, _ = feedback_loss(logits[:, pos], targets[pos], valid[pos], w)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(rm.parameters(), r.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
            state["step"] += 1
            state["batch"] += 1
            if state["batch"] == batches_per_epoch:
                state["epoch"], state["batch"] = state["epoch"] + 1, 0
            if state["step"] % r.eval_every == 0 or state["step"] >= total_steps:
                stop = run_eval(float(np.mean(losses))) or state["step"] >= total_steps
                losses = []
            if stop:
                break
    if not best_path.exists():
        run_eval(float(np.mean(losses)) if losses else float("nan"))
    return best_path
