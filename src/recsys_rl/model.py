"""GPT-2 next-item recommender (stage 2).

Item-ID tokens (+ optional feedback-feature embeddings) -> ``transformers.GPT2Model``
-> tied output projection. Trained with sampled softmax and logQ correction,
evaluated with full softmax over the vocabulary. The trained model is frozen as the
reference policy ``pi_ref`` and as the behaviour policy for off-policy correction.

Inputs are the right-padded batches of ``EventStore.gather``. The hidden state at position
``t`` sees listens ``<= t`` together with their feedback (played-ratio bin, like/dislike
flags that ``gather`` already filtered for leaks) and predicts the item at ``t + 1``.
Causal attention never looks at the right padding, so no attention mask is needed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPT2Config, GPT2Model

from recsys_rl.config import ModelConfig
from recsys_rl.data import N_SPECIAL_TOKENS, PAD_IDX
from recsys_rl.utils import load_checkpoint, save_checkpoint

# Batch columns the model reads; feedback features are summed with the item embedding.
INPUT_COLUMNS = ("item_idx", "ratio_bin", "length_bin", "is_organic", "liked", "disliked")


def to_tensors(batch: Mapping[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    """``EventStore.gather`` output -> tensors on ``device`` (integer columns as int64)."""
    out = {}
    for name, values in batch.items():
        t = torch.from_numpy(np.ascontiguousarray(values))
        out[name] = (t if t.dtype == torch.bool else t.long()).to(device, non_blocking=True)
    return out


class GPT2Recommender(nn.Module):
    def __init__(self, cfg: ModelConfig, n_items: int, max_seq_len: int):
        """``n_items`` is the vocabulary size including the special tokens."""
        super().__init__()
        self.cfg = cfg
        self.n_items = n_items
        d = cfg.n_embd
        self.item_emb = nn.Embedding(n_items, d, padding_idx=PAD_IDX)
        self.feature_emb = nn.ModuleDict()
        if cfg.use_feedback_tokens:
            sizes = {
                "ratio_bin": cfg.n_ratio_bins,
                "length_bin": cfg.n_length_bins,
                "is_organic": 2,
                "liked": 2,
                "disliked": 2,
            }
            self.feature_emb.update({name: nn.Embedding(n, d) for name, n in sizes.items()})
        self.backbone = GPT2Model(
            GPT2Config(
                vocab_size=1,  # token embeddings come from item_emb via inputs_embeds
                bos_token_id=None,
                eos_token_id=None,
                n_positions=max_seq_len,
                n_embd=d,
                n_layer=cfg.n_layer,
                n_head=cfg.n_head,
                resid_pdrop=cfg.dropout,
                embd_pdrop=cfg.dropout,
                attn_pdrop=cfg.dropout,
            )
        )
        for emb in (self.item_emb, *self.feature_emb.values()):
            nn.init.normal_(emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[PAD_IDX].zero_()

    def forward(
        self, batch: Mapping[str, torch.Tensor], extra: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Hidden states ``[B, T, d]``; position ``t`` is the query for the item at ``t + 1``.

        ``extra`` (``[B, T, d]``) is added to the input embeddings (the reward model's audio
        features)."""
        x = self.item_emb(batch["item_idx"])
        for name, emb in self.feature_emb.items():
            x = x + emb(batch[name])
        if extra is not None:
            x = x + extra
        return self.backbone(inputs_embeds=x).last_hidden_state

    def scores(self, hidden: torch.Tensor) -> torch.Tensor:
        """Full-softmax logits over the vocabulary; special tokens get ``-inf``."""
        logits = hidden @ self.item_emb.weight.T
        logits[..., :N_SPECIAL_TOKENS] = float("-inf")
        return logits

    def next_item_logits(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """``[B, n_items]`` logits for the item after the last listen of each sequence."""
        hidden = self(batch)
        last = batch["mask"].sum(1) - 1
        return self.scores(hidden[torch.arange(len(hidden), device=hidden.device), last])


def save_model(path: Path, model: GPT2Recommender, **extra) -> None:
    save_checkpoint(
        path,
        model=model.state_dict(),
        model_cfg=asdict(model.cfg),
        n_items=model.n_items,
        max_seq_len=model.backbone.config.n_positions,
        **extra,
    )


def load_model(path: str | Path, device: torch.device | str = "cpu") -> GPT2Recommender:
    ckpt = load_checkpoint(path)
    model = GPT2Recommender(ModelConfig(**ckpt["model_cfg"]), ckpt["n_items"], ckpt["max_seq_len"])
    model.load_state_dict(ckpt["model"])
    return model.to(device)


def sampled_softmax_loss(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    item_weight: torch.Tensor,
    negatives: torch.Tensor,
    log_q: torch.Tensor,
) -> torch.Tensor:
    """Mean cross-entropy of the target against shared sampled negatives, logQ-corrected.

    ``hidden`` is ``[N, d]``, ``targets`` ``[N]``, ``negatives`` ``[K]`` item indices drawn from
    the proposal ``q``; ``log_q`` is ``log q`` over the whole vocabulary. Each logit is
    ``<h, e_i> - log q(i)``, which makes the sampled softmax a consistent estimate of the full
    one; negatives that coincide with the target (accidental hits) are masked out.
    """
    pos = (hidden * item_weight[targets]).sum(-1) - log_q[targets]
    neg = hidden @ item_weight[negatives].T - log_q[negatives]
    neg = neg.masked_fill(negatives[None, :] == targets[:, None], float("-inf"))
    logits = torch.cat([pos[:, None], neg], dim=1).float()
    return F.cross_entropy(logits, torch.zeros_like(targets))
