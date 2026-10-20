import torch
import torch.nn.functional as F

from recsys_rl.config import ModelConfig
from recsys_rl.data import N_SPECIAL_TOKENS, PAD_IDX
from recsys_rl.model import GPT2Recommender, load_model, sampled_softmax_loss, save_model

CFG = ModelConfig(n_layer=2, n_head=2, n_embd=16, dropout=0.0)
N_ITEMS = 20


def make_batch(b: int = 3, t: int = 8, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    lengths = torch.tensor([t, t - 3, 1])[:b]
    mask = torch.arange(t)[None, :] < lengths[:, None]
    batch = {
        "item_idx": torch.randint(N_SPECIAL_TOKENS, N_ITEMS, (b, t), generator=g),
        "ratio_bin": torch.randint(0, CFG.n_ratio_bins, (b, t), generator=g),
        "length_bin": torch.randint(0, CFG.n_length_bins, (b, t), generator=g),
        "is_organic": torch.randint(0, 2, (b, t), generator=g),
        "liked": torch.randint(0, 2, (b, t), generator=g),
        "disliked": torch.randint(0, 2, (b, t), generator=g),
    }
    batch = {k: v * mask for k, v in batch.items()}  # padding rows are zero, item PAD_IDX
    return batch | {"mask": mask}


def make_model() -> GPT2Recommender:
    torch.manual_seed(0)
    return GPT2Recommender(CFG, N_ITEMS, max_seq_len=8).eval()


def test_forward_and_scores():
    model, batch = make_model(), make_batch()
    hidden = model(batch)
    assert hidden.shape == (3, 8, CFG.n_embd)
    logits = model.next_item_logits(batch)
    assert logits.shape == (3, N_ITEMS)
    assert torch.isinf(logits[:, :N_SPECIAL_TOKENS]).all()
    assert torch.isfinite(logits[:, N_SPECIAL_TOKENS:]).all()
    # the last valid position of each row is used, padding does not change it
    assert torch.allclose(logits[1], model.scores(hidden[1, 4]))
    assert model.item_emb.weight[PAD_IDX].abs().sum() == 0


def test_hidden_state_sees_only_the_past():
    model, batch = make_model(), make_batch()
    t = 4
    changed = {k: v.clone() for k, v in batch.items()}
    changed["item_idx"][:, t + 1 :] = N_SPECIAL_TOKENS  # future items and their feedback
    for name in ("ratio_bin", "liked", "disliked"):
        changed[name][:, t + 1 :] = 1 - changed[name][:, t + 1 :].clamp(max=1)
    with torch.no_grad():
        a, b = model(batch), model(changed)
    assert torch.allclose(a[:, : t + 1], b[:, : t + 1], atol=1e-6)
    assert not torch.allclose(a[:, t + 1 :], b[:, t + 1 :])


def test_no_feedback_ablation_ignores_features():
    cfg = ModelConfig(n_layer=1, n_head=2, n_embd=16, dropout=0.0, use_feedback_tokens=False)
    model = GPT2Recommender(cfg, N_ITEMS, max_seq_len=8).eval()
    batch = make_batch()
    changed = batch | {"ratio_bin": torch.zeros_like(batch["ratio_bin"])}
    with torch.no_grad():
        assert torch.equal(model(batch), model(changed))


def test_sampled_softmax_matches_full_softmax():
    torch.manual_seed(0)
    hidden, weight = torch.randn(5, 4), torch.randn(N_ITEMS, 4)
    targets = torch.randint(N_SPECIAL_TOKENS, N_ITEMS, (5,))
    items = torch.arange(N_SPECIAL_TOKENS, N_ITEMS)
    full = F.cross_entropy(hidden @ weight[items].T, targets - N_SPECIAL_TOKENS)
    # every item as a negative + uniform proposal: logQ is a constant shift, the target's own
    # copy among the negatives is masked, so the sampled loss equals the full one
    log_q = torch.full((N_ITEMS,), -3.0)
    assert torch.allclose(sampled_softmax_loss(hidden, targets, weight, items, log_q), full)
    # logQ correction: a non-uniform q changes logits by -log q(i)
    log_q = torch.log(torch.rand(N_ITEMS) + 0.1)
    corrected = F.cross_entropy(hidden @ weight[items].T - log_q[items], targets - 2)
    assert torch.allclose(sampled_softmax_loss(hidden, targets, weight, items, log_q), corrected)


def test_save_and_load(tmp_path):
    model, batch = make_model(), make_batch()
    save_model(tmp_path / "m.pt", model, step=3)
    loaded = load_model(tmp_path / "m.pt").eval()
    with torch.no_grad():
        assert torch.equal(model(batch), loaded(batch))
