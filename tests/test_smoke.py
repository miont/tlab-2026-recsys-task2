import torch
from transformers import GPT2Config, GPT2Model

from recsys_rl.config import load_config, save_config
from recsys_rl.utils import get_device, set_seed


def test_config_roundtrip(tmp_path):
    cfg = load_config("configs/base.yaml", run_name="test")
    assert cfg.run_name == "test"
    assert cfg.data.max_seq_len == 256
    save_config(cfg, tmp_path / "cfg.yaml")
    assert load_config(tmp_path / "cfg.yaml") == cfg


def test_tiny_gpt2_forward():
    set_seed(0)
    model = GPT2Model(GPT2Config(vocab_size=100, n_positions=32, n_embd=32, n_layer=1, n_head=2))
    out = model(input_ids=torch.randint(0, 100, (2, 16))).last_hidden_state
    assert out.shape == (2, 16, 32)
    assert get_device().type in {"cpu", "cuda", "mps"}
