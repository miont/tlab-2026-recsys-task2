import json

import numpy as np
import polars as pl
import pytest
import torch
import torch.nn.functional as F

from recsys_rl import evaluate, rm_eval
from recsys_rl.config import (
    Config,
    DataConfig,
    ModelConfig,
    RewardModelConfig,
    TrainConfig,
)
from recsys_rl.data import DAY_SECONDS, N_SPECIAL_TOKENS, OOV_IDX, PAD_IDX, prepare
from recsys_rl.reward_model import (
    HEAD_INDEX,
    HEADS,
    RewardModel,
    build_feedback_set,
    build_reward_model,
    context_repeat_counts,
    feedback_loss,
    feedback_targets,
    item_length_bins,
    load_audio_embeddings,
    load_reward_model,
    predict,
    prefix_repeat_counts,
    train_reward_model,
)
from recsys_rl.train import train

DATA_CFG = DataConfig(session_gap_minutes=30)
MODEL_CFG = ModelConfig(n_layer=1, n_head=2, n_embd=16, dropout=0.0)


def test_feedback_targets():
    cols = {
        "item_idx": np.array([5, 5, OOV_IDX, 5, 6]),
        "played_ratio_pct": np.array([100, 20, 60, 50, 140]),
        "timestamp": np.array([0, 100, 200, 300, 10_000]),
        "next_gap": np.array([100, 100, 100, 9_700, -1]),
        "like_delay": np.array([5, -1, 5, 10_300, -1]),
        "dislike_delay": np.array([-1, 3, -1, -1, -1]),
    }
    targets, valid = feedback_targets(cols, cutoff=10_500, data_cfg=DATA_CFG, skip_threshold=0.3)
    t = {name: targets[:, i].tolist() for name, i in HEAD_INDEX.items()}
    assert t["ratio"] == pytest.approx([1.0, 0.2, 0.6, 0.5, 1.0])
    assert t["listen_plus"] == [1, 0, 1, 1, 1]
    assert t["skip"] == [0, 1, 0, 0, 0]
    assert t["like"] == [1, 0, 1, 0, 0]  # the like of listen 3 comes after the cutoff
    assert t["dislike"] == [0, 1, 0, 0, 0]
    assert t["continue"] == [1, 1, 1, 0, 0]  # gap 9700 s > 30 min; the last listen has none
    assert valid[:, 0].tolist() == [True, True, False, True, True]  # OOV
    # continuation of the last listen is unknown: ts + 30 min is after the cutoff
    assert valid[:, HEAD_INDEX["continue"]].tolist() == [True, True, False, True, False]


def test_repeat_counts():
    items = torch.tensor([[2, 3, 2, 2, PAD_IDX], [4, 4, 5, PAD_IDX, PAD_IDX]])
    mask = items != PAD_IDX
    assert prefix_repeat_counts(items, mask).tolist() == [[0, 0, 1, 2, 0], [0, 1, 0, 0, 0]]
    cands = torch.tensor([[2, 3, 9], [4, 5, PAD_IDX]])
    assert context_repeat_counts(items, mask, cands).tolist() == [[3, 1, 0], [2, 1, 0]]


N_ITEMS = 20


def make_rm(audio: bool = False, seed: int = 0) -> RewardModel:
    torch.manual_seed(seed)
    length_bin = torch.randint(0, MODEL_CFG.n_length_bins, (N_ITEMS,))
    emb = torch.randn(N_ITEMS, 4) if audio else None
    cfg = RewardModelConfig(n_ensemble=3, head_hidden=8)
    return RewardModel(
        cfg, MODEL_CFG, N_ITEMS, max_seq_len=8, item_length_bin=length_bin, audio=emb
    )


def make_batch(b: int = 2, t: int = 8, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    batch = {
        "item_idx": torch.randint(N_SPECIAL_TOKENS, 8, (b, t), generator=g),  # many repeats
        "ratio_bin": torch.randint(0, MODEL_CFG.n_ratio_bins, (b, t), generator=g),
        "length_bin": torch.randint(0, MODEL_CFG.n_length_bins, (b, t), generator=g),
        "is_organic": torch.zeros(b, t, dtype=torch.long),
        "liked": torch.randint(0, 2, (b, t), generator=g),
        "disliked": torch.randint(0, 2, (b, t), generator=g),
    }
    return batch | {"mask": torch.ones(b, t, dtype=torch.bool)}


@pytest.mark.parametrize("audio", [False, True])
def test_teacher_forcing_matches_score(audio):
    rm, batch = make_rm(audio).eval(), make_batch()
    with torch.no_grad():
        logits = rm.teacher_forcing(batch)
        assert logits.shape == (3, 2, 7, len(HEADS))
        for t in (1, 4, 7):  # the listen at position t, scored as a candidate after [:t]
            context = {k: v.clone() for k, v in batch.items()}
            context["mask"][:, t:] = False
            context["item_idx"][:, t:] = PAD_IDX
            probs = rm.score(context, batch["item_idx"][:, t : t + 1])
            assert torch.allclose(probs[:, :, 0], torch.sigmoid(logits[:, :, t - 1]), atol=1e-5)


def test_prediction_does_not_see_own_feedback_or_future():
    rm, batch = make_rm().eval(), make_batch()
    t = 4
    changed = {k: v.clone() for k, v in batch.items()}
    for name in ("ratio_bin", "liked", "disliked"):  # feedback of listen t and later
        changed[name][:, t:] = (changed[name][:, t:] + 1) % 2
    changed["item_idx"][:, t + 1 :] = 9
    with torch.no_grad():
        a, b = rm.teacher_forcing(batch), rm.teacher_forcing(changed)
    assert torch.allclose(a[:, :, :t], b[:, :, :t])  # predictions for listens 1..t
    assert not torch.allclose(a[:, :, t:], b[:, :, t:])


def test_overrides_and_bad_override():
    rm, batch = make_rm().eval(), make_batch()
    with torch.no_grad():
        base = rm.teacher_forcing(batch)
        forced = rm.teacher_forcing(batch, {"repeat_count": 0})
        assert not torch.allclose(base, forced)
        with pytest.raises(KeyError):
            rm.teacher_forcing(batch, {"ratio_bin": 1})


def test_feedback_loss_matches_manual_bce():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, len(HEADS))
    targets = torch.rand(5, len(HEADS))
    valid = torch.rand(5, len(HEADS)) > 0.3
    weights = torch.tensor([[1.0, 0, 2, 1, 1], [0.0, 1, 1, 1, 3]])
    total, per_head = feedback_loss(logits, targets, valid, weights)
    bce = F.binary_cross_entropy_with_logits(logits, targets.expand_as(logits), reduction="none")
    for h in range(len(HEADS)):
        member = [
            (bce[e, :, h] * weights[e] * valid[:, h]).sum() / (weights[e] * valid[:, h]).sum()
            for e in range(2)
        ]
        assert torch.allclose(per_head[h], torch.stack(member).mean())
    assert torch.allclose(total, per_head.sum())


def write_events(raw_dir, n_users=16, n_items=100, days=4, step=1800):
    """Users cycle through the items (one listen per ``step``); some full plays get a like."""
    rng = np.random.default_rng(0)
    ts = np.arange(0, days * DAY_SECONDS, step)
    item = (20 * np.arange(n_users)[:, None] + np.arange(len(ts))[None, :]) % n_items + 100
    listens = pl.DataFrame(
        {
            "uid": np.repeat(np.arange(n_users), len(ts)),
            "timestamp": np.tile(ts, n_users),
            "item_id": item.ravel(),
            "played_ratio_pct": rng.choice([3, 60, 100], size=n_users * len(ts)),
            "is_organic": rng.integers(0, 2, n_users * len(ts)),
            "track_length_seconds": rng.choice([100, 200, 400], size=n_users * len(ts)),
            "event_type": "listen",
        }
    )
    likes = listens.filter(
        (pl.col("played_ratio_pct") == 100) & (pl.col("item_id") % 3 == 0)
    ).with_columns(pl.col("timestamp") + 10, event_type=pl.lit("like"))
    events = (
        pl.concat([listens, likes])
        .sort("uid", "timestamp")
        .with_columns(
            pl.col("uid").cast(pl.UInt32),
            pl.col("timestamp").cast(pl.UInt32),
            pl.col("item_id").cast(pl.UInt32),
            pl.col("played_ratio_pct").cast(pl.UInt16),
            pl.col("track_length_seconds").cast(pl.UInt32),
            pl.col("is_organic").cast(pl.UInt8),
            pl.col("event_type").cast(pl.Enum(["listen", "like", "dislike"])),
        )
    )
    raw_dir.mkdir(parents=True)
    events.write_parquet(raw_dir / "multi_event.parquet")


def make_cfg(tmp_path, run_name="rm", **rm) -> Config:
    return Config(
        data_root=str(tmp_path),
        run_name=run_name,
        data=DataConfig(max_seq_len=16, window_stride=8, min_item_count=1),
        model=MODEL_CFG,
        train=TrainConfig(batch_size=8, n_negatives=16, warmup_steps=2, amp=False, max_steps=4),
        rm=RewardModelConfig(
            n_ensemble=2, head_hidden=8, batch_size=8, warmup_steps=2, amp=False, **rm
        ),
    )


@pytest.fixture
def prepared_data(tmp_path):
    write_events(tmp_path / "raw" / "flat" / "50m")
    cfg = make_cfg(tmp_path)
    return cfg, prepare(cfg)


def test_feedback_set_scores_every_period_listen_once(prepared_data):
    cfg, prepared = prepared_data
    store, bounds = prepared.store, prepared.bounds
    for split, start in (("val", bounds.val_start), ("test", bounds.test_start)):
        fs = build_feedback_set(store, bounds, split, 16, cfg.data, 0.3)
        ts = store.columns["timestamp"][fs.rows]
        assert (ts >= start).all() and len(np.unique(fs.rows)) == fs.n_events
        expected = (store.columns["timestamp"] >= start) & (
            store.columns["timestamp"] < (bounds.val_end if split == "val" else 2**40)
        )
        assert fs.n_events == expected.sum()  # every user has train history here
        assert fs.n_scored.sum() == fs.n_events and (fs.lengths <= 16).all()
        # the scored listens are the last ones of each window and follow it in order
        assert np.array_equal(
            fs.rows,
            np.concatenate(
                [
                    np.arange(s + n - k, s + n)
                    for s, n, k in zip(fs.starts, fs.lengths, fs.n_scored, strict=True)
                ]
            ),
        )
        assert (fs.lengths - fs.n_scored >= 8).all()  # at least seq_len // 2 of context
    small = build_feedback_set(store, bounds, "val", 16, cfg.data, 0.3, max_users=3)
    assert len(np.unique(np.searchsorted(store.offsets, small.rows, side="right"))) == 3


def test_train_rm_from_base_resume_and_eval(prepared_data, tmp_path):
    base_cfg, prepared = prepared_data
    base_cfg.run_name = "base"
    base_path = train(base_cfg, prepared, torch.device("cpu"))

    cfg = make_cfg(tmp_path, base_run="base", max_steps=4, eval_every=2, user_fraction=0.5)
    fresh = build_reward_model(cfg, prepared.items)
    base_state = torch.load(base_path, weights_only=False)["model"]
    assert all(torch.equal(fresh.encoder.state_dict()[k], v) for k, v in base_state.items())

    best = train_reward_model(cfg, prepared, torch.device("cpu"))
    log = [json.loads(line) for line in (cfg.run_dir / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == [2, 4]
    assert all(np.isfinite(r["loss"]) and np.isfinite(r["val_loss"]) for r in log)
    assert {"val_ratio_mae", "val_like_auc", "val_continue_auc"} <= set(log[0])
    cfg.rm.max_steps = 6
    train_reward_model(cfg, prepared, torch.device("cpu"))
    log = (cfg.run_dir / "log.jsonl").read_text().splitlines()
    assert [json.loads(r)["step"] for r in log] == [2, 4, 6]

    rm = load_reward_model(best)
    assert torch.equal(rm.item_length_bin, item_length_bins(prepared.items, cfg.data))
    store, bounds = prepared.store, prepared.bounds
    fs = build_feedback_set(store, bounds, "test", rm.max_seq_len, cfg.data, 0.3)
    probs = predict(rm, store, fs, batch_size=3)
    assert probs.shape == (2, fs.n_events, len(HEADS))
    assert ((probs > 0) & (probs < 1)).all()

    preds = {"rm": probs.mean(0)} | rm_eval.baseline_predictions(store, bounds, fs, cfg.data, 0.3)
    quality = rm_eval.quality_table(preds, fs)
    assert quality["method"].to_list() == ["rm", "global", "user_mean", "item_mean", "user+item"]
    assert quality.filter(method="global")["ratio_mean_pred"].item() == pytest.approx(
        preds["global"][0, 0], rel=1e-5
    )
    slices = rm_eval.event_slices(store, prepared.items, fs, cfg.data)
    assert slices.height == fs.n_events
    hist = rm_eval.history_features(store)
    item, session = store.columns["item_id"], store.columns["session_id"]
    for u in (0, 5):  # brute force over the user's rows
        lo, hi = store.offsets[u], store.offsets[u + 1]
        seen = [item[r] in item[lo:r] for r in range(lo, hi)]
        pos = [int((session[lo:r] == session[r]).sum()) for r in range(lo, hi)]
        assert hist["is_repeat"][lo:hi].to_list() == seen
        assert hist["session_position"][lo:hi].to_list() == pos
    table = rm_eval.slice_table(preds, fs, slices)
    assert set(table["slice"]) == set(slices.columns)
    calib = rm_eval.calibration_table(preds, fs, "ratio")
    assert calib.height == 10 * len(preds)
    assert rm_eval.uncertainty_table(probs, fs)["events"].sum() == fs.valid[:, 0].sum()
    sens = rm_eval.sensitivity(rm, store, fs, cfg.model.n_length_bins)
    assert sens.height == cfg.model.n_length_bins + len(rm_eval.REPEAT_COUNTS) + 2

    es = evaluate.build_eval_set(store, bounds, "test", rm.max_seq_len, 0.5)
    per_item, top_k = rm_eval.catalog_profile(rm, store, prepared.items, es, n_users=5, k=10)
    assert per_item.height == len(prepared.items) and top_k.height == 5
    assert per_item["mean_score"].is_between(0, 1).all()


def test_baselines_on_known_rows(prepared_data):
    cfg, prepared = prepared_data
    store, bounds = prepared.store, prepared.bounds
    fs = build_feedback_set(store, bounds, "val", 16, cfg.data, 0.3)
    preds = rm_eval.baseline_predictions(store, bounds, fs, cfg.data, 0.3, prior=0.0)
    train_rows = store.columns["timestamp"] < bounds.train_end
    ratio = np.minimum(store.columns["played_ratio_pct"], 100) / 100
    assert preds["global"][0, 0] == pytest.approx(ratio[train_rows].mean(), rel=1e-5)
    u = 3  # with prior 0 the user baseline is the plain train mean
    lo, hi = store.offsets[u], store.offsets[u + 1]
    rows = np.flatnonzero(train_rows[lo:hi]) + lo
    sel = (fs.rows >= lo) & (fs.rows < hi)
    assert np.allclose(preds["user_mean"][sel, 0], ratio[rows].mean())


def test_audio_embeddings(tmp_path):
    cfg = Config(data_root=str(tmp_path))
    cfg.raw_dir.mkdir(parents=True)
    pl.DataFrame(
        {"item_id": [10, 12, 99], "normalized_embed": [[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]]}
    ).write_parquet(cfg.raw_dir / "embeddings.parquet")
    items = pl.DataFrame(
        {"item_id": [10, 11, 12], "idx": [2, 3, 4]}, schema={"item_id": pl.UInt32, "idx": pl.UInt32}
    )
    emb = load_audio_embeddings(cfg, items)
    assert emb.shape == (5, 2) and emb.dtype == np.float16
    assert emb.tolist() == [[0, 0], [0, 0], [1, 0], [0, 0], [0, 1]]
    assert (cfg.processed_dir / "audio_embeddings.npy").exists()  # cached
