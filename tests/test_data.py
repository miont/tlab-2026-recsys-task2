import polars as pl
import pytest

from recsys_rl import data
from recsys_rl.config import DataConfig

D = data.DAY_SECONDS
CFG = DataConfig(test_days=1, val_days=1, gap_minutes=30, min_item_count=1, max_vocab_size=1)
LAST = 10 * D
EVENT_TYPE = pl.Enum(["listen", "dislike", "like", "undislike", "unlike"])


def make_events() -> pl.LazyFrame:
    rows = [
        # uid, timestamp, item_id, played_ratio_pct, event_type
        (1, 0, 10, 100, "listen"),
        (1, 100, 11, 100, "listen"),
        (1, 8 * D - 3605, 11, 30, "listen"),  # last train second
        (1, 8 * D - 3600, 12, 100, "listen"),  # first gap second
        (1, 8 * D, 13, 60, "listen"),  # val
        (1, 9 * D, 14, 50, "listen"),  # test, exactly at the Listen+ threshold
        (1, 9 * D + 5, 14, None, "like"),
        (1, 9 * D + 10, 15, 49, "listen"),
        (2, 9 * D + 10, 16, 100, "listen"),  # user absent from train
    ]
    return pl.LazyFrame(
        rows,
        schema={
            "uid": pl.UInt32,
            "timestamp": pl.UInt32,
            "item_id": pl.UInt32,
            "played_ratio_pct": pl.UInt16,
            "event_type": EVENT_TYPE,
        },
        orient="row",
    )


def test_split_bounds():
    b = data.SplitBounds.from_last_timestamp(LAST, CFG)
    assert b == data.SplitBounds(
        train_end=8 * D - 3600, val_start=8 * D - 1800, val_end=9 * D - 1800, test_start=9 * D
    )
    assert data.SplitBounds.from_events(make_events(), CFG) == data.SplitBounds.from_last_timestamp(
        9 * D + 10, CFG
    )
    with pytest.raises(ValueError):
        data.SplitBounds.from_last_timestamp(D, CFG)


def test_temporal_split():
    splits = data.temporal_split(make_events(), data.SplitBounds.from_last_timestamp(LAST, CFG))
    items = {name: lf.collect()["item_id"].to_list() for name, lf in splits.items()}
    assert items == {"train": [10, 11, 11], "val": [13], "test": [14, 14, 15]}  # 12 in the gap


def test_eval_targets():
    splits = data.temporal_split(make_events(), data.SplitBounds.from_last_timestamp(LAST, CFG))
    targets = data.eval_targets(splits["test"], threshold=0.5)
    assert targets.to_dicts() == [{"uid": 1, "item_id": [14]}]  # likes and ratio 49 excluded


def test_sessions():
    events = data.add_session_ids(make_events(), gap_minutes=30).collect()
    assert events["session_id"].to_list() == [0, 0, 1, 1, 2, 3, 3, 3, 0]


def test_vocab_and_encoding():
    train = data.temporal_split(make_events(), data.SplitBounds.from_last_timestamp(LAST, CFG))[
        "train"
    ]
    vocab = data.build_vocab(train, CFG)
    assert vocab.to_dicts() == [{"item_id": 11, "idx": data.N_SPECIAL_TOKENS, "count": 2}]
    encoded = data.encode_items(train, vocab).collect()
    n = data.N_SPECIAL_TOKENS
    assert encoded["item_idx"].to_list() == [data.OOV_IDX, n, n]
