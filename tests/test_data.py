import numpy as np
import polars as pl
import pytest

from recsys_rl import data
from recsys_rl.config import Config, DataConfig

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


def make_store_events(extra: list[tuple] = ()) -> pl.LazyFrame:
    rows = [
        # uid, timestamp, item_id, played_ratio_pct, track_length_seconds, event_type
        (1, 0, 10, 100, 100, "listen"),
        (1, 10, 99, None, None, "like"),  # no listen of item 99: dropped
        (1, 50, 11, 3, 200, "listen"),
        (1, 100, 10, 60, 100, "listen"),
        (1, 120, 10, None, None, "like"),  # -> listen at 100, before the next listen
        (1, 3000, 11, None, None, "like"),  # after session 0 ended + 30 min: dropped
        (1, 5000, 12, 100, 400, "listen"),  # new session
        (1, 5100, 11, 100, 200, "listen"),
        (1, 5105, 11, None, None, "dislike"),  # -> listen at 5100
        (1, 5150, 12, None, None, "like"),  # -> listen at 5000, but after the next listen
        (2, 0, 10, 100, 100, "listen"),
        (2, 10, 13, 100, 100, "listen"),
        *extra,
    ]
    return pl.LazyFrame(
        [(*r[:5], 0, r[5]) for r in rows],
        schema={
            "uid": pl.UInt32,
            "timestamp": pl.UInt32,
            "item_id": pl.UInt32,
            "played_ratio_pct": pl.UInt16,
            "track_length_seconds": pl.UInt32,
            "is_organic": pl.UInt8,
            "event_type": EVENT_TYPE,
        },
        orient="row",
    ).sort("uid", "timestamp", maintain_order=True)


STORE_CFG = DataConfig(min_item_count=1, max_seq_len=2, window_stride=1)


def make_store() -> data.EventStore:
    vocab = pl.DataFrame(
        {"item_id": [10], "idx": [2]}, schema={"item_id": pl.UInt32, "idx": pl.UInt32}
    )
    return data.EventStore.from_table(data.build_event_table(make_store_events(), vocab, STORE_CFG))


def test_event_table_features():
    store = make_store()
    c = store.columns
    assert store.uids.tolist() == [1, 2]
    assert store.offsets.tolist() == [0, 5, 7]
    assert c["item_idx"].tolist() == [2, 1, 2, 1, 1, 2, 1]
    assert c["session_id"].tolist() == [0, 0, 0, 1, 1, 0, 0]
    assert c["next_gap"].tolist() == [50, 50, 4900, 100, -1, 10, -1]
    assert c["like_delay"].tolist() == [-1, -1, 20, 150, -1, -1, -1]
    assert c["dislike_delay"].tolist() == [-1, -1, -1, -1, 5, -1, -1]
    # ratio edges [0, 5, 10, 25, 49, 75, 99, 100]; length edges [120, 150, 180, 210, 240, 300]
    assert c["ratio_bin"].tolist() == [7, 1, 5, 7, 7, 7, 7]
    assert c["length_bin"].tolist() == [0, 3, 0, 6, 3, 0, 0]


def test_windows_and_contexts():
    store = make_store()
    starts, lengths = store.train_windows(train_end=5100, seq_len=2, stride=1)
    assert starts.tolist() == [2, 1, 0, 5]  # user 1: train rows 0..3, end-aligned
    assert lengths.tolist() == [2, 2, 2, 2]
    starts, lengths = store.contexts(cutoff=5100, seq_len=3)
    assert starts.tolist() == [1, 5]
    assert lengths.tolist() == [3, 2]


def test_gather_feedback_flags_are_leak_free():
    store = make_store()
    batch = store.gather(np.array([1, 5]), np.array([3, 2]), seq_len=3, cutoff=5100)
    assert batch["mask"].tolist() == [[True] * 3, [True, True, False]]
    assert batch["item_idx"].tolist() == [[1, 2, 1], [2, 1, data.PAD_IDX]]
    # the like of row 3 (delay 150) came after the next listen (gap 100): not an input
    assert batch["liked"].tolist() == [[0, 1, 0], [0, 0, 0]]
    rows_2_to_4 = (np.array([2]), np.array([3]))
    assert store.gather(*rows_2_to_4, seq_len=3, cutoff=5106)["disliked"].tolist() == [[0, 0, 1]]
    assert store.gather(*rows_2_to_4, seq_len=3, cutoff=5105)["disliked"].tolist() == [[0, 0, 0]]


def test_prepare_builds_and_reuses_cache(tmp_path):
    raw = tmp_path / "raw" / "flat" / "50m"
    raw.mkdir(parents=True)
    far = (2, 3 * D, 13, 100, 100, "listen")  # last timestamp: puts all other events in train
    make_store_events([far]).collect().write_parquet(raw / "multi_event.parquet")
    cfg = Config(data_root=str(tmp_path), data=STORE_CFG)

    first = data.prepare(cfg)
    assert first.bounds == data.SplitBounds.from_last_timestamp(3 * D, STORE_CFG)
    assert first.items["item_id"].to_list()[0] == 10  # most listened in train
    train_rows = first.store.columns["timestamp"] < first.bounds.train_end
    assert train_rows.sum() == 7  # the far event is in the store but not in train windows
    assert (first.window_starts + first.window_lengths <= 7).all()

    again = data.prepare(cfg)
    assert np.array_equal(again.window_starts, first.window_starts)
    assert isinstance(again.store.columns["item_idx"], np.memmap)

    cfg.data.window_stride = 2
    rebuilt = data.prepare(cfg)
    assert len(rebuilt.window_starts) < len(first.window_starts)
