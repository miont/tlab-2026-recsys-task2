import numpy as np
import polars as pl
import pytest
import torch

from recsys_rl import data, evaluate
from recsys_rl.config import DataConfig, ModelConfig
from recsys_rl.model import GPT2Recommender

D = data.DAY_SECONDS
CFG = DataConfig(test_days=1, val_days=1, gap_minutes=30)
BOUNDS = data.SplitBounds.from_last_timestamp(10 * D, CFG)  # train < 8D - 3600, test >= 9D


def make_store() -> data.EventStore:
    rows = [
        # uid, timestamp, item_id, played_ratio_pct
        (1, 0, 10, 100),
        (1, 100, 11, 100),
        (1, 200, 12, 30),
        (1, 300, 11, 100),
        (1, 8 * D, 13, 60),  # val Listen+
        (1, 8 * D + 10, 14, 40),  # val, not Listen+
        (1, 9 * D, 15, 100),  # test, listened twice
        (1, 9 * D + 5, 15, 100),
        (2, 0, 10, 100),
        (2, 8 * D, 12, 10),  # no val Listen+: user 2 is not in the val set
        (2, 9 * D, 11, 80),
        (3, 8 * D, 13, 100),  # no train history: in no eval set
    ]
    events = pl.LazyFrame(
        [(*r, 200, 1, "listen") for r in rows],
        schema={
            "uid": pl.UInt32,
            "timestamp": pl.UInt32,
            "item_id": pl.UInt32,
            "played_ratio_pct": pl.UInt16,
            "track_length_seconds": pl.UInt32,
            "is_organic": pl.UInt8,
            "event_type": pl.Enum(["listen", "like", "dislike"]),
        },
        orient="row",
    )
    vocab = make_items()
    return data.EventStore.from_table(data.build_event_table(events, vocab, CFG))


def make_items() -> pl.DataFrame:
    return pl.DataFrame(
        {"item_id": range(10, 16), "idx": range(2, 8), "count": [5, 4, 3, 2, 1, 1]},
        schema={"item_id": pl.UInt32, "idx": pl.UInt32, "count": pl.UInt32},
    )


def test_eval_sets():
    store = make_store()
    val = evaluate.build_eval_set(store, BOUNDS, "val", seq_len=3, threshold=0.5)
    assert val.users.tolist() == [0]
    assert [t.tolist() for t in val.targets] == [[13]]
    assert (val.context_starts.tolist(), val.context_ends.tolist()) == ([1], [4])
    assert val.cutoff == BOUNDS.train_end

    test = evaluate.build_eval_set(store, BOUNDS, "test", seq_len=3, threshold=0.5)
    assert test.users.tolist() == [0, 1]
    assert [t.tolist() for t in test.targets] == [[15, 15], [11]]
    assert test.context_ends.tolist() == [6, 10]  # context includes the val day
    assert [h.tolist() for h in test.histories(store)] == [[10, 11, 12, 11, 13, 14], [10, 12]]
    with pytest.raises(ValueError):
        evaluate.build_eval_set(store, BOUNDS, "train", seq_len=3, threshold=0.5)


def test_baselines():
    store = make_store()
    val = evaluate.build_eval_set(store, BOUNDS, "val", seq_len=3, threshold=0.5)
    # train Listen+ counts: 10 x2, 11 x2 (12 was played to 30%), ties by item id
    assert evaluate.popular_items(store, BOUNDS.train_end, days=10, threshold=0.5).tolist() == [
        10,
        11,
    ]
    assert evaluate.recommend_most_pop(store, val, k=2, days=10).tolist() == [[10, 11]]
    assert evaluate.recommend_repeat(store, val, k=3, days=10).tolist() == [[11, 12, 10]]
    with pytest.raises(ValueError):  # user 1 has heard both popular items
        evaluate.recommend_most_pop(store, val, k=2, days=10, exclude_seen=True)

    test = evaluate.build_eval_set(store, BOUNDS, "test", seq_len=3, threshold=0.5)
    # popularity is counted before the test cutoff, so the val Listen+ of 13 counts too
    recs = evaluate.recommend_most_pop(store, test, k=2, days=10)
    assert recs.tolist() == [[10, 11], [10, 11]]
    metrics = evaluate.evaluate_recs(recs, test, n_items=6, ks=(1, 2))
    assert metrics["recall@2"] == pytest.approx(0.5)  # user 2 hits item 11
    assert metrics["coverage@2"] == pytest.approx(2 / 6)


def test_recommend_model_and_repeat_filter():
    store, items = make_store(), make_items()
    torch.manual_seed(0)
    model = GPT2Recommender(ModelConfig(n_layer=1, n_head=2, n_embd=8), 8, max_seq_len=3)
    test = evaluate.build_eval_set(store, BOUNDS, "test", seq_len=3, threshold=0.5)
    idx_to_item = evaluate.idx_to_item_array(items)
    assert idx_to_item.tolist() == [0, 0, 10, 11, 12, 13, 14, 15]
    recs = evaluate.recommend_model(model, store, test, idx_to_item, k=6, batch_size=1)
    assert sorted(recs[0].tolist()) == [10, 11, 12, 13, 14, 15]
    # user 1 has heard everything but 15, user 2 only 10 and 12
    filtered = evaluate.recommend_model(model, store, test, idx_to_item, k=1, exclude_seen=True)
    assert filtered[0].tolist() == [15]
    assert filtered[1][0] not in {10, 12}
    with pytest.raises(ValueError):
        evaluate.recommend_model(model, store, test, idx_to_item, k=2, exclude_seen=True)


def make_meta() -> pl.DataFrame:
    items = make_items().with_columns(mean_ratio=pl.lit(0.5), track_length=pl.lit(200.0))
    artists = pl.DataFrame({"item_id": [10, 11, 12, 13, 14, 15], "artist_id": [1, 1, 2, 2, 3, 4]})
    albums = pl.DataFrame({"item_id": [10, 11, 99], "album_id": [7, 8, 9]})
    return evaluate.item_meta(artists, albums, items)


def test_item_meta():
    meta = make_meta()
    assert meta["item_id"].to_list() == [10, 11, 12, 13, 14, 15, 99]
    assert meta["pop_rank"].to_list() == [1, 2, 3, 4, 5, 6, None]
    assert meta["album_id"].to_list() == [7, 8, None, None, None, None, 9]


def test_explain_user_and_profile():
    store, meta = make_store(), make_meta()
    torch.manual_seed(0)
    model = GPT2Recommender(ModelConfig(n_layer=1, n_head=2, n_embd=8), 8, max_seq_len=3)
    test = evaluate.build_eval_set(store, BOUNDS, "test", seq_len=3, threshold=0.5)
    idx_to_item = evaluate.idx_to_item_array(make_items())
    ex = evaluate.explain_user(
        model, store, test, 0, idx_to_item, meta, k=3, n_history=2, most_pop_days=10
    )
    assert ex.uid == 1
    assert ex.history["item_id"].to_list() == [13, 14]  # the end of the 3-listen context
    assert ex.history["artist_id"].to_list() == [2, 3]
    assert ex.recs.group_by("method").len().sort("method")["len"].to_list() == [3, 3, 3]
    model_recs = ex.recs.filter(method="model")
    assert model_recs["prob"].is_sorted(descending=True)
    repeat = ex.recs.filter(method="Repeat")
    assert repeat["item_id"].to_list() == [14, 13, 11]
    assert repeat["heard"].to_list() == [1, 1, 2] and repeat["artist_heard"].all()
    # target 15 (twice) is the only unheard item, by an unheard artist
    assert ex.targets["item_id"].to_list() == [15, 15]
    assert ex.targets["heard"].to_list() == [0, 0] and not ex.targets["artist_heard"].any()

    # user 2 heard 10 and 12; its target 11 is unheard
    filtered = evaluate.explain_user(
        model, store, test, 1, idx_to_item, meta, k=2, exclude_seen=True, most_pop_days=10
    )
    model_recs = filtered.recs.filter(method="model")
    assert set(model_recs["item_id"].to_list()) <= {11, 13, 14, 15}
    assert filtered.recs.filter(method="MostPop")["item_id"].to_list() == [11, 13]
    assert filtered.targets["model_rank"].item() in {1, 2, 3, 4}

    recs = evaluate.recommend_repeat(store, test, k=2, days=10)  # [[14, 13], [12, 10]]
    profile = evaluate.recs_profile(recs, test, store, meta, k=2)
    assert profile["heard@2"] == pytest.approx(1.0)
    assert profile["median_pop_rank@2"] == pytest.approx(3.5)  # ranks 5, 4, 3, 1
    single = evaluate.recs_profile(np.array([[15, 12]]), test.subset([1]), store, meta, k=2)
    # user 2 heard 10 and 12: 15 is a new artist, 12 was heard
    assert single == pytest.approx(
        {"heard@2": 0.5, "known_artist@2": 0.0, "new_artist@2": 0.5, "median_pop_rank@2": 4.5}
    )
