import numpy as np
import pytest

from recsys_rl.data import EventStore
from recsys_rl.longterm import (
    decile_table,
    length_table,
    session_outcomes,
    user_return_table,
    validity_table,
)
from recsys_rl.reward_model import HEAD_INDEX, HEADS, FeedbackSet

CUTOFF = 10_000


def make_data() -> tuple[EventStore, FeedbackSet]:
    """User 0: sessions of 3 and 2 listens in the period and one listen after it; user 1: a
    session of 2 and a last listen whose session may continue past the cutoff."""
    store = EventStore(
        uids=np.array([10, 11], dtype=np.uint32),
        offsets=np.array([0, 6, 9]),
        columns={
            "timestamp": np.array([0, 100, 200, 5000, 5100, 200_000, 0, 50, 9000]),
            "session_id": np.array([0, 0, 0, 1, 1, 2, 0, 0, 1]),
            "length_bin": np.array([0, 1, 1, 0, 2, 0, 1, 2, 0]),
        },
    )
    rows = np.array([0, 1, 2, 3, 4, 6, 7, 8])
    targets = np.zeros((len(rows), len(HEADS)), dtype=np.float32)
    targets[:, HEAD_INDEX["continue"]] = [1, 1, 0, 1, 0, 1, 0, 0]
    valid = np.ones_like(targets, dtype=bool)
    valid[-1, HEAD_INDEX["continue"]] = False  # 9000 + 30 min is after the cutoff
    fs = FeedbackSet(
        name="val",
        cutoff=CUTOFF,
        rows=rows,
        starts=np.array([0]),
        lengths=np.array([1]),
        n_scored=np.array([1]),
        targets=targets,
        valid=valid,
    )
    return store, fs


def test_session_outcomes():
    store, fs = make_data()
    out = session_outcomes(store, fs, session_gap_minutes=30, return_from=100_000)
    assert out["user"].to_list() == [0, 0, 0, 0, 0, 1, 1, 1]
    assert out["remaining_listens"].to_list() == [2, 1, 0, 1, 0, 1, 0, 0]
    assert out["session_position"].to_list() == [0, 1, 2, 0, 1, 0, 1, 0]
    assert out["remaining_minutes"].to_list() == pytest.approx(
        [200 / 60, 100 / 60, 0, 100 / 60, 0, 50 / 60, 0, 0]
    )
    assert out["censored"].to_list() == [False] * 7 + [True]
    assert out["returns"].to_list() == [True] * 5 + [False] * 3
    assert session_outcomes(store, fs, 30)["returns"].null_count() == len(out)


def test_validity_tables():
    store, fs = make_data()
    outcomes = session_outcomes(store, fs, session_gap_minutes=30, return_from=100_000)
    pred = np.full(fs.targets.shape, 0.5, dtype=np.float32)
    pred[:, HEAD_INDEX["continue"]] = 0.2 + 0.6 * fs.targets[:, HEAD_INDEX["continue"]]
    pred[:, HEAD_INDEX["ratio"]] = np.linspace(0.1, 0.9, len(pred))
    preds = {"rm": pred}

    table = validity_table(preds, fs, outcomes, heads=("continue", "ratio"))
    row = table.filter((table["head"] == "continue") & (table["outcome"] == "continue")).row(
        0, named=True
    )
    assert row["events"] == 7 and row["auc"] == pytest.approx(1.0)
    assert row["within_user"] > 0.5
    lengths = table.filter(
        (table["head"] == "continue") & (table["outcome"] == "remaining_listens")
    )
    assert lengths["events"].item() == 7  # the censored session is dropped
    assert np.isnan(lengths["auc"].item())
    assert set(table["outcome"]) == {
        "continue",
        "remaining_listens",
        "remaining_minutes",
        "returns",
    }

    users = user_return_table(preds, fs, outcomes, heads=("continue",))
    activity = users.filter(users["method"] == "activity").row(0, named=True)
    assert activity["users"] == 2 and activity["return_rate"] == 0.5
    assert activity["auc"] == pytest.approx(1.0)  # the returning user has more listens
    no_return = session_outcomes(store, fs, session_gap_minutes=30)
    with pytest.raises(ValueError):
        user_return_table(preds, fs, no_return)

    h = HEAD_INDEX["continue"]
    for within in (False, True):
        deciles = decile_table(pred[:, h], fs.valid[:, h], outcomes, within_user=within)
        assert deciles["events"].sum() == 7
    by_length = length_table(preds, fs, outcomes)
    assert by_length["length_bin"].to_list() == [0, 1, 2]
    assert by_length["events"].sum() == len(fs.rows)
    # bin 0: rows 0 and 3 continue, row 8 is unknown; bin 1: rows 1, 2, 6; bin 2: rows 4, 7
    assert by_length["continue_rate"].to_list() == pytest.approx([1.0, 2 / 3, 0.0])
