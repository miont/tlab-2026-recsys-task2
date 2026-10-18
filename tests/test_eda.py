import polars as pl

from recsys_rl import eda


def test_bin_index_and_labels():
    df = pl.DataFrame({"x": [-3, 0, 5, 6, 100, 101]})
    bins = df.select(eda.bin_index(pl.col("x"), [0, 5, 100]))["bin"].to_list()
    assert bins == [0, 0, 1, 2, 2, 3]
    assert eda.bin_labels([0, 5, 100]) == ["(-inf, 0]", "(0, 5]", "(5, 100]", "(100, inf]"]


def test_first_vs_repeat_flag():
    listens = pl.LazyFrame({"uid": [1, 1, 1, 2], "item_id": [7, 8, 7, 7]})
    flags = eda.with_repeat_flag(listens).collect()["repeat"].to_list()
    assert flags == [False, False, True, False]
