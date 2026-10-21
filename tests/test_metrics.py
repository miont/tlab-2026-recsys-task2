import numpy as np
import pytest

from recsys_rl import metrics

L2 = 1 / np.log2(3)  # DCG discount at rank 2


def test_relevance_metrics_by_hand():
    recs = np.array([[1, 2, 3], [4, 5, 6]])
    targets = [np.array([2, 9, 2]), np.array([7])]  # duplicates collapse to the set {2, 9}
    m = metrics.relevance_metrics(recs, targets, ks=(1, 3))
    assert m["recall@1"] == m["ndcg@1"] == m["map@1"] == 0.0
    # user 1: one hit at rank 2 out of 2 targets; user 2: no hits
    assert m["recall@3"] == pytest.approx(0.5 / 2)
    assert m["ndcg@3"] == pytest.approx(L2 / (1 + L2) / 2)
    assert m["map@3"] == pytest.approx((1 / 2) / 2 / 2)


def test_relevance_metrics_perfect_ranking():
    recs = np.array([[5, 6, 1, 2]])
    m = metrics.relevance_metrics(recs, [np.array([6, 5])], ks=(2, 4))
    assert m["recall@2"] == m["ndcg@2"] == m["map@2"] == pytest.approx(1.0)
    assert m["ndcg@4"] == pytest.approx(1.0)


def test_yambda_paper_metrics_by_hand():
    recs = np.array([[1, 2, 3], [4, 5, 6]])
    targets = [np.array([2, 9, 2]), np.array([7])]
    m = metrics.yambda_paper_metrics(recs, targets, ks=(3,))
    # recall divides by target events (3 and 1); NDCG is pooled: sum(DCG) / sum(IDCG)
    assert m["recall@3"] == pytest.approx((1 / 3) / 2)
    assert m["ndcg@3"] == pytest.approx(L2 / ((1 + L2 + 0.5) + 1))


def test_input_validation():
    with pytest.raises(ValueError, match="duplicate"):
        metrics.relevance_metrics(np.array([[1, 1]]), [np.array([1])], ks=(2,))
    with pytest.raises(ValueError, match="at least one target"):
        metrics.relevance_metrics(np.array([[1, 2]]), [np.array([], dtype=int)], ks=(2,))
    with pytest.raises(ValueError, match="out of range"):
        metrics.coverage(np.array([[1, 2]]), k=3, n_items=5)


def test_coverage_and_gini():
    recs = np.array([[1, 2], [1, 0]])
    assert metrics.coverage(recs, k=2, n_items=10) == pytest.approx(0.3)
    # exposure counts over a 3-item catalogue: sorted [1, 1, 2] -> (-2*1 + 0*1 + 2*2) / (3 * 4)
    assert metrics.gini(recs, k=2, n_items=3) == pytest.approx(1 / 6)
    assert metrics.gini(np.array([[0, 1], [2, 0], [1, 2]]), k=2, n_items=3) == pytest.approx(0.0)


def test_mean_pairwise_jaccard():
    recs = np.array([[1, 2], [2, 3], [4, 5]])
    assert metrics.mean_pairwise_jaccard(recs, k=2) == pytest.approx((1 / 3) / 3)
    same = np.tile(np.arange(10), (500, 1))
    assert metrics.mean_pairwise_jaccard(same, k=10, n_pairs=100) == pytest.approx(1.0)


def test_repeat_rate_and_item_values():
    recs = np.array([[1, 2], [3, 4]])
    histories = [np.array([2, 7]), np.array([], dtype=int)]
    assert metrics.repeat_rate(recs, histories, k=2) == pytest.approx(0.25)
    values = np.array([0.0, 10.0, 20.0, np.nan, 40.0])
    assert metrics.mean_item_value(recs, values, k=2) == pytest.approx(70 / 3)


def test_ild():
    emb = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    assert metrics.ild(np.array([[0, 1]]), emb, k=2) == pytest.approx(1.0)
    assert metrics.ild(np.array([[0, 1], [0, 2]]), emb, k=2, batch_size=1) == pytest.approx(0.5)


def test_roc_auc_and_average_precision():
    y = np.array([1, 0, 1, 0, 1])
    p = np.array([0.9, 0.8, 0.7, 0.3, 0.3])
    # pairs (pos, neg): 0.9>0.8, 0.9>0.3, 0.7<0.8, 0.7>0.3, 0.3<0.8, 0.3=0.3 -> 3.5 + ... / 6
    assert metrics.roc_auc(y, p) == pytest.approx((1 + 1 + 0 + 1 + 0 + 0.5) / 6)
    assert np.isnan(metrics.roc_auc([1, 1], [0.1, 0.2]))
    # thresholds 0.9, 0.8, 0.7, 0.3 (tie of a positive and a negative): P = 1, 1/2, 2/3, 3/5
    assert metrics.average_precision(y, p) == pytest.approx((1 + 2 / 3 + 3 / 5) / 3)
    assert metrics.average_precision([0, 1], [0.0, 1.0]) == 1.0
    assert np.isnan(metrics.average_precision([0, 0], [0.1, 0.2]))


def test_calibration():
    y = np.array([0.0, 1.0, 1.0, 1.0])
    p = np.array([0.05, 0.15, 0.95, 1.0])
    c = metrics.calibration_bins(y, p, n_bins=10)
    assert c["count"].tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 0, 2]
    assert c["mean_true"][[0, 1, 9]].tolist() == [0.0, 1.0, 1.0]
    assert np.isnan(c["mean_pred"][5])
    ece = metrics.expected_calibration_error(y, p, n_bins=10)
    assert ece == pytest.approx((0.05 + 0.85 + 2 * 0.025) / 4)
