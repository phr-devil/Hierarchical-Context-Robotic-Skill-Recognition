import numpy as np
import pytest

from crps import common


def test_rle_roundtrip():
    seq = [0, 0, 1, 1, 1, 0]
    runs = common.rle(seq)
    assert runs == [(0, 0, 1), (1, 2, 4), (0, 5, 5)]
    rebuilt = []
    for label, s, e in runs:
        rebuilt.extend([label] * (e - s + 1))
    assert rebuilt == seq


def test_seg_counts_perfect_prediction():
    gt = np.array([0, 0, 0, 1, 1, 1])
    tp, fp, fn = common.seg_counts(gt, gt, 0.5)
    assert (tp, fp, fn) == (2, 0, 0)


def test_seg_counts_requires_label_agreement():
    gt = np.array([0, 0, 0, 1, 1, 1])
    pr = np.array([1, 1, 1, 1, 1, 1])
    tp, fp, fn = common.seg_counts(gt, pr, 0.5)
    assert tp == 1          # only the label-1 segment can match
    assert fp == 0
    assert fn == 1


def test_seg_counts_iou_threshold():
    gt = np.array([0, 0, 0, 0])
    pr = np.array([1, 1, 0, 0])
    # predicted label-0 segment [2,3] against ground truth [0,3]: IoU 2/4 = 0.5
    assert common.seg_counts(gt, pr, 0.5)[0] == 1
    assert common.seg_counts(gt, pr, 0.75)[0] == 0


def test_match_bnd_respects_tolerance_and_uniqueness():
    gt = [1.0, 2.0]
    tp, fp, fn = common.match_bnd(gt, [1.1, 5.0], 0.2)
    assert (tp, fp, fn) == (1, 1, 1)
    # one prediction may not absorb two ground-truth boundaries
    tp, fp, fn = common.match_bnd([1.0, 1.05], [1.02], 0.2)
    assert tp == 1
    assert fn == 1


def test_match_bnd_empty_cases():
    assert common.match_bnd([], [1.0], 0.1) == (0, 1, 0)
    assert common.match_bnd([1.0], [], 0.1) == (0, 0, 1)


def test_compress_and_lev():
    assert common.compress([0, 0, 1, 1, 0]) == [0, 1, 0]
    assert common.lev([0, 1, 0], [0, 1, 0]) == 0
    assert common.lev([0, 1, 0], [0, 1]) == 1
    assert common.lev([], [1, 2]) == 2


def test_prior_correct_recovers_uniform_shape():
    p = np.array([[0.7, 0.2, 0.1]])
    w = np.array([1.0, 2.0, 4.0])
    q = common.prior_correct(p, w)
    assert q.shape == p.shape
    assert np.isclose(q.sum(), 1.0)
    # dividing by a larger weight must not increase the posterior mass
    assert q[0, 2] < q[0, 0]


def test_pool_metrics_on_synthetic_sequences():
    n, k = 40, 3
    gt = np.array([0] * 10 + [1] * 10 + [2] * 20)
    pred = gt.copy()
    seqs = [("ep", np.arange(n))]
    t = np.arange(n) * 0.2
    y = np.zeros(n, dtype=int)
    st = common.episode_stats(pred, seqs, gt, t, k)
    pooled = common.pool(st)
    assert np.isclose(pooled["seg_f1"], 1.0)
    assert np.isclose(pooled["macro_f1"], 1.0)
    assert np.isclose(pooled["acc"], 1.0)
    assert np.isclose(pooled["edit"], 0.0)
    assert pooled["n_seq"] == 1


def test_pool_reports_boundary_metrics():
    k = 3
    gt = np.array([0] * 5 + [1] * 5)
    pred = np.array([0] * 5 + [1] * 5)
    t = np.arange(10) * 0.2
    st = common.episode_stats(pred, [("ep", np.arange(10))], gt, t, k)
    pooled = common.pool(st)
    assert np.isclose(pooled["bnd_F1@0.5"], 1.0)
    assert pooled["bnd_fp_per_min@0.5"] >= 0.0


@pytest.mark.parametrize("thr", [0.5, 0.75])
def test_seg_counts_threshold_monotone(thr):
    gt = np.array([0] * 8 + [1] * 2)
    pr = np.array([0] * 6 + [1] * 4)
    tp_lo = common.seg_counts(gt, pr, 0.5)[0]
    tp_hi = common.seg_counts(gt, pr, 0.75)[0]
    assert tp_hi <= tp_lo
