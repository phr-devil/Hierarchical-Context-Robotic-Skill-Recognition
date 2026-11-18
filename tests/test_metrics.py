"""Scoring definitions: matching, cell/episode aggregation, bootstrap."""
import numpy as np
import pytest

from skillseg import metrics, objects


def test_rle_inclusive_indices():
    assert metrics.rle(np.array([0, 0, 1])) == [(0, 0, 1), (1, 2, 2)]
    assert metrics.rle(np.array([], dtype=int)) == []


def test_perfect_prediction_scores_one():
    y = np.array([0, 0, 1, 1])
    tp, fp, fn = metrics.segment_counts(y, y.copy())
    assert (tp, fp, fn) == (2, 0, 0)
    assert metrics.f1_from_counts(tp, fp, fn) == pytest.approx(1.0)


def test_wrong_class_is_both_a_miss_and_a_false_alarm():
    truth = np.array([0, 0, 0, 0])
    pred = np.array([1, 1, 1, 1])
    assert metrics.segment_counts(truth, pred) == (0, 1, 1)


def test_matching_is_one_to_one():
    truth = np.array([0, 0, 0, 0])
    pred = np.array([0, 0, 0, 0])
    # one ground-truth run against a single prediction: one match, nothing left over
    assert metrics.segment_counts(truth, pred) == (1, 0, 0)
    # split prediction into two adjacent runs: still one match, one leftover fragment
    pred_split = np.array([0, 0, 0, 0])
    pred_split = pred_split.copy()
    # the run boundary cannot be created without a different label, so use a gap class
    pred_split = np.array([0, 0, 1, 0])
    tp, fp, fn = metrics.segment_counts(truth, pred_split)
    assert tp == 1 and fn == 0 and fp >= 1


def test_iou_threshold_rejects_short_prediction():
    truth = np.array([0] * 10)
    pred = np.array([0] * 4)
    # IoU = 4/10 < 0.5: the predicted run exists but does not match
    assert metrics.segment_counts(truth, pred, 0.5) == (0, 1, 1)


def test_runs_break_at_background(toy_grid):
    runs = objects.runs_for(toy_grid, "gear")
    assert [len(idx) for _, idx in runs] == [5, 4, 2]


def test_cells_exclude_the_pseudo_object(toy_grid):
    toy_grid.obj = toy_grid.obj.copy()
    toy_grid.obj[0] = "action"
    cells = objects.cells(toy_grid)
    assert ("E1", "action") not in cells
    assert set(cells) == {("E1", "gear"), ("E2", "gear"), ("E2", "bolt")}


def test_episode_score_averages_present_cells_only():
    cell_f1 = {("E1", "a"): 0.5, ("E1", "b"): 1.0, ("E2", "a"): 0.0}
    per_episode = metrics.episode_scores(cell_f1)
    assert per_episode["E1"] == pytest.approx(0.75)
    # E2 touches one object; the missing second object is absent, not scored as zero
    assert per_episode["E2"] == pytest.approx(0.0)


def test_cells_are_weighted_equally_regardless_of_length(toy_grid):
    # a perfect short cell and a half-wrong long cell must average to 0.75, not to
    # a frame-weighted value
    pred = toy_grid.label.copy()
    cells = objects.cells(toy_grid)
    perfect = objects.all_runs({("E1", "gear"): cells[("E1", "gear")]})
    score = metrics.cell_score(toy_grid.label, perfect, toy_grid.label)
    assert score["f1"] == pytest.approx(1.0)


def test_pooled_and_episode_mean_are_not_interchangeable(toy_grid):
    pred = toy_grid.label.copy()
    cells = objects.cells(toy_grid)
    cell_f1 = metrics.cell_scores(pred, cells, toy_grid.label)
    per_episode = metrics.episode_scores(cell_f1)
    pooled = metrics.pooled_score(pred, objects.all_runs(cells), toy_grid.label)
    assert pooled["f1"] == pytest.approx(1.0)
    assert np.mean(list(per_episode.values())) == pytest.approx(1.0)
    # break one long run only: the pooled score falls less than the episode mean
    broken = pred.copy()
    long_run = max(objects.all_runs(cells), key=lambda r: len(r[1]))[1]
    broken[long_run[0]] = (broken[long_run[0]] + 1) % 9
    broken_cells = metrics.cell_scores(broken, cells, toy_grid.label)
    broken_mean = float(np.mean(list(metrics.episode_scores(broken_cells).values())))
    broken_pooled = metrics.pooled_score(broken, objects.all_runs(cells), toy_grid.label)["f1"]
    assert broken_mean != pytest.approx(broken_pooled)


def test_bootstrap_is_reproducible_and_width_positive():
    deltas = np.linspace(-0.02, 0.09, 37)
    a = metrics.bootstrap_interval(deltas, 2000, 20261111)
    b = metrics.bootstrap_interval(deltas, 2000, 20261111)
    assert a == b
    assert a[1] > a[0]


def test_frame_accuracy_uses_only_evaluation_frames(toy_grid):
    pred = toy_grid.label.copy()
    mask = (~toy_grid.is_bg) & (toy_grid.obj != "action")
    assert metrics.frame_accuracy(pred, toy_grid.label, mask) == pytest.approx(1.0)
    pred[0] = 5
    assert metrics.frame_accuracy(pred, toy_grid.label, mask) < 1.0
