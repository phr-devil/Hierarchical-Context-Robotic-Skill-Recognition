"""Decoding: prior correction, transition estimation, fixed-lag Viterbi."""
import numpy as np
import pytest

from skillseg import decode


def test_prior_correction_renormalises():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    corrected = decode.prior_correct(p, np.full(3, 1 / 3))
    assert np.allclose(corrected.sum(axis=1), 1.0)
    assert np.allclose(corrected, p)


def test_prior_correction_divides_inverse_frequency_weights():
    # The larger inverse-frequency weight belongs to the rarer class.
    p = np.array([[0.5, 0.5]])
    weights = np.array([10.0, 1.0])
    corrected = decode.prior_correct(p, weights)
    assert corrected[0, 1] > corrected[0, 0]


def test_prior_correction_uses_class_counts_not_inverse_counts():
    labels = np.array([0, 0, 0, 0, 1])
    weights = decode.class_weights(labels, 2)
    corrected = decode.prior_correct(np.array([[0.5, 0.5]]), weights)
    assert np.allclose(corrected, np.array([[0.8, 0.2]]))


def test_class_weights_are_inverse_frequency():
    labels = np.array([0, 0, 0, 1, 1, 2])
    w = decode.class_weights(labels, 3)
    # counts are 3, 2, 1 over 6 frames and 3 classes: 6 / (3 * count)
    assert w[0] == pytest.approx(6 / 9)
    assert w[1] == pytest.approx(1.0)
    assert w[2] == pytest.approx(2.0)


def test_transition_matrix_rows_sum_to_one():
    labels = np.array([0, 0, 1, 1, 2, 0])
    seqs = [("E", np.arange(6))]
    t = decode.transition_matrix(labels, seqs, 3, alpha=1.0)
    assert np.allclose(t.sum(axis=1), 1.0)


def test_background_label_lands_on_the_last_row():
    """Documents inherited behaviour: -1 indexes the last row or column.

    The frozen scoring program passed the full label array, background included,
    so outgoing transitions of background frames were added to the last skill.
    This is a defect that the reported numbers contain; it is pinned here so that
    it cannot be changed silently.
    """
    labels = np.array([2, -1, 0])
    seqs = [("E", np.arange(3))]
    t = decode.transition_matrix(labels, seqs, 3, alpha=0.0)
    # (2 -> -1) was counted as (2 -> 2) and (-1 -> 0) as (2 -> 0): the last skill
    # row ends up holding both a self-loop and an edge that no class-2 frame made
    assert t[2, 2] == pytest.approx(0.5)
    assert t[2, 0] == pytest.approx(0.5)


def test_masked_transition_ignores_background_on_both_axes():
    labels = np.array([0, 0, -1, 1, 1])
    seqs = [("E", np.arange(len(labels)))]
    frozen = decode.transition_matrix(labels, seqs, 2, alpha=1.0)
    masked = decode.transition_matrix_masked(labels, seqs, 2, alpha=1.0)
    assert np.allclose(frozen, [[0.5, 0.5], [0.25, 0.75]])
    assert np.allclose(masked, [[2 / 3, 1 / 3], [1 / 3, 2 / 3]])
    assert not np.allclose(masked, frozen)


def test_masked_transition_agrees_with_frozen_without_invalid_labels():
    labels = np.array([0, 1, 1, 2, 0, 2])
    seqs = [("A", np.arange(3)), ("B", np.arange(3, 6))]
    for alpha in (0.0, 1.0):
        expected = decode.transition_matrix(labels, seqs, 3, alpha)
        actual = decode.transition_matrix_masked(labels, seqs, 3, alpha)
        assert np.array_equal(actual, expected)


def test_masked_transition_rejects_other_out_of_range_labels():
    labels = np.array([0, 99, 1, -1, 0])
    seqs = [("E", np.arange(len(labels)))]
    masked = decode.transition_matrix_masked(labels, seqs, 2, alpha=1.0)
    assert np.allclose(masked, np.full((2, 2), 0.5))


def test_viterbi_removes_a_single_frame_flicker():
    k = 3
    posteriors = np.tile(np.array([0.90, 0.06, 0.04]), (40, 1))
    posteriors[20] = np.array([0.44, 0.52, 0.04])   # one frame leans the wrong way
    transition = np.full((k, k), 0.1) / (k - 1)
    np.fill_diagonal(transition, 0.9)
    pred = decode.decode_episodes(posteriors, [("E", np.arange(40))], transition, lag=3)
    assert (pred == 0).all()


def test_viterbi_follows_sustained_evidence():
    k = 3
    truth = np.array([0] * 30 + [1] * 30 + [2] * 30)
    posteriors = np.full((len(truth), k), 0.025)
    posteriors[np.arange(len(truth)), truth] = 0.95
    transition = np.full((k, k), 0.15) / (k - 1)
    np.fill_diagonal(transition, 0.85)
    pred = decode.decode_episodes(posteriors, [("E", np.arange(len(truth)))], transition, lag=3)
    assert (pred == truth).mean() > 0.95


def test_lag_commits_every_frame():
    posteriors = np.tile(np.eye(3)[0], (30, 1))
    transition = np.full((3, 3), 1 / 3)
    for lag in (1, 3):
        pred = decode.decode_episodes(posteriors, [("E", np.arange(30))], transition, lag)
        assert len(pred) == 30
        assert (pred == 0).all()


def test_episodes_are_decoded_independently():
    # two episodes with opposite evidence must not influence each other
    k = 3
    a = np.tile(np.eye(k)[0], (20, 1))
    b = np.tile(np.eye(k)[1], (20, 1))
    posteriors = np.vstack([a, b])
    seqs = [("A", np.arange(20)), ("B", np.arange(20, 40))]
    pred = decode.decode_episodes(posteriors, seqs, np.full((k, k), 1 / k), 2)
    assert (pred[:20] == 0).all() and (pred[20:] == 1).all()
