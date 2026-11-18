import numpy as np
import pytest

from crps import grid as gridmod, objects as objmod, protocol


def _toy_frames():
    # one episode, two segments separated by a 1.0 s gap
    ep = np.array(["e1"] * 6)
    t = np.array([0.4, 0.6, 0.8, 2.0, 2.2, 2.4])
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int16)
    pact = np.array(["pick", "pick", "pick", "place", "place", "place"])
    date = np.array(["2025-01-13"] * 6)
    return dict(ep=ep, t=t, y=y, pact=pact, date=date)


def test_grid_inserts_background_frames():
    g = gridmod.build_grid(_toy_frames(), stride=0.2)
    # gap 0.8 -> 2.0 with a 0.2 stride yields 1.0, 1.2, 1.4, 1.6, 1.8
    assert int(g["is_bg"].sum()) == 5
    assert len(g["t"]) == 11
    assert np.allclose(np.sort(g["t"]), np.sort(np.concatenate([
        _toy_frames()["t"], np.array([1.0, 1.2, 1.4, 1.6, 1.8])])))


def test_background_parent_is_inherited_or_unknown():
    g = gridmod.build_grid(_toy_frames(), stride=0.2)
    bg = g["is_bg"]
    # the gap sits between a "pick" and a "place" segment, so the parent is unknown
    assert set(g["pact"][bg].tolist()) == {gridmod.PARENT_UNKNOWN}


def test_causal_history_shape_and_padding():
    n, dim, w = 6, 3, 4
    feature = np.arange(n * dim, dtype=float).reshape(n, dim)
    ep = np.array(["e1"] * n)
    t = np.arange(n) * 0.2
    X = protocol.causal_history(feature, ep, t, w)
    assert X.shape == (n, w, dim)
    # first frame: only its own feature, left-padded with zeros
    assert np.allclose(X[0, :w - 1], 0.0)
    assert np.allclose(X[0, -1], feature[0])
    # from frame w-1 on the window is full and ends at the current frame
    assert np.allclose(X[w - 1], feature[:w])


def test_causal_history_ignores_future_frames():
    n, dim, w = 8, 2, 3
    feature = np.random.RandomState(0).randn(n, dim)
    ep = np.array(["e1"] * n)
    t = np.arange(n) * 0.2
    X = protocol.causal_history(feature, ep, t, w)
    changed = feature.copy()
    changed[-1] += 100.0
    Y = protocol.causal_history(changed, ep, t, w)
    assert np.allclose(X[:-1], Y[:-1])
    assert not np.allclose(X[-1], Y[-1])


def test_causal_history_never_crosses_episodes():
    feature = np.arange(8, dtype=float).reshape(8, 1)
    ep = np.array(["a"] * 4 + ["b"] * 4)
    t = np.array([0.0, 0.2, 0.4, 0.6, 0.0, 0.2, 0.4, 0.6])
    X = protocol.causal_history(feature, ep, t, 3)
    assert np.allclose(X[4, :2], 0.0)
    assert np.allclose(X[4, -1], feature[4])


def _toy_grid(n_per_ep=6):
    ep = np.array(["e1"] * n_per_ep + ["e2"] * n_per_ep)
    t = np.tile(np.arange(n_per_ep) * 0.2, 2)
    label = np.array([0, 0, 1, 1, 2, 2] * 2)
    obj = np.array(["A", "A", "A", "B", "B", "B"] * 2)
    is_bg = np.zeros(2 * n_per_ep, dtype=bool)
    return ep, t, label, obj, is_bg


def test_fold_masks_exclude_the_target_object():
    ep, t, label, obj, is_bg = _toy_grid()
    obj_eps = {"A": {"e1", "e2"}, "B": {"e1", "e2"}}
    train, test = protocol.fold_masks(ep, is_bg, obj, obj_eps, "A")
    assert all(obj[i] != "A" for i in train)
    assert len(test) == len(ep)
    obj_eps_small = {"A": {"e1"}, "B": {"e2"}}
    _, test_small = protocol.fold_masks(ep, is_bg, obj, obj_eps_small, "A")
    assert set(ep[test_small].tolist()) == {"e1"}


def test_fold_masks_drop_background():
    ep, t, label, obj, is_bg = _toy_grid()
    is_bg[3] = True
    obj_eps = {"A": {"e1"}, "B": {"e2"}}
    train, _ = protocol.fold_masks(ep, is_bg, obj, obj_eps, "A")
    assert 3 not in set(train.tolist())


def test_object_runs_returns_contiguous_runs():
    ep, t, label, obj, is_bg = _toy_grid()
    obj_eps = {"A": {"e1"}, "B": {"e2"}}
    runs = protocol.object_runs(ep, t, obj, is_bg, obj_eps, "A")
    assert len(runs) == 1
    key, idxs = runs[0]
    assert key.startswith("e1_")
    assert len(idxs) == 3


def test_transition_matrix_rows_sum_to_one():
    label = np.array([0, 0, 1, 1, 2, 2])
    seqs = [("e1", np.arange(6))]
    A = protocol.transition_matrix(label, seqs, 3)
    assert np.allclose(A.sum(1), 1.0)


def test_transition_matrix_is_smoothed():
    label = np.array([0, 1, 0, 1])
    seqs = [("e1", np.arange(4))]
    A = protocol.transition_matrix(label, seqs, 3)   # class 2 never observed
    assert A[2].sum() > 0
    assert np.all(A > 0)


def test_training_sequences_use_global_indices_and_skip_singletons():
    ep = np.array(["e1"] * 5)
    t = np.arange(5) * 0.2
    order = protocol.order_index(ep, t)
    label = np.zeros(5, dtype=int)
    train = np.array([0, 1, 4])          # frames 2,3 excluded
    seqs = protocol.training_sequences(ep, order, train, label)
    assert len(seqs) == 1
    _, idxs = seqs[0]
    assert idxs.tolist() == [0, 1, 4]


def test_decode_argmax_and_lag_shapes():
    n, k = 10, 3
    emission = np.zeros((n, k))
    emission[:, 0] = 1.0
    seqs = [("e1", np.arange(n))]
    A = np.full((k, k), 1.0 / k)
    assert np.all(protocol.decode(emission, seqs, A, "argmax") == 0)
    lag = protocol.decode(emission, seqs, A, "lag3")
    assert lag.shape == (n,)
    assert np.all(lag == 0)


def test_decode_rejects_unknown_decoder():
    emission = np.zeros((4, 2))
    with pytest.raises(ValueError):
        protocol.decode(emission, [("e", np.arange(4))], np.eye(2), "nope")


def test_object_grid_excludes_artefact_and_background():
    frames = dict(ep=np.array(["e1"] * 4), seg=np.array([0, 0, 1, 1]),
                  t=np.array([0.4, 0.6, 1.0, 1.2]))
    g = dict(ep=np.array(["e1"] * 5), t=np.array([0.4, 0.6, 0.8, 1.0, 1.2]),
             is_bg=np.array([False, False, True, False, False]))
    table = {("e1", 0): "BNC", ("e1", 1): "action"}
    obj_grid, episodes, objs, matched = objmod.build_object_grid(frames, g, table)
    assert matched == 4
    assert obj_grid[2] == objmod.UNKNOWN
    assert objs == ["BNC"]
    assert episodes["action"] == {"e1"}
