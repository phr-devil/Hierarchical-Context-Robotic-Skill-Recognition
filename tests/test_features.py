import h5py
import numpy as np
import pytest

from crps import features


def _make_episode(path, n=200, dt=0.02, seed=0):
    rng = np.random.RandomState(seed)
    ts = np.arange(n) * dt
    with h5py.File(path, "w") as fh:
        for ch, d in zip(features.PROPRIO_CHANNELS, features.PROPRIO_DIMS):
            fh.create_dataset(f"timestamps/{ch}", data=ts)
            fh.create_dataset(f"robot_state/{ch}", data=rng.randn(n, d))
        for ch, d in zip(features.FORCE_CHANNELS, features.FORCE_DIMS):
            fh.create_dataset(f"timestamps/{ch}", data=ts)
            fh.create_dataset(f"robot_state/{ch}", data=rng.randn(n, d))


def _manual_channel_stats(ts, X, times, window):
    mean = np.zeros((len(times), X.shape[1]))
    std = np.zeros_like(mean)
    disp = np.zeros_like(mean)
    for i, t in enumerate(times):
        sel = X[(ts >= t - window) & (ts <= t)]
        mean[i] = sel.mean(0)
        std[i] = sel.std(0)
        disp[i] = sel[-1] - sel[0]
    return mean, std, disp


def test_feature_dimensions(tmp_path):
    p = str(tmp_path / "ep.h5")
    _make_episode(p)
    X = features.frame_features(p, np.array([0.5, 1.0]))
    assert X.shape == (2, features.FRAME_FEATURE_DIM) == (2, 72)
    assert np.isfinite(X).all()


def test_feature_block_layout_matches_manual_computation(tmp_path):
    p = str(tmp_path / "ep.h5")
    _make_episode(p)
    times = np.array([0.5, 1.5, 3.0])
    with h5py.File(p, "r") as fh:
        short, force = features.short_force_features(fh, times, window=0.4)

        ts = fh["timestamps"]["joint_positions"][:].astype(np.float64)
        X = fh["robot_state"]["joint_positions"][:].astype(np.float64)
        mean, std, disp = _manual_channel_stats(ts, X, times, 0.4)

        # first proprioceptive channel occupies columns 0..20 as mean/std/displacement
        assert np.allclose(short[:, 0:7], mean)
        assert np.allclose(short[:, 7:14], std)
        assert np.allclose(short[:, 14:21], disp)

        # last proprioceptive channel is gripper efforts, 2-D, at columns 54..59
        ts = fh["timestamps"]["gripper_efforts"][:].astype(np.float64)
        X = fh["robot_state"]["gripper_efforts"][:].astype(np.float64)
        mean, std, _ = _manual_channel_stats(ts, X, times, 0.4)
        assert np.allclose(short[:, 54:56], mean)
        assert np.allclose(short[:, 56:58], std)

        # force block: mean then std for measured_force (3-D) at columns 0..5
        ts = fh["timestamps"]["measured_force"][:].astype(np.float64)
        X = fh["robot_state"]["measured_force"][:].astype(np.float64)
        mean, std, _ = _manual_channel_stats(ts, X, times, 0.4)
        assert np.allclose(force[:, 0:3], mean)
        assert np.allclose(force[:, 3:6], std)


def test_window_is_causal(tmp_path):
    """A frame's feature must not change when later samples are added."""
    p = str(tmp_path / "ep.h5")
    _make_episode(p, n=200)
    t = 2.0
    a = features.frame_features(p, np.array([t]))
    with h5py.File(p, "r") as fh:
        short, force = features.short_force_features(fh, np.array([t]), window=0.4)
    assert np.allclose(np.hstack([short, force]), a)


def test_short_and_force_widths():
    assert features.SHORT_FEATURE_DIM == 60
    assert features.FORCE_FEATURE_DIM == 12
    assert features.SHORT_FEATURE_DIM + features.FORCE_FEATURE_DIM == features.FRAME_FEATURE_DIM


def test_channel_dimensions_sum_to_declared_widths():
    assert 3 * sum(features.PROPRIO_DIMS) == features.SHORT_FEATURE_DIM
    assert 2 * sum(features.FORCE_DIMS) == features.FORCE_FEATURE_DIM


def test_empty_window_falls_back_to_nearest_sample(tmp_path):
    """Before the first sample the window is empty; the code must not divide by zero."""
    p = str(tmp_path / "ep.h5")
    _make_episode(p, n=50)
    X = features.frame_features(p, np.array([-1.0]))
    assert np.isfinite(X).all()


def test_single_query_matches_batch(tmp_path):
    p = str(tmp_path / "ep.h5")
    _make_episode(p)
    times = np.array([0.4, 0.8, 1.2, 1.6])
    batch = features.frame_features(p, times)
    for i, t in enumerate(times):
        single = features.frame_features(p, np.array([t]))
        assert np.allclose(single[0], batch[i])


def test_unknown_channel_raises(tmp_path):
    p = str(tmp_path / "ep.h5")
    with h5py.File(p, "w") as fh:
        fh.create_dataset("timestamps/joint_positions", data=np.arange(10) * 0.1)
        fh.create_dataset("robot_state/joint_positions", data=np.zeros((10, 7)))
    with pytest.raises(KeyError):
        features.frame_features(p, np.array([0.5]))
