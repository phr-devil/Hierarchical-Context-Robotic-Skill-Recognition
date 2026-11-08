"""72-dimensional frame features.

The feature set is deliberately limited to robot-state signals: proprioception
plus the wrist force/torque sensor. Over a causal window of ``W`` seconds it
stack, per proprioceptive channel, the window mean, the window standard
deviation and the net displacement (last minus first sample); over the two
force/torque channels only the mean and standard deviation are used, because a
torque offset is not informative about the skill in progress.

    5 proprioceptive channels x 3 statistics x channel width  =  60
    2 force/torque channels  x 2 statistics x channel width   =  12
                                                        total  =  72

Window statistics are computed on the raw sensor streams through cumulative
sums, so a single HDF5 read per episode serves every frame of that episode.
"""
import h5py
import numpy as np

PROPRIO_CHANNELS = [
    "joint_positions",
    "joint_velocities",
    "gripper_positions",
    "gripper_velocities",
    "gripper_efforts",
]
FORCE_CHANNELS = ["measured_force", "measured_torque"]

PROPRIO_DIMS = [7, 7, 2, 2, 2]
FORCE_DIMS = [3, 3]

DEFAULT_WINDOW = 0.4

SHORT_FEATURE_DIM = 3 * sum(PROPRIO_DIMS)   # 60
FORCE_FEATURE_DIM = 2 * sum(FORCE_DIMS)     # 12
FRAME_FEATURE_DIM = SHORT_FEATURE_DIM + FORCE_FEATURE_DIM  # 72


def _offsets(dims, trace_step):
    """Start offset and width of each channel inside a stacked feature block."""
    offsets = []
    a = 0
    for d in dims:
        offsets.append(a)
        a += trace_step * d
    return offsets


SHORT_OFFSETS = _offsets(PROPRIO_DIMS, 3)
FORCE_OFFSETS = _offsets(FORCE_DIMS, 2)


def _cumsum_pad(X):
    """Cumulative sum with a leading zero row, for O(1) window sums."""
    C = np.zeros((X.shape[0] + 1, X.shape[1]), dtype=np.float64)
    C[1:] = np.cumsum(X, axis=0)
    return C


def _window_bounds(ts, times, window):
    """Half-open sample index range [lo, hi) covering [t - window, t]."""
    lo = np.searchsorted(ts, times - window, "left")
    hi = np.searchsorted(ts, times, "right")
    return lo, hi


def short_force_features(h5, times, window=DEFAULT_WINDOW):
    """Compute the 60-dim short-window block and the 12-dim force block.

    Parameters
    ----------
    h5 : h5py.File
        An open REASSEMBLE episode file.
    times : array_like, shape (n,)
        Query timestamps in seconds.
    window : float
        Length of the causal window in seconds.

    Returns
    -------
    short : ndarray, shape (n, 60)
    force : ndarray, shape (n, 12)
    """
    times = np.asarray(times, dtype=np.float64)
    n = len(times)
    short = np.zeros((n, SHORT_FEATURE_DIM))
    force = np.zeros((n, FORCE_FEATURE_DIM))

    for ci, ch in enumerate(PROPRIO_CHANNELS):
        ts = h5["timestamps"][ch][:].astype(np.float64)
        X = h5["robot_state"][ch][:].astype(np.float64)
        C = _cumsum_pad(X)
        C2 = _cumsum_pad(X * X)
        lo, hi = _window_bounds(ts, times, window)
        nn = (hi - lo).astype(np.float64)
        nn[nn < 1] = 1
        mean = (C[hi] - C[lo]) / nn[:, None]
        meansq = (C2[hi] - C2[lo]) / nn[:, None]
        std = np.sqrt(np.clip(meansq - mean * mean, 0, None))
        last = X[np.clip(hi - 1, 0, len(X) - 1)]
        first = X[np.clip(lo, 0, len(X) - 1)]
        d = X.shape[1]
        o = SHORT_OFFSETS[ci]
        short[:, o:o + d] = mean
        short[:, o + d:o + 2 * d] = std
        short[:, o + 2 * d:o + 3 * d] = last - first

    for ci, ch in enumerate(FORCE_CHANNELS):
        ts = h5["timestamps"][ch][:].astype(np.float64)
        X = h5["robot_state"][ch][:].astype(np.float64)
        C = _cumsum_pad(X)
        C2 = _cumsum_pad(X * X)
        lo, hi = _window_bounds(ts, times, window)
        nn = (hi - lo).astype(np.float64)
        nn[nn < 1] = 1
        mean = (C[hi] - C[lo]) / nn[:, None]
        meansq = (C2[hi] - C2[lo]) / nn[:, None]
        std = np.sqrt(np.clip(meansq - mean * mean, 0, None))
        d = X.shape[1]
        o = FORCE_OFFSETS[ci]
        force[:, o:o + d] = mean
        force[:, o + d:o + 2 * d] = std

    return short, force


def frame_features(h5_path, times, window=DEFAULT_WINDOW):
    """72-dimensional features for ``times`` of a single episode file."""
    with h5py.File(h5_path, "r") as h5:
        short, force = short_force_features(h5, times, window)
    return np.nan_to_num(np.hstack([short, force]), nan=0.0)
