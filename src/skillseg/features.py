"""Frame features and causal stacking.

Each frame is described by 60 proprioceptive dimensions (mean, standard
deviation and net displacement over a 0.4 s causal window, for joint positions,
joint velocities, gripper positions, gripper velocities and gripper efforts) and
12 force/torque dimensions (mean and standard deviation of measured force and
torque over the same window).  No camera, audio, event or object-identity signal
enters the features.

The recurrent arms consume the last ``history_frames`` such vectors, left-padded
with zeros at episode start, so the representation is causal: a frame at time
``t`` never sees anything after ``t``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from .config import Config
from .data import Grid


def _cumulative(x: np.ndarray) -> np.ndarray:
    c = np.zeros((x.shape[0] + 1, x.shape[1]), dtype=np.float64)
    c[1:] = np.cumsum(x, axis=0)
    return c


def episode_sequences(grid: Grid) -> List[Tuple[str, np.ndarray]]:
    """Frame indices of every episode, ordered by time.

    The grid stores annotated frames first and background frames afterwards, so
    the stored order is not the temporal order.  Anything that builds windows or
    transition counts must use this ordering.
    """
    out: List[Tuple[str, np.ndarray]] = []
    for ep in sorted(set(grid.ep.tolist())):
        idx = np.where(grid.ep == ep)[0]
        out.append((ep, idx[np.argsort(grid.t[idx])]))
    return out


def frame_features(grid: Grid, cfg: Config, verbose: bool = False) -> np.ndarray:
    """72-dimensional frame features for every grid frame."""
    n = len(grid)
    short = np.zeros((n, 3 * sum(cfg.proprio_dims)), dtype=np.float64)
    force = np.zeros((n, 2 * sum(cfg.force_dims)), dtype=np.float64)

    short_off, acc = [], 0
    for d in cfg.proprio_dims:
        short_off.append(acc); acc += 3 * d
    force_off, acc = [], 0
    for d in cfg.force_dims:
        force_off.append(acc); acc += 2 * d

    by_episode: Dict[str, List[int]] = {}
    for i, ep in enumerate(grid.ep.tolist()):
        by_episode.setdefault(ep, []).append(i)

    for ep, idxs in by_episode.items():
        idxs = np.asarray(idxs)
        times = grid.t[idxs]
        with h5py.File(cfg.dataset.episode_path(ep), "r") as h:
            for ci, channel in enumerate(cfg.proprio_channels):
                ts = h["timestamps"][channel][:].astype(np.float64)
                x = h["robot_state"][channel][:].astype(np.float64)
                c1, c2 = _cumulative(x), _cumulative(x * x)
                lo = np.searchsorted(ts, times - cfg.window_s, "left")
                hi = np.searchsorted(ts, times, "right")
                count = np.maximum((hi - lo).astype(np.float64), 1.0)
                mean = (c1[hi] - c1[lo]) / count[:, None]
                meansq = (c2[hi] - c2[lo]) / count[:, None]
                std = np.sqrt(np.clip(meansq - mean * mean, 0.0, None))
                last = x[np.clip(hi - 1, 0, len(x) - 1)]
                first = x[np.clip(lo, 0, len(x) - 1)]
                d = x.shape[1]
                short[idxs, short_off[ci]:short_off[ci] + d] = mean
                short[idxs, short_off[ci] + d:short_off[ci] + 2 * d] = std
                short[idxs, short_off[ci] + 2 * d:short_off[ci] + 3 * d] = last - first
            for ci, channel in enumerate(cfg.force_channels):
                ts = h["timestamps"][channel][:].astype(np.float64)
                x = h["robot_state"][channel][:].astype(np.float64)
                c1, c2 = _cumulative(x), _cumulative(x * x)
                lo = np.searchsorted(ts, times - cfg.window_s, "left")
                hi = np.searchsorted(ts, times, "right")
                count = np.maximum((hi - lo).astype(np.float64), 1.0)
                mean = (c1[hi] - c1[lo]) / count[:, None]
                meansq = (c2[hi] - c2[lo]) / count[:, None]
                std = np.sqrt(np.clip(meansq - mean * mean, 0.0, None))
                d = x.shape[1]
                force[idxs, force_off[ci]:force_off[ci] + d] = mean
                force[idxs, force_off[ci] + d:force_off[ci] + 2 * d] = std

    short = np.nan_to_num(short, nan=0.0)
    force = np.nan_to_num(force, nan=0.0)
    out = np.nan_to_num(np.hstack([short, force]), nan=0.0).astype(np.float32)
    if verbose:
        print(f"frame features {out.shape} dtype={out.dtype}")
    return out


def causal_stack(features: np.ndarray, sequences: Sequence[Tuple[str, np.ndarray]],
                 history: int, out_path: Optional[str | Path] = None) -> np.ndarray:
    """Stack the last ``history`` frames of each episode into a tensor.

    Returns an ``(N, history, D)`` array; when ``out_path`` is given the array is
    written through a memmap so that the development split (~800 MB at
    ``N = 185,451``) does not have to fit in memory.
    """
    n, dim = features.shape
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        stacked = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                            shape=(n, history, dim))
    else:
        stacked = np.zeros((n, history, dim), dtype=np.float32)

    for _, idxs in sequences:
        idxs = np.asarray(idxs)
        for k in range(len(idxs)):
            lo = max(0, k - history + 1)
            sel = idxs[lo:k + 1]
            stacked[idxs[k], history - len(sel):] = features[sel]

    if out_path is not None:
        stacked.flush()
    return stacked
