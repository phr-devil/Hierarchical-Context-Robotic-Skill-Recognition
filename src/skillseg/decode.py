"""Decoding: class-prior correction and fixed-lag Viterbi.

Two post-processing steps sit between the posteriors and the scored sequence, and
both are held fixed across arms:

* the posteriors are divided by the training class frequencies (a prior
  correction, not a threshold);
* a Viterbi pass runs over each episode with a three-frame lag, so a frame is
  committed only after three further frames have been observed.  That keeps the
  decoder causal at the 0.6 s nominal wait while removing isolated flickers.

Both use statistics estimated on the development split only: the class weights
from development frame counts, the transition matrix from development sequences.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Inverse-frequency weights over non-background frames."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    return len(labels) / (n_classes * np.maximum(counts, 1.0))


def prior_correct(posteriors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Divide by class prior and renormalise, row by row."""
    q = posteriors / weights[None, :]
    total = q.sum(axis=1, keepdims=True)
    total[total <= 0] = 1.0
    return q / total


def transition_matrix(labels: np.ndarray, sequences: Sequence[Tuple[str, np.ndarray]],
                      n_classes: int, alpha: float = 1.0) -> np.ndarray:
    """Row-normalised transition counts with additive smoothing.

    .. warning::
       ``labels`` is the full grid label array, in which background frames carry
       ``-1``.  Negative indexing maps those frames onto the **last** skill row,
       so that row also accumulates the outgoing transitions of background
       frames.  This is the behaviour of the frozen implementation and it is
       preserved here deliberately: the reported scores were produced with it.
       It is a defect, not a modelling choice, and it is recorded as such in
       ``README.md`` and ``docs/SCORING.md``.  Do not "fix" it and then compare
       the result against the published numbers.

    Pass a label array without negative entries if a clean matrix is wanted; that
    variant was never used for any reported number.
    """
    counts = np.full((n_classes, n_classes), float(alpha))
    for _, idx in sequences:
        y = labels[idx]
        for a, b in zip(y[:-1], y[1:]):
            counts[int(a), int(b)] += 1.0
    row = counts.sum(axis=1, keepdims=True)
    row[row <= 0] = 1.0
    return counts / row


def viterbi_lag(posteriors: np.ndarray, transition: np.ndarray,
                groups: Sequence[np.ndarray], out: np.ndarray, lag: int) -> np.ndarray:
    """Fixed-lag Viterbi; writes committed labels into ``out`` in place.

    ``groups`` holds the frame indices of each episode in temporal order.  Frames
    closer than ``lag`` to the end of an episode are committed by the final
    backtrack, so the whole episode is labelled.
    """
    log_transition = np.log(np.maximum(transition, 1e-12))
    for idx in groups:
        p = posteriors[idx]
        length = len(p)
        if length == 0:
            continue
        logp = np.log(np.maximum(p, 1e-300))
        back = np.zeros((length, posteriors.shape[1]), dtype=np.int16)
        assigned = np.zeros(length, dtype=bool)
        delta = logp[0].copy()
        for i in range(1, length):
            cand = delta[:, None] + log_transition
            back[i] = cand.argmax(axis=0)
            delta = cand.max(axis=0) + logp[i]
            if i - lag >= 0:
                state = int(delta.argmax())
                for k in range(i, i - lag, -1):
                    state = int(back[k, state])
                out[idx[i - lag]] = state
                assigned[i - lag] = True
        path = np.zeros(length, dtype=int)
        state = int(delta.argmax())
        path[length - 1] = state
        for i in range(length - 1, 0, -1):
            state = int(back[i, state])
            path[i - 1] = state
        for i in range(length):
            if not assigned[i]:
                out[idx[i]] = path[i]
    return out


def decode_episodes(posteriors: np.ndarray, sequences: Sequence[Tuple[str, np.ndarray]],
                    transition: np.ndarray, lag: int) -> np.ndarray:
    """Decode every episode independently and return one label per frame."""
    pred = np.zeros(posteriors.shape[0], dtype=np.int64)
    viterbi_lag(posteriors, transition, [idx for _, idx in sequences], pred, lag)
    return pred
