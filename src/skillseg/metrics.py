"""Scoring: one-to-one segment matching, cell and episode aggregation, decision.

The three definitions below are the ones numbered (7)-(9) in the manuscript.
They are implemented here exactly as the frozen scoring program implements them,
including two choices that are easy to get wrong and are therefore stated:

**(7) Cell score.**  For one ``(episode, object)`` cell, the ground-truth and
predicted label sequences are run-length encoded.  Each ground-truth run is
matched to at most one predicted run *of the same class*, chosen by highest
frame-index intersection-over-union, greedily, first come first served.  A match
counts only if that IoU is at least 0.5.  Then

    S = 2 TP / (2 TP + FP + FN)

where ``FP`` counts predicted runs left unmatched and ``FN`` counts ground-truth
runs left unmatched.  Runs never cross an object boundary or a background gap,
because the cell is cut at those points before encoding.

**(8) Episode score.**  A cell exists only if the episode actually contains
frames of that object, so an episode is never penalised for objects it does not
touch: ``S_e`` is the unweighted mean over the cells that exist,

    S_e = (1 / |O_e|) * sum_{o in O_e} S_{e,o}

and the arm's score is the mean of ``S_e`` over episodes, after the seeds are
averaged inside each episode.  Averaging cells first means a short visit to one
object weighs as much as a long one; that is a measurement choice, not a bug, and
the manuscript states it.

**(9) Paired effect.**  With ``P_e`` and ``A_e`` the seed-averaged episode scores,

    d_e = P_e - A_e,   d = mean_e d_e

and the 95% interval is the percentile interval of 4,000 bootstrap resamples of
the 37 episodes.  The episode is the resampling unit; seeds are averaged inside
it.  The 473 cells and the 3 x 37 seed-episode pairs are diagnostics and are never
treated as independent observations.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import Config

CellKey = Tuple[str, str]


def rle(sequence: np.ndarray) -> List[Tuple[int, int, int]]:
    """Run-length encoding: ``(label, start, end)`` with inclusive indices."""
    if len(sequence) == 0:
        return []
    out: List[Tuple[int, int, int]] = []
    start = 0
    for i in range(1, len(sequence) + 1):
        if i == len(sequence) or sequence[i] != sequence[start]:
            out.append((int(sequence[start]), start, i - 1))
            start = i
    return out


def segment_counts(truth: np.ndarray, pred: np.ndarray,
                   threshold: float = 0.5) -> Tuple[int, int, int]:
    """One-to-one, same-class, best-IoU matching.  Returns ``(tp, fp, fn)``."""
    gt_runs = rle(truth)
    pred_runs = rle(pred)
    used = [False] * len(pred_runs)
    tp = 0
    for label, gs, ge in gt_runs:
        best, best_iou = -1, 0.0
        for j, (plabel, ps, pe) in enumerate(pred_runs):
            if used[j] or plabel != label:
                continue
            inter = max(0, min(ge, pe) - max(gs, ps) + 1)
            union = (ge - gs + 1) + (pe - ps + 1) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best = iou, j
        if best >= 0 and best_iou >= threshold:
            used[best] = True
            tp += 1
    return tp, sum(1 for u in used if not u), len(gt_runs) - tp


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def cell_score(pred: np.ndarray, runs: Sequence[Tuple[str, np.ndarray]],
               labels: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Score one ``(episode, object)`` cell over all of its runs."""
    tp = fp = fn = 0
    for _, idx in runs:
        a, b, c = segment_counts(labels[idx], pred[idx], threshold)
        tp += a; fp += b; fn += c
    return {"f1": f1_from_counts(tp, fp, fn), "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def cell_scores(pred: np.ndarray, cells: Mapping[CellKey, Sequence[Tuple[str, np.ndarray]]],
                labels: np.ndarray, threshold: float = 0.5) -> Dict[CellKey, float]:
    return {key: cell_score(pred, runs, labels, threshold)["f1"] for key, runs in cells.items()}


def episode_scores(cell_f1: Mapping[CellKey, float]) -> Dict[str, float]:
    """Equation (8): unweighted mean over the cells that exist in the episode."""
    grouped: Dict[str, List[float]] = {}
    for (episode, _), value in cell_f1.items():
        grouped.setdefault(episode, []).append(value)
    return {ep: float(np.mean(values)) for ep, values in grouped.items()}


def pooled_score(pred: np.ndarray, runs: Sequence[Tuple[str, np.ndarray]],
                 labels: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Frame-pooled score: one count table over every run.

    Diagnostic only.  It weights long segments more than ``episode_scores`` does
    and is never used for the decision; the two are reported side by side so that
    a reader can see the difference rather than being shown one of them.
    """
    stat = cell_score(pred, runs, labels, threshold)
    return {"f1": stat["f1"], "tp": stat["tp"], "fp": stat["fp"], "fn": stat["fn"]}


def frame_accuracy(pred: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    return float((pred[mask] == labels[mask]).mean())


def per_class_f1(pred: np.ndarray, labels: np.ndarray, mask: np.ndarray,
                 n_classes: int) -> Dict[str, float]:
    """Frame-level per-class F1 over the evaluation frames.

    Frame-level, not segment-level: a class can score well here while every one of
    its segment boundaries is wrong.  The manuscript keeps the two apart.
    """
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(conf, (labels[mask], pred[mask]), 1)
    tp = np.diag(conf).astype(float)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    return {str(i): float(2 * tp[i] / max(2 * tp[i] + fp[i] + fn[i], 1)) for i in range(n_classes)}


# --------------------------------------------------------------- paired inference
def paired_effect(a_episode: Mapping[str, float], p_episode: Mapping[str, float],
                  episodes: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """Equation (9) point estimate, without the interval."""
    episodes = sorted(episodes if episodes is not None else a_episode.keys())
    a = np.array([a_episode[e] for e in episodes], dtype=float)
    p = np.array([p_episode[e] for e in episodes], dtype=float)
    d = p - a
    return {
        "episodes": list(episodes),
        "A": a.tolist(),
        "P": p.tolist(),
        "delta": d.tolist(),
        "mean": float(d.mean()),
        "A_mean": float(a.mean()),
        "P_mean": float(p.mean()),
        "sd": float(d.std(ddof=1)),
        "n_pos": int((d > 0).sum()),
        "n_tie": int((np.abs(d) < 1e-12).sum()),
        "n_episodes": int(len(d)),
    }


def bootstrap_interval(delta: Sequence[float], resamples: int, seed: int) -> Tuple[float, float]:
    """Percentile interval over episode resamples."""
    d = np.asarray(delta, dtype=float)
    rng = np.random.RandomState(seed)
    draws = d[rng.randint(0, len(d), size=(resamples, len(d)))].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def decide(deltas: Sequence[float], cfg: Config,
           bootstrap: Optional[Tuple[float, float]] = None) -> Dict[str, object]:
    """Apply the pre-registered rule and report both readings of the width clause.

    The rule as coded is ordered: a result that clears the effect floor, has a
    strictly positive interval and improves enough episodes is ``CONFIRMED``; a
    result whose interval lies entirely below the equivalence bound is
    ``NOT_CONFIRMED``; everything else is ``INCONCLUSIVE``.  The pre-registration
    also fixed a maximum interval width (2.0 percentage points), but that
    condition sits in the prose of the "otherwise" branch rather than in the
    ordered test.

    In the reported run the effect cleared all three ordered conditions while the
    interval was wider than 2.0 pt.  Both readings are therefore returned:
    ``verdict`` follows the coded order, ``verdict_strict`` additionally requires
    the width condition.  Neither is suppressed; the manuscript reports the pair.
    """
    from scipy import stats

    d = np.asarray(deltas, dtype=float)
    n = len(d)
    lo, hi = bootstrap if bootstrap is not None else bootstrap_interval(
        d, cfg.bootstrap_resamples, cfg.bootstrap_seed)
    delta = float(d.mean())
    width = hi - lo
    pos = int((d > 0).sum())
    tie = int((np.abs(d) < 1e-12).sum())

    t = stats.ttest_1samp(d, 0.0)
    if np.all(np.abs(d) < 1e-12):
        wilcoxon_p: Optional[float] = None
    else:
        try:
            wilcoxon_p = float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
        except ValueError:
            wilcoxon_p = None
    sign_p = float(stats.binomtest(pos, n - tie, 0.5, alternative="greater").pvalue) \
        if n - tie > 0 else 1.0

    confirm_delta = cfg.prereg["confirm_delta"]
    equiv_upper = cfg.prereg["equiv_upper"]
    min_improved = int(cfg.prereg["min_improved"])
    max_width = cfg.prereg["max_ci_width"]

    if delta >= confirm_delta and lo > 0 and pos >= min_improved:
        verdict = "CONFIRMED"
    elif hi < equiv_upper:
        verdict = "NOT_CONFIRMED"
    else:
        verdict = "INCONCLUSIVE"

    underpowered = bool(width > max_width)
    verdict_strict = "INCONCLUSIVE" if (underpowered and verdict == "CONFIRMED") else verdict

    return {
        "n_episodes": n,
        "delta": delta,
        "ci95": [lo, hi],
        "width": width,
        "t_stat": float(t.statistic),
        "t_p": float(t.pvalue),
        "wilcoxon_p": wilcoxon_p,
        "sign_pos": pos,
        "sign_tie": tie,
        "sign_p_one_sided": sign_p,
        "rules": {"confirm_delta": confirm_delta, "equiv_upper": equiv_upper,
                  "min_improved": min_improved, "max_ci_width": max_width,
                  "bootstrap_resamples": cfg.bootstrap_resamples,
                  "bootstrap_seed": cfg.bootstrap_seed},
        "underpowered": underpowered,
        "verdict": verdict,
        "verdict_strict": verdict_strict,
        "clause_conflict": bool(underpowered and verdict == "CONFIRMED"),
    }
