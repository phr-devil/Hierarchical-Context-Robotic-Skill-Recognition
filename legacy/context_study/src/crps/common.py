"""Causal decoders and evaluation metrics.

All decoders are strictly causal: the output at frame ``i`` depends only on
emissions at frames ``0..i`` (plus a fixed decision lag, used to buy accuracy at
a known and reported delay). Sequences are passed as explicit index groups so
that a single global emission array can be decoded episode by episode.

Metric definitions match the paper:
  * segmental F1  -- segment matching by IoU threshold, same class label
  * boundary F1   -- boundary timestamp matching within a temporal tolerance
  * per-class F1  -- confusion-matrix F1 with IoU-matched segments
"""
import numpy as np
from collections import defaultdict

# Frame stride of the annotation grid, in seconds. Used only by the pooled
# episode helpers below, which convert spans from frames to minutes.
STRIDE = 0.2

# Temporal tolerances (seconds) for boundary matching; 0.2 s and 0.5 s are
# reported in the paper.
TOLS = (0.2, 0.5)


# --------------------------------------------------------------------------
# sequence grouping
# --------------------------------------------------------------------------
def seqs_by_seg(ep, seg, t):
    """Group frame indices by (episode, parent-segment) and sort by time."""
    groups = defaultdict(list)
    for i, (a, b) in enumerate(zip(ep, seg)):
        groups[(a, b)].append(i)
    out = []
    for key, idxs in groups.items():
        idxs = np.array(idxs)
        out.append((key, idxs[np.argsort(t[idxs])]))
    return out


def seqs_by_ep(ep, t):
    """Group frame indices by episode and sort by time."""
    groups = defaultdict(list)
    for i, a in enumerate(ep):
        groups[a].append(i)
    out = []
    for key, idxs in groups.items():
        idxs = np.array(idxs)
        out.append((key, idxs[np.argsort(t[idxs])]))
    return out


# --------------------------------------------------------------------------
# decoders
# --------------------------------------------------------------------------
def prior_correct(p, w):
    """Undo the balanced class weighting that the emission model was trained with.

    ``w`` are the per-class weights (K,) estimated on the training fold only.
    """
    q = p / w[None, :]
    s = q.sum(1, keepdims=True)
    s[s <= 0] = 1.0
    return q / s


def dec_hmm_filter(P_, A, groups, out):
    """Forward filtering (no smoothing): the online reference decoder."""
    for idxs in groups:
        p = P_[idxs]
        L = len(p)
        a = p[0] / max(p[0].sum(), 1e-300)
        o = np.empty(L, dtype=int)
        o[0] = a.argmax()
        for i in range(1, L):
            a = (a @ A) * p[i]
            ssum = a.sum()
            a = a / ssum if ssum > 0 else p[i]
            o[i] = a.argmax()
        out[idxs] = o


def viterbi_lag(P_, A, groups, out, lag):
    """Fixed-lag Viterbi decoding.

    Frames are committed once they are ``lag`` steps behind the frontier, using
    the back-pointer of the current best state. The tail (last ``lag`` frames)
    is filled from the final backtracked path, so the caller must treat the last
    ``lag`` predictions of a sequence as non-causal.
    """
    logA = np.log(np.maximum(A, 1e-12))
    for idxs in groups:
        p = P_[idxs]
        L = len(p)
        logp = np.log(np.maximum(p, 1e-300))
        back = np.zeros((L, P_.shape[1]), dtype=np.int16)
        assigned = np.zeros(L, dtype=bool)
        delta = logp[0].copy()
        for i in range(1, L):
            cand = delta[:, None] + logA
            back[i] = cand.argmax(0)
            delta = cand.max(0) + logp[i]
            if i - lag >= 0:
                st = int(delta.argmax())
                for k in range(i, i - lag, -1):
                    st = int(back[k, st])
                out[idxs[i - lag]] = st
                assigned[i - lag] = True
        path = np.zeros(L, dtype=int)
        st = int(delta.argmax())
        path[L - 1] = st
        for i in range(L - 1, 0, -1):
            st = int(back[i, st])
            path[i - 1] = st
        for i in range(L):
            if not assigned[i]:
                out[idxs[i]] = path[i]


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def rle(seq):
    """Run-length encode: list of (label, start, end) with inclusive ends."""
    if len(seq) == 0:
        return []
    out = []
    s = 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[s]:
            out.append((int(seq[s]), s, i - 1))
            s = i
    return out


def seg_counts(gt, pr, thr=0.5):
    """IoU-matched segment counts: (true positives, false positives, misses).

    A ground-truth segment is a true positive if some unused predicted segment
    of the same class reaches IoU >= ``thr``.
    """
    g = rle(gt)
    p = rle(pr)
    used = [False] * len(p)
    tp = 0
    for (gl, gs, ge) in g:
        best = -1
        biou = 0.0
        for pi, (pl, ps, pe) in enumerate(p):
            if used[pi] or pl != gl:
                continue
            inter = max(0, min(ge, pe) - max(gs, ps) + 1)
            union = (ge - gs + 1) + (pe - ps + 1) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > biou:
                biou = iou
                best = pi
        if best >= 0 and biou >= thr:
            used[best] = True
            tp += 1
    return tp, sum(1 for u in used if not u), len(g) - tp


def compress(seq):
    """Collapse consecutive duplicates: the label sequence used for edit distance."""
    if len(seq) == 0:
        return []
    o = [int(seq[0])]
    for i in range(1, len(seq)):
        if seq[i] != seq[i - 1]:
            o.append(int(seq[i]))
    return o


def lev(a, b):
    """Levenshtein distance between two label sequences."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[-1]


def match_bnd(gt_t, pr_t, tol):
    """Match predicted boundary times to ground-truth times within ``tol`` seconds."""
    used = [False] * len(pr_t)
    tp = 0
    for g in gt_t:
        best, bd = -1, 1e9
        for j, p_ in enumerate(pr_t):
            if used[j]:
                continue
            dd = abs(p_ - g)
            if dd <= tol and dd < bd:
                bd, best = dd, j
        if best >= 0:
            used[best] = True
            tp += 1
    return tp, len(pr_t) - tp, len(gt_t) - tp


def episode_stats(pred, seqs, y, t, K):
    """Per-sequence metric accumulators (confusion, segment counts, edit, boundaries)."""
    st = {}
    for key, idxs in seqs:
        e_ = key[0] if isinstance(key, (tuple, list)) else key
        if e_ not in st:
            st[e_] = dict(conf=np.zeros((K, K), dtype=np.int64),
                          tp50=0, fp50=0, fn50=0, tp75=0, fp75=0, fn75=0,
                          edit_sum=0.0, n_seq=0, span=0.0,
                          bnd={tol: [0, 0, 0] for tol in TOLS})
        s = st[e_]
        gt = y[idxs]
        pr = pred[idxs]
        tt = t[idxs]
        np.add.at(s["conf"], (gt, pr), 1)
        a, b, c = seg_counts(gt, pr, 0.5)
        s["tp50"] += a
        s["fp50"] += b
        s["fn50"] += c
        a, b, c = seg_counts(gt, pr, 0.75)
        s["tp75"] += a
        s["fp75"] += b
        s["fn75"] += c
        cg, cp = compress(gt), compress(pr)
        s["edit_sum"] += lev(cg, cp) / max(len(cg), len(cp), 1)
        s["n_seq"] += 1
        s["span"] += (tt[-1] - tt[0] + STRIDE)
        bi = [i for i in range(1, len(gt)) if gt[i] != gt[i - 1]]
        pi = [i for i in range(1, len(pr)) if pr[i] != pr[i - 1]]
        gbt = [tt[i] for i in bi]
        pbt = [tt[i] for i in pi]
        for tol in TOLS:
            a, b, c = match_bnd(gbt, pbt, tol)
            s["bnd"][tol][0] += a
            s["bnd"][tol][1] += b
            s["bnd"][tol][2] += c
    return st


def pool(st, keys=None):
    """Pool per-sequence accumulators into the reported metrics."""
    keys = list(st) if keys is None else list(keys)
    K = st[keys[0]]["conf"].shape[0]
    conf = np.zeros((K, K), dtype=np.int64)
    tp50 = fp50 = fn50 = tp75 = fp75 = fn75 = 0
    es = 0.0
    ns = 0
    span = 0.0
    bnd = {tol: [0, 0, 0] for tol in TOLS}
    for k in keys:
        s = st[k]
        conf += s["conf"]
        tp50 += s["tp50"]
        fp50 += s["fp50"]
        fn50 += s["fn50"]
        tp75 += s["tp75"]
        fp75 += s["fp75"]
        fn75 += s["fn75"]
        es += s["edit_sum"]
        ns += s["n_seq"]
        span += s["span"]
        for tol in TOLS:
            for q in range(3):
                bnd[tol][q] += s["bnd"][tol][q]
    tp_c = np.diag(conf).astype(float)
    fp_c = conf.sum(0) - tp_c
    fn_c = conf.sum(1) - tp_c
    f1c = 2 * tp_c / np.maximum(2 * tp_c + fp_c + fn_c, 1)
    out = dict(macro_f1=float(np.mean(f1c)),
               per_class_f1={i: round(float(f1c[i]), 4) for i in range(K)},
               acc=float(tp_c.sum() / max(conf.sum(), 1)),
               seg_f1=2 * tp50 / max(2 * tp50 + fp50 + fn50, 1),
               seg_f1_iou75=2 * tp75 / max(2 * tp75 + fp75 + fn75, 1),
               edit=es / max(ns, 1), n_seq=ns, span_min=span / 60)
    for tol in TOLS:
        a, b, c = bnd[tol]
        pr_ = a / max(a + b, 1)
        rc = a / max(a + c, 1)
        out[f"bnd_P@{tol}"] = pr_
        out[f"bnd_R@{tol}"] = rc
        out[f"bnd_F1@{tol}"] = 2 * pr_ * rc / max(pr_ + rc, 1e-9)
        out[f"bnd_fp_per_min@{tol}"] = b / max(span / 60, 1e-9)
    return out
