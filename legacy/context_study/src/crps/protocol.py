"""Leave-one-object-out protocol and per-fold data selection.

The evaluation protocol is target-object frame exclusion, not unseen-object
generalisation: a fold holds out one assembly object, removes that object's
labelled frames from training, and scores only those frames at test time. The
object's episodes still contribute their other objects' frames, and the frames
are scored in the context of the full episode timeline.

Every quantity that a model consumes -- scalers, class priors, transition
counts, gate parameters -- is estimated inside the fold on the remaining
sixteen objects only.
"""
import numpy as np

from . import common


def causal_history(feature, ep, t, window_frames):
    """Stack a causal window of frame features per frame.

    ``window_frames`` is the history length in frames (3 s at a 0.2 s stride is
    15). Frames near the start of an episode are left-padded with zeros, so the
    tensor never uses a frame from the future or from another episode.
    """
    n = feature.shape[0]
    out = np.zeros((n, window_frames, feature.shape[1]), dtype=np.float32)
    for _, idxs in common.seqs_by_ep(ep, t):
        L = len(idxs)
        for kk in range(L):
            lo = max(0, kk - window_frames + 1)
            sel = idxs[lo:kk + 1]
            out[idxs[kk], window_frames - len(sel):] = feature[sel]
    return out


def order_index(ep, t):
    """Episode id -> frame indices sorted by time."""
    return {k: idxs for k, idxs in common.seqs_by_ep(ep, t)}


def fold_masks(ep, is_bg, obj_grid, obj_episodes, obj):
    """Training frame indices and test frame indices for one object fold.

    Training uses every labelled frame whose object differs from the held-out
    one; testing applies only to episodes that contain the held-out object.
    """
    test_frames = np.where(np.isin(ep, list(obj_episodes[obj])))[0]
    train = np.where((~is_bg) & (obj_grid != obj))[0]
    return train, test_frames


def training_sequences(ep, order, train, label):
    """Episode sequences restricted to training frames, with global indices.

    Indices must stay global: the labels and transition counts of a fold are
    indexed by the same frame numbers, so slicing with local positions would
    silently read frames from other episodes.
    """
    keep = set(int(x) for x in train)
    seqs = []
    for k in np.unique(ep[train]):
        idxs = [int(i) for i in order[k] if int(i) in keep]
        if len(idxs) >= 2:
            seqs.append((k, np.array(idxs)))
    return seqs


def object_runs(ep, t, obj_grid, is_bg, obj_episodes, obj):
    """Maximal contiguous runs of held-out-object frames, per episode."""
    runs = []
    for e in obj_episodes[obj]:
        idxs = np.where(ep == e)[0]
        idxs = idxs[np.argsort(t[idxs])]
        keep = (obj_grid[idxs] == obj) & (~is_bg[idxs])
        i = 0
        while i < len(idxs):
            if keep[i]:
                j = i
                while j < len(idxs) and keep[j]:
                    j += 1
                runs.append((f"{e}_{i}", idxs[i:j]))
                i = j
            else:
                i += 1
    return runs


def episode_sequences(ep, t, obj_episodes, obj):
    """Full-timeline sequences of the episodes that contain ``obj``."""
    out = []
    for e in obj_episodes[obj]:
        idxs = np.where(ep == e)[0]
        idxs = idxs[np.argsort(t[idxs])]
        out.append((e, idxs))
    return out


def transition_matrix(label, seqs, n_classes, alpha=1.0):
    """Laplace-smoothed skill transition matrix estimated on ``seqs`` only."""
    A = np.full((n_classes, n_classes), alpha)
    for _, idxs in seqs:
        yy = label[idxs]
        for a_, b_ in zip(yy[:-1], yy[1:]):
            A[int(a_), int(b_)] += 1
    s = A.sum(1, keepdims=True)
    s[s <= 0] = 1.0
    return A / s


def class_weights(y, n_classes):
    """Per-class weights equivalent to scikit-learn's ``class_weight='balanced'``."""
    cnt = np.bincount(y, minlength=n_classes).astype(float)
    return len(y) / (n_classes * np.maximum(cnt, 1))


def decode(emission, seqs, A, decoder, lag=3, out_size=None):
    """Decode emissions over ``seqs`` into a global prediction array."""
    n = emission.shape[0] if out_size is None else out_size
    pred = np.zeros(n, dtype=int)
    for _, idxs in seqs:
        if decoder == "filter":
            common.dec_hmm_filter(emission, A, [idxs], pred)
        elif decoder in ("lag", "lag3"):
            common.viterbi_lag(emission, A, [idxs], pred, lag)
        elif decoder == "argmax":
            pred[idxs] = emission[idxs].argmax(1)
        else:
            raise ValueError(f"unknown decoder: {decoder}")
    return pred


def boundary_f1_full(pred, epseq, label, t, tol=0.5):
    """Boundary F1 over the full episode timeline.

    The held-out object's frames are isolated runs, so they contain almost no
    internal boundaries; boundary quality is therefore measured on the complete
    timeline (all objects plus background), as a decoder-level descriptor.
    """
    tp = fp = fn = 0
    for _, idxs in epseq:
        g = label[idxs].astype(int)
        p = pred[idxs].astype(int)
        tt = t[idxs]
        bi = [i for i in range(1, len(g)) if g[i] != g[i - 1]]
        pi = [i for i in range(1, len(p)) if p[i] != p[i - 1]]
        a, b, c = common.match_bnd([tt[i] for i in bi], [tt[i] for i in pi], tol)
        tp += a
        fp += b
        fn += c
    pr_ = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    return 2 * pr_ * rc / max(pr_ + rc, 1e-9)


def latency_full(pred, epseq, label, t, wait, tol=1.0):
    """Detection latency decomposition over the full episode timeline.

    ``detected - true`` is the signed boundary-time error; adding the fixed
    decision wait gives the time at which the label would actually have been
    available. The absolute error and the wait are not added together, because
    they are not independent.
    """
    deltas = []
    for _, idxs in epseq:
        g = label[idxs].astype(int)
        p = pred[idxs].astype(int)
        tt = t[idxs]
        bi = [i for i in range(1, len(g)) if g[i] != g[i - 1]]
        pi = [i for i in range(1, len(p)) if p[i] != p[i - 1]]
        gbt = [tt[i] for i in bi]
        pbt = [tt[i] for i in pi]
        for gb in gbt:
            best = None
            bd = 1e9
            for pb in pbt:
                dd = abs(pb - gb)
                if dd <= tol and dd < bd:
                    bd = dd
                    best = pb
            if best is not None:
                deltas.append(float(best - gb))
    d = np.array(deltas)
    actual = d + wait
    return dict(n=int(len(d)),
                boundary_time_error_signed_mean=float(d.mean()) if len(d) else 0.0,
                boundary_time_error_abs_mean=float(np.abs(d).mean()) if len(d) else 0.0,
                boundary_time_error_abs_p90=float(np.percentile(np.abs(d), 90)) if len(d) else 0.0,
                fixed_wait_s=wait,
                actual_output_signed_mean=float(actual.mean()) if len(d) else 0.0,
                actual_output_abs_mean=float(np.abs(actual).mean()) if len(d) else 0.0,
                actual_output_abs_p90=float(np.percentile(np.abs(actual), 90)) if len(d) else 0.0)


def transitions_full(pred, epseq, label, pairs):
    """Recall of named skill transitions over the full episode timeline.

    ``pairs`` is a list of ``(from_index, to_index, tag)``; a transition counts
    as detected when the target class appears within two frames of the switch.
    """
    out = {}
    for frm, to, tag in pairs:
        det = 0
        total = 0
        for _, idxs in epseq:
            g = label[idxs].astype(int)
            p = pred[idxs].astype(int)
            for i in range(1, len(g)):
                if g[i - 1] == frm and g[i] == to:
                    total += 1
                    lo = max(0, i - 2)
                    hi = min(len(p), i + 3)
                    if to in p[lo:hi]:
                        det += 1
        out[tag] = (det, total)
    return out


def evaluate_runs(pred, runs, label, t, n_classes, iou=(0.5, 0.75), tolerances=(0.2, 0.5)):
    """Segmental and boundary metrics over the held-out object's frame runs.

    Returns the metric dict and the 9x9 confusion matrix that produced it.
    """
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    tp50 = fp50 = fn50 = 0
    tp75 = fp75 = fn75 = 0
    bnd = {tol: [0, 0, 0] for tol in tolerances}
    for _, idxs in runs:
        g = label[idxs].astype(int)
        p = pred[idxs].astype(int)
        tt = t[idxs]
        np.add.at(conf, (g, p), 1)
        a, b, c = common.seg_counts(g, p, iou[0])
        tp50 += a
        fp50 += b
        fn50 += c
        a, b, c = common.seg_counts(g, p, iou[1])
        tp75 += a
        fp75 += b
        fn75 += c
        bi = [i for i in range(1, len(g)) if g[i] != g[i - 1]]
        pi = [i for i in range(1, len(p)) if p[i] != p[i - 1]]
        gbt = [tt[i] for i in bi]
        pbt = [tt[i] for i in pi]
        for tol in tolerances:
            a, b, c = common.match_bnd(gbt, pbt, tol)
            bnd[tol][0] += a
            bnd[tol][1] += b
            bnd[tol][2] += c
    tp = np.diag(conf).astype(float)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    out = dict(macro_f1=float(np.mean(f1)),
               per_class_f1={i: float(f1[i]) for i in range(n_classes)},
               seg_f1=2 * tp50 / max(2 * tp50 + fp50 + fn50, 1),
               seg_f1_75=2 * tp75 / max(2 * tp75 + fp75 + fn75, 1))
    for tol in tolerances:
        a, b, c = bnd[tol]
        pr_ = a / max(a + b, 1)
        rc = a / max(a + c, 1)
        out[f"bnd_F1@{tol}"] = 2 * pr_ * rc / max(pr_ + rc, 1e-9)
    return out, conf
