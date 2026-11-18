"""Continuous evaluation timeline.

Annotated frames only exist inside labelled low-level segments, so adjacent
segments leave gaps in which nothing is observed. Decoding a sequence over the
annotated frames alone would silently skip those gaps and make boundary timing
meaningless, so the gaps are filled with *background* frames at the same stride
(label ``-1``); they are never scored, but the decoder has to traverse them.

Background frames get the parent action of their neighbours when both agree and
the sentinel ``none`` otherwise. That value is used for analysis only and never
as a model input.
"""
import numpy as np

LABEL_BACKGROUND = -1
PARENT_UNKNOWN = "none"


def build_grid(frames, stride):
    """Interleave annotated frames with background frames on a common timeline."""
    ep = frames["ep"]
    t = frames["t"]
    label = frames["y"]
    pact = frames["pact"]
    date = frames["date"]

    by_ep = {}
    for i, e in enumerate(ep):
        by_ep.setdefault(e, []).append(i)

    g_ep, g_t, g_lab, g_pact, g_date = [], [], [], [], []
    for e in sorted(by_ep):
        idxs = np.array(sorted(by_ep[e], key=lambda i: t[i]))
        tt = t[idxs]
        yy = label[idxs]
        pa = pact[idxs]
        dt = date[idxs]
        order = np.argsort(tt)
        tt, yy, pa, dt = tt[order], yy[order], pa[order], dt[order]

        for k in range(len(tt)):
            g_ep.append(e)
            g_t.append(tt[k])
            g_lab.append(int(yy[k]))
            g_pact.append(pa[k])
            g_date.append(dt[k])

        for k in range(1, len(tt)):
            gap = tt[k] - tt[k - 1]
            if gap > stride + 1e-6:
                for tb in np.arange(tt[k - 1] + stride, tt[k], stride):
                    g_ep.append(e)
                    g_t.append(float(tb))
                    g_lab.append(LABEL_BACKGROUND)
                    g_pact.append(pa[k - 1] if pa[k - 1] == pa[k] else PARENT_UNKNOWN)
                    g_date.append(dt[k])

    grid_label = np.array(g_lab)
    return dict(
        ep=np.array(g_ep),
        t=np.array(g_t),
        label=grid_label,
        pact=np.array(g_pact),
        date=np.array(g_date),
        is_bg=grid_label < 0,
    )


def episode_index(ep):
    """Map episode id -> array of frame indices sorted by time (grid must be time-ordered)."""
    out = {}
    for i, e in enumerate(ep):
        out.setdefault(e, []).append(i)
    return {k: np.array(v) for k, v in out.items()}


def parent_vocabulary(pact):
    """Sorted parent-action labels, excluding the background sentinels."""
    return sorted(set(pact.tolist()) - {PARENT_UNKNOWN, "no", "nan"})


def parent_onehot(pact, vocabulary):
    """One-hot parent actions; rows of sentinel values stay all-zero."""
    index = {a: i for i, a in enumerate(vocabulary)}
    oh = np.zeros((len(pact), len(vocabulary)))
    for i, a in enumerate(pact):
        if a in index:
            oh[i, index[a]] = 1.0
    return oh
