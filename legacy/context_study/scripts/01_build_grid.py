#!/usr/bin/env python
"""Step 01 - build the continuous timeline and the decoded emissions.

Fills the gaps between annotated segments with background frames, extracts the
72-dimensional feature for every frame of the resulting timeline, and fits the
two leave-one-date-out reference models on it:

* ``probs_sf``  - single-frame baseline, 72 dimensions
* ``probs_sfa`` - same features plus the ground-truth parent action (oracle)

The single-frame posteriors on annotated frames are cross-checked against the
values cached in step 00; a large disagreement means the feature or the frame
inventory changed.

Output: ``<work-dir>/grid.npz``
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from crps import config as cfgmod  # noqa: E402
from crps import data, features, grid as gridmod, models  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=os.environ.get("CRPS_DATA_ROOT", "data/REASSEMBLE"))
    ap.add_argument("--work-dir", default=os.environ.get("CRPS_WORK_DIR", "work"))
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = cfgmod.load(args.config)
    seed = cfgmod.set_seeds(cfg)
    window = float(cfg["data"]["window"])
    stride = float(cfg["data"]["stride"])
    C = float(cfg["single_frame"]["C"])
    max_iter = int(cfg["single_frame"]["max_iter"])

    frames_npz = np.load(os.path.join(args.work_dir, "frames.npz"), allow_pickle=True)
    frames = dict(
        ep=np.array([str(x) for x in frames_npz["ep"]]),
        seg=frames_npz["seg"].astype(int),
        t=frames_npz["t"].astype(float),
        y=frames_npz["y"].astype(int),
        pact=np.array([str(x) for x in frames_npz["pact"]]),
        date=np.array([str(x) for x in frames_npz["date"]]),
    )
    classes = [str(x) for x in frames_npz["classes"]]
    frames_probs_sf = frames_npz["probs_sf"].astype(np.float64)
    k = len(classes)

    grid = gridmod.build_grid(frames, stride)
    n_total = len(grid["t"])
    n_bg = int(grid["is_bg"].sum())
    print(f"grid frames={n_total}  annotated={n_total - n_bg}  background={n_bg} "
          f"({100 * n_bg / n_total:.2f}%)")

    t0 = time.time()
    short = np.zeros((n_total, features.SHORT_FEATURE_DIM))
    force = np.zeros((n_total, features.FORCE_FEATURE_DIM))
    for i, ep in enumerate(sorted(set(grid["ep"].tolist()))):
        mask = grid["ep"] == ep
        with data.open_episode(args.data_root, ep) as h5:
            s, f = features.short_force_features(h5, grid["t"][mask], window)
        short[mask] = s
        force[mask] = f
        if (i + 1) % 20 == 0 or i + 1 == len(set(grid["ep"].tolist())):
            print(f"  features {i + 1} episodes  t={time.time() - t0:.0f}s", flush=True)
    ALL = np.nan_to_num(np.hstack([short, force]), nan=0.0)
    short = np.nan_to_num(short, nan=0.0)
    force = np.nan_to_num(force, nan=0.0)

    dates = [str(x) for x in frames_npz["dates"]]
    label = grid["label"]

    probs_sf, cweights = models.fit_lodo_softmax(
        ALL, label, grid["date"], k, C=C, seed=seed, max_iter=max_iter,
        train_mask=~grid["is_bg"])
    print(f"leave-one-date-out SF done  t={time.time() - t0:.0f}s")

    # cross-check against step 00 on the annotated frames
    lookup = {(frames["ep"][i], round(float(frames["t"][i]), 6)): i for i in range(len(frames["t"]))}
    diffs = []
    for i in range(n_total):
        if grid["is_bg"][i]:
            continue
        key = (grid["ep"][i], round(float(grid["t"][i]), 6))
        j = lookup.get(key)
        if j is not None:
            diffs.append(np.abs(probs_sf[i] - frames_probs_sf[j]).max())
    diffs = np.array(diffs) if diffs else np.array([0.0])
    print(f"SF consistency: matched={len(diffs)}/{n_total - n_bg}  "
          f"max|delta|={diffs.max():.2e}  mean|delta|={diffs.mean():.2e}")

    act_names = gridmod.parent_vocabulary(grid["pact"])
    pact_oh = gridmod.parent_onehot(grid["pact"], act_names)
    print(f"parent actions={act_names}")
    probs_sfa, _ = models.fit_lodo_softmax(
        np.hstack([ALL, pact_oh]), label, grid["date"], k, C=C, seed=seed,
        max_iter=max_iter, train_mask=~grid["is_bg"])
    print(f"leave-one-date-out SFA done  t={time.time() - t0:.0f}s")

    out_path = os.path.join(args.work_dir, "grid.npz")
    np.savez_compressed(
        out_path,
        grid_ep=grid["ep"], grid_t=grid["t"], grid_label=label,
        grid_pact=grid["pact"], grid_date=grid["date"], is_bg=grid["is_bg"],
        SHORT=short.astype(np.float32), FORCE=force.astype(np.float32),
        ALL=ALL.astype(np.float32),
        probs_sf=probs_sf.astype(np.float32), probs_sfa=probs_sfa.astype(np.float32),
        pact_oh=pact_oh.astype(np.float32), act_names=np.array(act_names),
        classes=np.array(classes), dates=np.array(dates),
        cweights=np.array([cweights[d] for d in dates]),
    )
    print(f"saved {out_path}  total wall={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
