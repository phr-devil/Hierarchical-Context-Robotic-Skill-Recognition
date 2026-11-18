#!/usr/bin/env python
"""Step 00 - enumerate annotated frames and cache the single-frame baseline.

Reads the development split of REASSEMBLE, extracts the 72-dimensional frame
feature for every annotated frame, and fits the leave-one-date-out single-frame
classifier that later steps use as the visible-object reference.

Output: ``<work-dir>/frames.npz``
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from crps import config as cfgmod  # noqa: E402
from crps import data, features, models  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=os.environ.get("CRPS_DATA_ROOT", "data/REASSEMBLE"),
                    help="clone of the REASSEMBLE repository (contains data/ and splits/)")
    ap.add_argument("--work-dir", default=os.environ.get("CRPS_WORK_DIR", "work"),
                    help="directory for derived artefacts")
    ap.add_argument("--config", default=None, help="path to a JSON config")
    ap.add_argument("--limit-episodes", type=int, default=0,
                    help="use only the first N episodes (debugging)")
    args = ap.parse_args()

    cfg = cfgmod.load(args.config)
    seed = cfgmod.set_seeds(cfg)
    window = float(cfg["data"]["window"])
    stride = float(cfg["data"]["stride"])
    os.makedirs(args.work_dir, exist_ok=True)

    split_path = os.path.join(args.data_root, cfg["data"]["split_file"])
    episodes = data.read_split(split_path)
    if args.limit_episodes:
        episodes = episodes[:args.limit_episodes]
    print(f"episodes={len(episodes)}  window={window}s  stride={stride}s  seed={seed}")

    t0 = time.time()
    frames = data.annotated_frames(args.data_root, episodes, window, stride)
    n = frames["n"]
    k = len(frames["classes"])
    print(f"annotated frames={n}  skills={k}  {frames['classes'].tolist()}")

    X = np.zeros((n, features.FRAME_FEATURE_DIM))
    for i, ep in enumerate(sorted(set(frames["ep"].tolist()))):
        mask = frames["ep"] == ep
        X[mask] = features.frame_features(data.episode_path(args.data_root, ep),
                                          frames["t"][mask], window)
        if (i + 1) % 20 == 0 or i + 1 == len(set(frames["ep"].tolist())):
            print(f"  features {i + 1} episodes  t={time.time() - t0:.0f}s", flush=True)

    dates = sorted(set(frames["date"].tolist()))
    probs_sf, cweights = models.fit_lodo_softmax(
        X, frames["y"], frames["date"], k,
        C=float(cfg["single_frame"]["C"]), seed=seed,
        max_iter=int(cfg["single_frame"]["max_iter"]))
    print(f"leave-one-date-out SF done  dates={dates}  t={time.time() - t0:.0f}s")

    np.savez_compressed(
        os.path.join(args.work_dir, "frames.npz"),
        ep=frames["ep"], seg=frames["seg"], t=frames["t"], y=frames["y"],
        pact=frames["pact"], date=frames["date"], classes=frames["classes"],
        probs_sf=probs_sf.astype(np.float32),
        dates=np.array(dates),
        cweights=np.array([cweights[d] for d in dates]),
    )
    print(f"saved {os.path.join(args.work_dir, 'frames.npz')}  total wall={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
