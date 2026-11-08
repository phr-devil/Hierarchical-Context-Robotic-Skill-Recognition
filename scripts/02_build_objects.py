#!/usr/bin/env python
"""Step 02 - attach assembly-object labels and derive the object partitions.

Every frame inherits the object named by its enclosing parent segment. Frames
of the ``action`` artefact label are dropped, leaving the 17 assembly objects
that define the leave-one-object-out folds.

Output: ``<work-dir>/objects.npz`` and ``<work-dir>/objects.json``
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from crps import config as cfgmod  # noqa: E402
from crps import data, objects as objmod  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=os.environ.get("CRPS_DATA_ROOT", "data/REASSEMBLE"))
    ap.add_argument("--work-dir", default=os.environ.get("CRPS_WORK_DIR", "work"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit-episodes", type=int, default=0,
                    help="use only the first N episodes (debugging)")
    args = ap.parse_args()

    cfg = cfgmod.load(args.config)
    split_path = os.path.join(args.data_root, cfg["data"]["split_file"])
    episodes = data.read_split(split_path)
    if args.limit_episodes:
        episodes = episodes[:args.limit_episodes]

    frames_npz = np.load(os.path.join(args.work_dir, "frames.npz"), allow_pickle=True)
    frames = dict(
        ep=np.array([str(x) for x in frames_npz["ep"]]),
        seg=frames_npz["seg"].astype(int),
        t=frames_npz["t"].astype(float),
    )
    grid_npz = np.load(os.path.join(args.work_dir, "grid.npz"), allow_pickle=True)
    grid = dict(
        ep=np.array([str(x) for x in grid_npz["grid_ep"]]),
        t=grid_npz["grid_t"].astype(float),
        is_bg=grid_npz["is_bg"].astype(bool),
    )

    table = data.object_table(args.data_root, episodes)
    obj_grid, obj_episodes, real_objects, matched = objmod.build_object_grid(frames, grid, table)
    counts = objmod.object_frame_counts(obj_grid)
    n_annot = int((~grid["is_bg"]).sum())
    print(f"annotated={n_annot}  matched={matched}  objects={len(real_objects)}")

    np.savez_compressed(os.path.join(args.work_dir, "objects.npz"),
                        obj_label=obj_grid,
                        real_objects=np.array(real_objects),
                        artifact=np.array([objmod.ARTIFACT_OBJECT]))

    meta = dict(
        n=int(len(grid["t"])),
        n_annot=n_annot,
        n_unknown=int(grid["is_bg"].sum()),
        n_real_objects=len(real_objects),
        artifact=objmod.ARTIFACT_OBJECT,
        real_objects=real_objects,
        object_frame_counts={o: int(counts.get(o, 0)) for o in real_objects},
        object_episode_counts={o: len(obj_episodes.get(o, set())) for o in real_objects},
        object_episodes={o: sorted(obj_episodes.get(o, set())) for o in real_objects},
        artifact_frame_counts=int(counts.get(objmod.ARTIFACT_OBJECT, 0)),
        artifact_episodes=sorted(obj_episodes.get(objmod.ARTIFACT_OBJECT, set())),
    )
    with open(os.path.join(args.work_dir, "objects.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"{'object':16s} {'episodes':>8s} {'frames':>8s}")
    for o in real_objects:
        print(f"{o:16s} {len(obj_episodes.get(o, set())):8d} {counts.get(o, 0):8d}")
    print(f"saved {os.path.join(args.work_dir, 'objects.npz')} and objects.json")


if __name__ == "__main__":
    main()
