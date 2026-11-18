#!/usr/bin/env python
"""Check that this implementation rebuilds a reference development grid exactly.

The reference is the grid file the frozen scoring run consumed, an ``.npz`` with
``grid_ep``, ``grid_t``, ``grid_label``, ``is_bg`` and ``ALL``.  The comparison is
exact for labels and times and uses a zero tolerance for the 72-dimensional
features.

    python scripts/90_verify_dev_grid.py --reference /path/to/frozen_grid.npz \
        --dataset-root /data/REASSEMBLE

Exit code 0 means the grid, the labels and the features are identical.
"""
from __future__ import annotations

import json

import numpy as np

from _cli import base_parser, resolve_config


def main() -> int:
    parser = base_parser("verify the rebuilt development grid against a reference")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--split", default="train", choices=("train", "test"))
    args = parser.parse_args()

    config = resolve_config(args)
    from skillseg.data import build_grid
    from skillseg.features import frame_features

    reference = np.load(args.reference, allow_pickle=True)
    ref_ep = np.array([str(x) for x in reference["grid_ep"]])
    ref_t = reference["grid_t"].astype(np.float64)
    ref_label = reference["grid_label"].astype(np.int64)
    ref_bg = reference["is_bg"].astype(bool)
    ref_all = reference["ALL"].astype(np.float32)

    episodes = config.dataset.split(args.split)
    grid = build_grid(episodes, config, verbose=True)
    features = frame_features(grid, config, verbose=True)

    report = {
        "split": args.split,
        "reference_frames": int(len(ref_t)),
        "rebuilt_frames": int(len(grid)),
        "frames_equal": bool(len(ref_t) == len(grid)),
    }
    if not report["frames_equal"]:
        print(json.dumps(report, indent=1))
        return 1

    got_ep = np.array([str(x) for x in grid.ep])
    report["episode_ids_equal"] = bool(np.array_equal(ref_ep, got_ep))
    report["times_max_abs_diff"] = float(np.max(np.abs(ref_t - grid.t)))
    report["labels_equal"] = bool(np.array_equal(ref_label, grid.label))
    report["background_equal"] = bool(np.array_equal(ref_bg, (grid.label < 0)))
    report["features_max_abs_diff"] = float(np.max(np.abs(ref_all - features)))
    report["features_shape_equal"] = bool(ref_all.shape == features.shape)

    ok = (report["episode_ids_equal"] and report["times_max_abs_diff"] == 0.0
          and report["labels_equal"] and report["background_equal"]
          and report["features_shape_equal"] and report["features_max_abs_diff"] == 0.0)
    report["identical"] = bool(ok)
    print(json.dumps(report, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
