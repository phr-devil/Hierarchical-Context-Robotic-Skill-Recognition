#!/usr/bin/env python
"""Report the environment and confirm that the protocol configuration loads.

Exits non-zero only if a dependency or the protocol file is broken.  A missing
dataset is reported as a warning: the repository is meant to be inspectable
without the third-party corpus.
"""
from __future__ import annotations

import json
import platform
import sys

from _cli import REPO_ROOT, base_parser, resolve_config


def main() -> int:
    args = base_parser("check environment and configuration").parse_args()
    report = {"python": platform.python_version(), "executable": sys.executable}

    for name in ("numpy", "scipy", "sklearn", "torch", "h5py"):
        try:
            module = __import__(name)
            report[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            report[name] = f"MISSING ({exc.__class__.__name__})"

    try:
        from skillseg import __version__ as package_version
        from skillseg.config import load_config

        report["skillseg"] = package_version
        config = load_config(args.config)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({**report, "config_error": str(exc)}, indent=1))
        return 1

    report["protocol"] = {
        "skills": list(config.skills),
        "actions": list(config.actions),
        "feature_dim": config.feature_dim,
        "history_frames": config.history_frames,
        "window_s": config.window_s,
        "stride_s": config.stride_s,
        "hidden": config.gru["hidden"],
        "lag_frames": config.lag_frames,
        "seeds": list(config.seeds),
        "train_seeds": [config.train_seed(i) for i in range(len(config.seeds))],
        "prereg": config.prereg,
        "iou_threshold": config.iou_threshold,
        "bootstrap_resamples": config.bootstrap_resamples,
    }

    config = resolve_config(args)
    splits = {}
    for which in ("train", "test"):
        try:
            splits[which] = len(config.dataset.split(which))
        except FileNotFoundError as exc:
            splits[which] = f"missing: {exc}"
    report["splits"] = splits
    report["dataset_root"] = str(config.dataset.root)
    report["dataset_present"] = config.dataset.data_dir.exists()

    print(json.dumps(report, indent=1, ensure_ascii=False))
    if not report["dataset_present"]:
        print("\nWARNING: dataset directory not found; build/train/score steps cannot run.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
