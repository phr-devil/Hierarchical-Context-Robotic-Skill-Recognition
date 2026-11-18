#!/usr/bin/env python
"""Grid censoring: how many native skill segments survive the 5 Hz grid.

Label side only -- no prediction is read, so this cannot be tuned against a score.

    python scripts/50_censoring.py --split test --dataset-root /data/REASSEMBLE
"""
from __future__ import annotations

import json
from pathlib import Path

from _cli import artifacts_dir, base_parser, resolve_config


def main() -> int:
    parser = base_parser("grid censoring diagnostic")
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--artifacts", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = resolve_config(args)
    from skillseg import pipeline
    from skillseg.data import Grid, native_segments

    artifacts = artifacts_dir(args)
    grid = Grid.load(artifacts / f"{args.split}_grid.npz")
    segments = native_segments(grid.episodes, config)
    stats = pipeline.grid_censoring(grid, segments, config)

    out = Path(args.out) if args.out else Path("results") / f"censoring_{args.split}.json"
    pipeline.save_json(out, stats)
    print(json.dumps({
        "split": args.split,
        "segments": stats["n_segments"],
        "pooled_share_at_most_two_frames": round(stats["pooled"]["share_at_most_two_frames"], 4),
        "Release": round(stats["per_class"]["Release"]["share_at_most_two_frames"], 4),
        "Lift": round(stats["per_class"]["Lift"]["share_at_most_two_frames"], 4),
        "written": str(out),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
