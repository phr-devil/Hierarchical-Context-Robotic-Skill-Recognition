#!/usr/bin/env python
"""Build the evaluation grid, the 72-dimensional features and the causal stack.

Example
-------
    python scripts/10_build_grid.py --split train --dataset-root /data/REASSEMBLE
    python scripts/10_build_grid.py --split test  --dataset-root /data/REASSEMBLE
"""
from __future__ import annotations

import json

from _cli import artifacts_dir, base_parser, resolve_config


def main() -> int:
    parser = base_parser("build grid and features for one official split")
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--artifacts", default=None, help="output directory (default: ./artifacts)")
    args = parser.parse_args()

    config = resolve_config(args)
    out = artifacts_dir(args)

    from skillseg import pipeline

    meta = pipeline.prepare_split(config, args.split, out, verbose=True)
    meta["artifacts"] = str(out)
    print(json.dumps(meta, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
