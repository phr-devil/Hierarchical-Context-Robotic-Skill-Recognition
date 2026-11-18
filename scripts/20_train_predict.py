#!/usr/bin/env python
"""Train one seed's arms on the development split and freeze test predictions.

This step reads test features but never test labels, and it writes the prediction
file before anything is scored.  Run it once per pre-registered seed:

    python scripts/20_train_predict.py --seed-index 0 --dataset-root /data/REASSEMBLE
    python scripts/20_train_predict.py --seed-index 1 --dataset-root /data/REASSEMBLE
    python scripts/20_train_predict.py --seed-index 2 --dataset-root /data/REASSEMBLE
"""
from __future__ import annotations

import json

from _cli import artifacts_dir, base_parser, resolve_config


def main() -> int:
    parser = base_parser("train A/P/S for one seed and freeze test predictions")
    parser.add_argument("--seed-index", type=int, required=True,
                        help="index into seeds.values in the configuration (0, 1, 2)")
    parser.add_argument("--artifacts", default=None)
    parser.add_argument("--pred-dir", default=None,
                        help="where predictions are written (default: <artifacts>/predictions)")
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()

    config = resolve_config(args)
    if args.seed_index not in range(len(config.seeds)):
        raise SystemExit(f"--seed-index must be in 0..{len(config.seeds) - 1}")

    import os

    os.environ.setdefault("OMP_NUM_THREADS", str(args.torch_threads))
    import torch

    torch.set_num_threads(args.torch_threads)

    from skillseg import pipeline

    artifacts = artifacts_dir(args)
    pred_dir = artifacts / "predictions" if args.pred_dir is None else args.pred_dir

    info = pipeline.train_and_predict(config, artifacts, artifacts, pred_dir, args.seed_index)
    print(json.dumps(info, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
