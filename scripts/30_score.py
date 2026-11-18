#!/usr/bin/env python
"""Score the frozen predictions against the test labels.

Run order matters: the holdout is scored once, after all seeds have produced
their prediction files, and the prediction hashes are checked before any label is
read.  Re-running this step on unchanged predictions is harmless; re-running the
previous step in order to change the outcome is not.

    python scripts/30_score.py
"""
from __future__ import annotations

import json
from pathlib import Path

from _cli import artifacts_dir, base_parser, resolve_config


def main() -> int:
    parser = base_parser("score frozen predictions, from artifacts only")
    parser.add_argument("--artifacts", default=None)
    parser.add_argument("--pred-dir", default=None)
    parser.add_argument("--out", default=None, help="scores json (default: results/scores.json)")
    args = parser.parse_args()

    config = resolve_config(args, require_root=False)
    from skillseg import pipeline
    from skillseg.data import Grid

    artifacts = artifacts_dir(args)
    pred_dir = artifacts / "predictions" if args.pred_dir is None else Path(args.pred_dir)

    grid = Grid.load(artifacts / "test_grid.npz")
    infos, files = [], []
    for index in range(len(config.seeds)):
        info_path = pred_dir / f"seed{index}.json"
        if not info_path.exists():
            raise SystemExit(f"missing {info_path}; run scripts/20_train_predict.py first")
        infos.append(json.loads(info_path.read_text(encoding="utf-8")))
        files.append(pred_dir / f"pred_seed{index}.npz")

    scores = pipeline.score_predictions(config, grid, files, infos)
    out = Path(args.out) if args.out else Path("results") / "scores.json"
    pipeline.save_json(out, scores)

    print(json.dumps({
        "episodes": scores["n_episodes"],
        "cells": scores["n_cells"],
        "A": round(scores["arm_means"]["A"], 4),
        "P": round(scores["arm_means"]["P"], 4),
        "S": round(scores["arm_means"]["S"], 4),
        "delta_pp": round(100 * scores["paired"]["mean"], 4),
        "ci95_pp": [round(100 * v, 4) for v in scores["bootstrap_ci"]],
        "episodes_improved": f"{scores['paired']['n_pos']}/{scores['n_episodes']}",
        "positive_in_every_seed": scores["seed_consistency"]["n_episodes_positive_in_every_seed"],
        "written": str(out),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
