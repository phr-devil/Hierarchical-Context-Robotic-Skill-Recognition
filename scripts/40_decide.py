#!/usr/bin/env python
"""Apply the pre-registered decision rule to a scores file.

Both readings of the interval-width clause are written out.  The script prints
them side by side and never prints only the favourable one.

    python scripts/40_decide.py --scores results/scores.json
"""
from __future__ import annotations

import json
from pathlib import Path

from _cli import base_parser, resolve_config


def main() -> int:
    parser = base_parser("apply the pre-registered decision rule")
    parser.add_argument("--scores", default="results/scores.json")
    parser.add_argument("--out", default="results/decision.json")
    args = parser.parse_args()

    config = resolve_config(args, require_root=False)
    from skillseg import pipeline

    scores = json.loads(Path(args.scores).read_text(encoding="utf-8"))
    decision = pipeline.decide(config, scores)
    pipeline.save_json(args.out, decision)

    print(json.dumps({
        "delta_pp": round(100 * decision["delta"], 4),
        "ci95_pp": [round(100 * v, 4) for v in decision["ci95"]],
        "width_pp": round(100 * decision["width"], 4),
        "max_width_pp": round(100 * decision["rules"]["max_ci_width"], 4),
        "episodes_improved": f"{decision['sign_pos']}/{decision['n_episodes']}",
        "verdict": decision["verdict"],
        "verdict_strict_width_enforced": decision["verdict_strict"],
        "clause_conflict": decision["clause_conflict"],
        "written": args.out,
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
