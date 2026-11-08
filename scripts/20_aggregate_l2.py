#!/usr/bin/env python
"""Step 20 - aggregate the main experiment into the reported table.

Averages are taken over folds (one value per held-out object), not over frames,
so that an object with many frames does not dominate. Per-class averages are
also over folds; a frame-pooled per-class view is produced by step 21.

Output: ``<work-dir>/l2_analysis.json``
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

SCHEMES = ["SF|lag3", "GRUctx|lag3", "Shufctx|lag3", "SFA_Oracle|lag3", "LODO_SF|lag3(seen)"]
METRICS = ["seg_f1", "macro_f1", "bnd_F1_full_0_5", "bnd_F1_full_0_2", "seg_f1_75"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", default=os.environ.get("CRPS_WORK_DIR", "work"))
    args = ap.parse_args()

    with open(os.path.join(args.work_dir, "l2_eval.json"), "r") as fh:
        d = json.load(fh)
    objects = d["objects"]
    results = d["results"]
    volmatch = d["volume_match"]

    g = np.load(os.path.join(args.work_dir, "grid.npz"), allow_pickle=True)
    classes = [str(x) for x in g["classes"]]

    def metric(o, scheme, m):
        return results[o].get(scheme, {}).get(m, float("nan"))

    pooled = {}
    for m in METRICS:
        pooled[m] = {s: float(np.mean([metric(o, s, m) for o in objects])) for s in SCHEMES}
    pooled["seg_f1"]["VolMatch_SF|lag3(seen,small)"] = float(
        np.mean([volmatch[o]["mean_seg_f1"] for o in objects]))

    per_class = {}
    for i, c in enumerate(classes):
        row = {}
        for s in ["SF|lag3", "GRUctx|lag3"]:
            row[s] = float(np.mean([results[o][s]["per_class_f1"][str(i)] for o in objects]))
        row["gain"] = row["GRUctx|lag3"] - row["SF|lag3"]
        per_class[c] = {k: round(v, 4) for k, v in row.items()}

    dseg = {o: results[o]["GRUctx|lag3"]["seg_f1"] - results[o]["SF|lag3"]["seg_f1"]
            for o in objects}
    n_pos = sum(1 for v in dseg.values() if v > 0)
    n_neg = sum(1 for v in dseg.values() if v < 0)

    def transition_recall(tag, scheme):
        det = sum(results[o][scheme]["transitions_full"][tag][0] for o in objects)
        tot = sum(results[o][scheme]["transitions_full"][tag][1] for o in objects)
        return det, tot

    transitions = {tag: {s: transition_recall(tag, s) for s in ["SF|lag3", "GRUctx|lag3"]}
                   for tag in ["Align->Push", "Lift->Push", "Release->Grasp"]}

    ep_all = []
    for o in objects:
        ep_all.extend(results[o].get("episode_paired", []))
    dep = np.array([r["dSegF1"] for r in ep_all])

    lat_sf = [results[o]["SF|lag3"]["latency_full"] for o in objects]
    latency = {
        "boundary_time_error_abs_mean": float(np.mean([x["boundary_time_error_abs_mean"] for x in lat_sf])),
        "boundary_time_error_abs_p90": float(np.mean([x["boundary_time_error_abs_p90"] for x in lat_sf])),
        "fixed_wait_s": lat_sf[0]["fixed_wait_s"],
        "actual_output_abs_mean": float(np.mean([x["actual_output_abs_mean"] for x in lat_sf])),
        "actual_output_abs_p90": float(np.mean([x["actual_output_abs_p90"] for x in lat_sf])),
    }

    transfer = float(np.mean([results[o]["LODO_SF|lag3(seen)"]["seg_f1"]
                              - results[o]["SF|lag3"]["seg_f1"] for o in objects]))

    out = dict(
        n_objects=len(objects),
        pooled_metrics=pooled,
        per_class_f1_mean_over_folds=per_class,
        per_object_dSegF1={k: round(v, 4) for k, v in dseg.items()},
        n_objects_positive=int(n_pos),
        n_objects_negative=int(n_neg),
        mean_dSegF1=float(np.mean(list(dseg.values()))),
        transitions_full=transitions,
        episode_paired=dict(n_pairings=int(len(ep_all)), n_episodes=int(dep.size),
                            mean_dSegF1=float(dep.mean()),
                            improve_frac=float((dep > 0).mean()),
                            degrade_frac=float((dep < 0).mean())),
        transfer_cost=dict(
            mean=transfer,
            definition="leave-one-date-out posteriors on the same test frames (object seen) "
                       "minus the object-unseen single-frame baseline"),
        latency_l2_SF_lag3_full=latency,
    )
    path = os.path.join(args.work_dir, "l2_analysis.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)

    print("=== fold-averaged metrics ===")
    for m in METRICS:
        print(f"  {m:20s} " + "  ".join(f"{s}={pooled[m][s]:.4f}" for s in SCHEMES))
    print(f"  {'seg_f1 VolMatch':20s} {pooled['seg_f1']['VolMatch_SF|lag3(seen,small)']:.4f}")
    print("\n=== per-class F1, mean over folds ===")
    for c, row in per_class.items():
        print(f"  {c:10s} SF={row['SF|lag3']:.3f}  GRUctx={row['GRUctx|lag3']:.3f}  "
              f"gain={row['gain']:+.3f}")
    print(f"\nsegmental F1 gain: mean={out['mean_dSegF1']:+.4f}  "
          f"positive objects={n_pos}/{len(objects)}")
    print(f"episode-level pairing: n={out['episode_paired']['n_episodes']}  "
          f"improved={out['episode_paired']['improve_frac'] * 100:.1f}%")
    print("\n=== transitions (full episode timeline) ===")
    for tag, row in transitions.items():
        print(f"  {tag:14s} SF={row['SF|lag3'][0]}/{row['SF|lag3'][1]}  "
              f"GRUctx={row['GRUctx|lag3'][0]}/{row['GRUctx|lag3'][1]}")
    print(f"\nlatency (SF|lag3, full timeline): boundary error "
          f"{latency['boundary_time_error_abs_mean']:.3f}s + wait {latency['fixed_wait_s']}s "
          f"-> output {latency['actual_output_abs_mean']:.3f}s")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
