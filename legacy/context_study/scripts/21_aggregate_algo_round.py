#!/usr/bin/env python
"""Step 21 - aggregate the five-model comparison.

Produces the fold-averaged summary, the frame-pooled per-class F1 (the
complement to the fold-averaged view of step 20), the pairwise confusion
fractions that explain where the gain comes from, and the episode-level
coverage of each comparison.

Output: ``<work-dir>/algo_round_agg.json``
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

SCHEMES = ["SF|lag3", "GRUskill|lag3", "DualHead|lag3", "GRUctx|lag3", "GateCtx|lag3"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", default=os.environ.get("CRPS_WORK_DIR", "work"))
    args = ap.parse_args()

    with open(os.path.join(args.work_dir, "algo_round.json"), "r") as fh:
        R = json.load(fh)
    g = np.load(os.path.join(args.work_dir, "grid.npz"), allow_pickle=True)
    gep = np.array([str(x) for x in g["grid_ep"]])
    gdt = np.array([str(x) for x in g["grid_date"]])
    classes = [str(x) for x in g["classes"]]
    K = len(classes)
    ep2date = {e: d for e, d in zip(gep, gdt)}

    res = R["results"]
    objs = R["objects"]

    agg = {s: dict(seg=[], macro=[], bf5=[], pc={}) for s in SCHEMES}
    pooled_conf = {s: np.zeros((K, K), dtype=np.int64) for s in SCHEMES}
    ep_rows = []
    for o in objs:
        oj = res[o]
        for s in SCHEMES:
            if s not in oj:
                continue
            agg[s]["seg"].append(oj[s]["seg_f1"])
            agg[s]["macro"].append(oj[s]["macro_f1"])
            agg[s]["bf5"].append(oj[s]["bnd_F1_full_0_5"])
            for k, v in oj[s]["per_class_f1"].items():
                agg[s]["pc"].setdefault(int(k), []).append(v)
            pooled_conf[s] += np.array(oj[s]["conf"])
        ep_rows.extend(oj.get("episode_paired", []))

    keys = {s: "segF1_" + s.split("|")[0] for s in SCHEMES}
    arr = {s: np.array([r[keys[s]] for r in ep_rows]) for s in SCHEMES}

    coverage = {}
    for a, b in [("GRUctx|lag3", "SF|lag3"), ("DualHead|lag3", "SF|lag3"),
                 ("GRUskill|lag3", "SF|lag3"), ("GateCtx|lag3", "GRUctx|lag3"),
                 ("GateCtx|lag3", "SF|lag3")]:
        coverage[f"{a.split('|')[0]}>{b.split('|')[0]}"] = float((arr[a] > arr[b]).mean())

    per_class = {}
    for s in SCHEMES:
        conf = pooled_conf[s]
        tp = np.diag(conf).astype(float)
        fp = conf.sum(0) - tp
        fn = conf.sum(1) - tp
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        per_class[s] = {classes[i]: round(float(f1[i]), 4) for i in range(K)}

    def confusion_fraction(s, frm, to):
        conf = pooled_conf[s]
        tot = conf[frm].sum()
        return float(conf[frm, to] / tot) if tot > 0 else 0.0

    idx = {c: i for i, c in enumerate(classes)}
    mechanism = {}
    for s in SCHEMES:
        mechanism[s] = dict(
            Lift_to_Approach=round(confusion_fraction(s, idx["Lift"], idx["Approach"]), 4),
            Approach_to_Lift=round(confusion_fraction(s, idx["Approach"], idx["Lift"]), 4),
            Push_to_Grasp=round(confusion_fraction(s, idx["Push"], idx["Grasp"]), 4),
            Grasp_to_Push=round(confusion_fraction(s, idx["Grasp"], idx["Push"]), 4))

    summary = {s: dict(seg_mean=round(float(np.mean(agg[s]["seg"])), 4),
                       seg_std=round(float(np.std(agg[s]["seg"])), 4),
                       macro_mean=round(float(np.mean(agg[s]["macro"])), 4),
                       bf5_mean=round(float(np.mean(agg[s]["bf5"])), 4)) for s in SCHEMES}
    gain = {s: round(summary[s]["seg_mean"] - summary["SF|lag3"]["seg_mean"], 4)
            for s in SCHEMES if s != "SF|lag3"}

    per_date = {s: {} for s in SCHEMES}
    for r in ep_rows:
        d = ep2date.get(r["episode"], "NA")
        for s in SCHEMES:
            per_date[s].setdefault(d, []).append(r[keys[s]])
    per_date_mean = {s: {d: round(float(np.mean(v)), 4) for d, v in per_date[s].items()}
                     for s in SCHEMES}

    out = dict(summary=summary, gain_vs_SF=gain, per_class_pooled=per_class,
               mechanism=mechanism, episode_coverage=coverage,
               n_episodes=len(ep_rows), per_date_segF1=per_date_mean,
               hparams=R.get("hparams"))
    path = os.path.join(args.work_dir, "algo_round_agg.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)

    print("=== segmental F1, mean over 17 object folds ===")
    for s in SCHEMES:
        gain_s = f"{gain[s]:+.4f}" if s in gain else "-"
        print(f"  {s:14s} segF1={summary[s]['seg_mean']:.4f}  "
              f"macro={summary[s]['macro_mean']:.4f}  "
              f"boundaryF1={summary[s]['bf5_mean']:.4f}  gain_vs_SF={gain_s}")
    print("\n=== frame-pooled per-class F1 ===")
    for s in SCHEMES:
        print(f"  {s:14s} " + "  ".join(f"{c}={per_class[s][c]:.3f}" for c in classes))
    print("\n=== episode-level coverage (111 development episodes) ===")
    for k, v in coverage.items():
        print(f"  {k:18s} = {v * 100:.1f}%")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
