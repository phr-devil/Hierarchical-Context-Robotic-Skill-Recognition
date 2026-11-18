#!/usr/bin/env python
"""Step 22 - statistical audit of the fold structure and the headline claim.

Three things are checked because they determine how the numbers may be read.

1. Fold non-independence. The 17 object folds share most of their training
   episodes and overlap in their test episodes, so they are correlated
   replications and not 17 independent tests.
2. Honest significance. The independent unit is the recorded episode, not the
   fold. A paired Wilcoxon test and a bootstrap confidence interval are computed
   over episodes, with a date-clustered variant as a conservative check, because
   episodes cluster by recording date.
3. Metric separation. Skill recognition, segmental F1 and full-episode boundary
   F1 are reported separately: a gain in the first two is not evidence of better
   temporal localisation.

Output: ``<work-dir>/stats_audit.json``
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from crps import config as cfgmod, data  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=os.environ.get("CRPS_DATA_ROOT", "data/REASSEMBLE"))
    ap.add_argument("--work-dir", default=os.environ.get("CRPS_WORK_DIR", "work"))
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = cfgmod.load(args.config)
    boot_seed = int(cfg["seeds"]["bootstrap_seed"])
    episodes = data.read_split(os.path.join(args.data_root, cfg["data"]["split_file"]))

    with open(os.path.join(args.work_dir, "l2_eval.json"), "r") as fh:
        EVAL = json.load(fh)
    with open(os.path.join(args.work_dir, "objects.json"), "r") as fh:
        OBJ = json.load(fh)
    g = np.load(os.path.join(args.work_dir, "grid.npz"), allow_pickle=True)
    classes = [str(x) for x in g["classes"]]

    real_objects = EVAL["objects"]
    if EVAL.get("subset"):
        print(f"note: the evaluation file covers only {len(real_objects)} objects "
              f"(--max-objects {EVAL['subset']}); the audit describes that subset")
    obj_eps = {o: set(OBJ["object_episodes"][o]) for o in real_objects}

    # (1) fold overlap
    overlap = []
    train_overlap = []
    for i, a in enumerate(real_objects):
        for b in real_objects[i + 1:]:
            overlap.append(len(obj_eps[a] & obj_eps[b]))
            train_overlap.append(len((set(episodes) - obj_eps[a]) & (set(episodes) - obj_eps[b])))
    test_sizes = {o: len(obj_eps[o]) for o in real_objects}

    def date_of(ep):
        return ep[:10]

    date_count = Counter(date_of(e) for e in episodes)
    min_train_per_date = {}
    for o in real_objects:
        cnt = Counter(date_of(e) for e in set(episodes) - obj_eps[o])
        min_train_per_date[o] = int(min(cnt.values())) if cnt else 0

    # (2) episode-level pairing
    ep2d = defaultdict(list)
    for o in real_objects:
        for rec in EVAL["results"][o].get("episode_paired", []):
            ep2d[rec["episode"]].append(rec["segF1_GRUctx"] - rec["segF1_SF"])
    uniq = list(ep2d.keys())
    ep_mean = np.array([float(np.mean(ep2d[e])) for e in uniq])
    n_ep = len(uniq)
    improved = int((ep_mean > 0).sum())
    degraded = int((ep_mean < 0).sum())

    w = wilcoxon(ep_mean)
    rng = np.random.default_rng(boot_seed)
    boot = np.array([ep_mean[rng.integers(0, n_ep, n_ep)].mean() for _ in range(4000)])
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    date_dmeans = defaultdict(list)
    for e in uniq:
        date_dmeans[date_of(e)].append(float(np.mean(ep2d[e])))
    date_keys = list(date_dmeans.keys())
    date_boot = []
    for _ in range(2000):
        sel = rng.choice(date_keys, size=len(date_keys), replace=True)
        vals = []
        for d in sel:
            vals.extend(date_dmeans[d])
        date_boot.append(float(np.mean(vals)))
    date_ci = [float(np.percentile(date_boot, 2.5)), float(np.percentile(date_boot, 97.5))]
    date_means = {d: float(np.mean(v)) for d, v in date_dmeans.items()}
    date_w = wilcoxon(np.array(list(date_means.values())))

    # (3) metric separation
    per_class = {}
    for i, c in enumerate(classes):
        sf = np.mean([EVAL["results"][o]["SF|lag3"]["per_class_f1"][str(i)] for o in real_objects])
        gr = np.mean([EVAL["results"][o]["GRUctx|lag3"]["per_class_f1"][str(i)] for o in real_objects])
        per_class[c] = dict(SF=round(float(sf), 3), GRUctx=round(float(gr), 3),
                            gain=round(float(gr - sf), 4))
    bf_sf = float(np.mean([EVAL["results"][o]["SF|lag3"]["bnd_F1_full_0_5"] for o in real_objects]))
    bf_gr = float(np.mean([EVAL["results"][o]["GRUctx|lag3"]["bnd_F1_full_0_5"] for o in real_objects]))
    seg_sf = float(np.mean([EVAL["results"][o]["SF|lag3"]["seg_f1"] for o in real_objects]))
    seg_gr = float(np.mean([EVAL["results"][o]["GRUctx|lag3"]["seg_f1"] for o in real_objects]))
    shuf = float(np.mean([EVAL["results"][o]["Shufctx|lag3"]["seg_f1"] for o in real_objects]))

    out = dict(
        fold_nonindependence=dict(
            n_objects=len(real_objects),
            test_set_size_mean=round(float(np.mean(list(test_sizes.values()))), 1),
            test_set_size_range=[min(test_sizes.values()), max(test_sizes.values())],
            mean_pairwise_test_overlap_episodes=round(float(np.mean(overlap)), 1),
            mean_pairwise_train_overlap_episodes=round(float(np.mean(train_overlap)), 1),
            note="the 17 object folds share most training episodes and overlap in test "
                 "episodes; they are correlated replications, not 17 independent tests"),
        episode_independence=dict(
            n_dev_episodes=len(episodes), n_unique_episodes_in_pairing=n_ep,
            improved_episodes=improved, degraded_episodes=degraded,
            improved_frac=round(improved / n_ep, 3)),
        date_correlation=dict(
            n_dates=len(date_count),
            episodes_per_date=dict(sorted(date_count.items())),
            min_training_episodes_per_date_across_folds=int(min(min_train_per_date.values())),
            date_level_mean_dSegF1=date_means,
            note="leaving out one object never empties a date from training, but test "
                 "episodes still cluster by date; date-level signs are reported"),
        honest_significance=dict(
            unit=f"{n_ep} independent episodes, paired SF vs GRUctx",
            wilcoxon_W=round(float(w.statistic), 1),
            wilcoxon_p_two_sided=float(w.pvalue),
            wilcoxon_p_one_sided=float(w.pvalue / 2),
            episode_mean_dSegF1=round(float(ep_mean.mean()), 4),
            bootstrap_95ci_mean_dSegF1=[round(ci[0], 4), round(ci[1], 4)],
            date_clustered_bootstrap_95ci=[round(date_ci[0], 4), round(date_ci[1], 4)],
            date_level_wilcoxon_p_two_sided=float(date_w.pvalue)),
        three_metric_separation=dict(
            skill_recognition_per_class_F1=per_class,
            segmental_F1=dict(SF=round(seg_sf, 4), GRUctx=round(seg_gr, 4),
                              gain=round(seg_gr - seg_sf, 4)),
            full_episode_boundary_F1_at_0_5=dict(
                SF=round(bf_sf, 4), GRUctx=round(bf_gr, 4),
                note="boundary localisation is flat; the gain is label-level"),
            shuffle_control_segF1=round(shuf, 4)),
    )
    path = os.path.join(args.work_dir, "stats_audit.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"folds: test size {out['fold_nonindependence']['test_set_size_mean']} episodes, "
          f"pairwise test overlap "
          f"{out['fold_nonindependence']['mean_pairwise_test_overlap_episodes']} episodes")
    print(f"episodes: {n_ep}, improved {improved} ({improved / n_ep * 100:.1f}%), "
          f"mean dSegF1={ep_mean.mean():+.4f}")
    print(f"Wilcoxon (episode level): W={w.statistic:.1f}  p={w.pvalue:.3e}  "
          f"bootstrap 95% CI={ci}")
    print(f"date-clustered 95% CI={date_ci}")
    print(f"boundary F1 full timeline: SF={bf_sf:.4f}  GRUctx={bf_gr:.4f}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
