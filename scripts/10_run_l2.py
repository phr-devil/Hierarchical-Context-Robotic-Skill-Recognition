#!/usr/bin/env python
"""Step 10 - main leave-one-object-out experiment (single-frame vs GRU context).

For each of the 17 assembly objects the whole pipeline is refitted on the other
sixteen objects: the SF classifier, the parent-action GRU, the soft context
classifier, the shuffle control and the oracle, together with the transition
matrix and the class weights used for decoding. Only the held-out object's
labelled frames are scored.

Reported schemes
    SF|lag3              single-frame baseline
    GRUctx|lag3          predicted parent action appended as four extra inputs
    Shufctx|lag3         same, with the predicted context permuted frame-wise
    SFA_Oracle|lag3      ground-truth parent action (conditional upper bound)
    LODO_SF|lag3(seen)   same test frames, object seen in training
    VolMatch_SF|lag3     object seen, matched (small) training volume

Output: ``<work-dir>/l2_eval.json``
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from crps import common, config as cfgmod, models, protocol  # noqa: E402
from crps.models import CausalGRU  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", default=os.environ.get("CRPS_WORK_DIR", "work"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-objects", type=int, default=0,
                    help="evaluate only the first N objects (smoke test)")
    args = ap.parse_args()

    cfg = cfgmod.load(args.config)
    seed = cfgmod.set_seeds(cfg)
    hidden = int(cfg["gru"]["hidden"])
    layers = int(cfg["gru"]["layers"])
    epochs = int(cfg["gru"]["epochs"])
    batch = int(cfg["gru"]["batch"])
    subsample = int(cfg["gru"]["subsample"])
    lr = float(cfg["gru"]["lr"])
    C = float(cfg["single_frame"]["C"])
    max_iter = int(cfg["single_frame"]["max_iter"])
    wait = float(cfg["decode"]["nominal_wait_seconds"])
    lag = int(cfg["decode"]["lag_frames"])
    iou = tuple(float(x) for x in cfg["evaluation"]["iou_thresholds"])
    tolerances = tuple(float(x) for x in cfg["evaluation"]["boundary_tolerances"])
    n_vol_seeds = int(cfg["seeds"]["volume_match_seeds"])
    W = int(cfg["data"]["history_frames"])

    g = np.load(os.path.join(args.work_dir, "grid.npz"), allow_pickle=True)
    gep = np.array([str(x) for x in g["grid_ep"]])
    gt = g["grid_t"].astype(float)
    glab = g["grid_label"].astype(int)
    gdt = np.array([str(x) for x in g["grid_date"]])
    is_bg = g["is_bg"].astype(bool)
    ALL = g["ALL"].astype(np.float64)
    pact_oh = g["pact_oh"].astype(np.float64)
    probs_sf_lodo = g["probs_sf"].astype(np.float64)
    act_names = [str(x) for x in g["act_names"]]
    classes = [str(x) for x in g["classes"]]
    n = len(gt)
    K = len(classes)
    A = len(act_names)

    ob = np.load(os.path.join(args.work_dir, "objects.npz"), allow_pickle=True)
    obj_label = np.array([str(x) for x in ob["obj_label"]])
    real_objects = [str(x) for x in ob["real_objects"]]
    with open(os.path.join(args.work_dir, "objects.json"), "r") as fh:
        meta = json.load(fh)
    obj_episodes = {o: set(meta["object_episodes"][o]) for o in real_objects}
    if args.max_objects:
        real_objects = real_objects[:args.max_objects]

    idx = {c: i for i, c in enumerate(classes)}
    idxA, idxP = idx["Align"], idx["Push"]
    idxL, idxRel, idxG = idx["Lift"], idx["Release"], idx["Grasp"]

    print(f"grid={n}  K={K}  A={A}  objects={len(real_objects)}  seed={seed}")

    order = protocol.order_index(gep, gt)
    X_all = protocol.causal_history(ALL, gep, gt, W)
    act_lab = np.array([act_names.index(a) if a in act_names else -1
                        for a in np.array([str(x) for x in g["grid_pact"]])])

    results = {}
    volmatch = {}
    t0 = time.time()
    for oi, o in enumerate(real_objects):
        runs = protocol.object_runs(gep, gt, obj_label, is_bg, obj_episodes, o)
        epseq = protocol.episode_sequences(gep, gt, obj_episodes, o)
        train, test_frames = protocol.fold_masks(gep, is_bg, obj_label, obj_episodes, o)
        trA = train[act_lab[train] >= 0]

        seqs_tr = protocol.training_sequences(gep, order, train, glab)
        A_o = protocol.transition_matrix(glab, seqs_tr, K)
        cw_o = protocol.class_weights(glab[train], K)

        sf_te = models.fit_single_frame(ALL[train], glab[train], ALL[test_frames], K,
                                        C=C, seed=seed, max_iter=max_iter)[0]

        # a factory, so the model is initialised after the subsample draw
        m_ac = models.train_gru(
            lambda: CausalGRU(ALL.shape[1], hidden, A, layers=layers),
            X_all[trA], act_lab[trA], epochs=epochs, batch=batch,
            subsample=subsample, lr=lr, n_classes=A)
        Pact = models.predict_proba(m_ac, X_all[test_frames])
        Pact_full = np.zeros((n, A))
        Pact_full[test_frames] = Pact

        Xtr_soft = np.hstack([ALL[trA], Pact_full[trA]])
        Xte_soft = np.hstack([ALL[test_frames], Pact_full[test_frames]])
        soft_te, clf_soft, sc_soft = models.fit_single_frame(
            Xtr_soft, glab[trA], Xte_soft, K, C=C, seed=seed, max_iter=max_iter)

        rng = np.random.RandomState(0)
        shuf = Pact.copy()
        rng.shuffle(shuf)
        shuf_te = models.predict_softmax(clf_soft, sc_soft,
                                         np.hstack([ALL[test_frames], shuf]), K)

        Xtr_sfa = np.hstack([ALL[trA], pact_oh[trA]])
        Xte_sfa = np.hstack([ALL[test_frames], pact_oh[test_frames]])
        sfa_te = models.fit_single_frame(Xtr_sfa, glab[trA], Xte_sfa, K,
                                         C=C, seed=seed, max_iter=max_iter)[0]

        emissions = {}
        for name, em in [("SF|lag3", sf_te), ("GRUctx|lag3", soft_te),
                         ("Shufctx|lag3", shuf_te), ("SFA_Oracle|lag3", sfa_te)]:
            full = np.zeros((n, K))
            full[test_frames] = common.prior_correct(em, cw_o)
            emissions[name] = full

        o_res = {}
        for name, em in emissions.items():
            pred = protocol.decode(em, epseq, A_o, "lag", lag=lag)
            r, conf = protocol.evaluate_runs(pred, runs, glab, gt, K, iou=iou,
                                             tolerances=tolerances)
            bf5 = protocol.boundary_f1_full(pred, epseq, glab, gt, 0.5)
            bf2 = protocol.boundary_f1_full(pred, epseq, glab, gt, 0.2)
            latf = protocol.latency_full(pred, epseq, glab, gt, wait)
            trf = protocol.transitions_full(pred, epseq, glab,
                                            [(idxA, idxP, "Align->Push"),
                                             (idxL, idxP, "Lift->Push"),
                                             (idxRel, idxG, "Release->Grasp")])
            o_res[name] = dict(seg_f1=r["seg_f1"], macro_f1=r["macro_f1"],
                               seg_f1_75=r["seg_f1_75"],
                               bnd_F1_0_5=r["bnd_F1@0.5"], bnd_F1_0_2=r["bnd_F1@0.2"],
                               bnd_F1_full_0_5=bf5, bnd_F1_full_0_2=bf2,
                               per_class_f1=r["per_class_f1"], conf=conf.tolist(),
                               latency_full=latf, transitions_full=trf,
                               align_push_det=int(trf["Align->Push"][0]),
                               align_push_total=int(trf["Align->Push"][1]),
                               lift_push_det=int(trf["Lift->Push"][0]),
                               lift_push_total=int(trf["Lift->Push"][1]),
                               release_grasp_det=int(trf["Release->Grasp"][0]),
                               release_grasp_total=int(trf["Release->Grasp"][1]))
        o_res["SFA_Oracle|lag3"]["note"] = (
            "requires the true parent action at every frame; object-unseen training "
            "with oracle context, conditional upper bound only")

        # episode-level pairing (SF vs GRUctx), one decode per episode
        runs_by_ep = defaultdict(list)
        for key, ridx in runs:
            runs_by_ep[key.split("_")[0]].append((key, ridx))
        ep_paired = []
        for ep, eidx in epseq:
            pred_sf = protocol.decode(emissions["SF|lag3"], [(ep, eidx)], A_o, "lag", lag=lag)
            pred_gru = protocol.decode(emissions["GRUctx|lag3"], [(ep, eidx)], A_o, "lag", lag=lag)
            r_sf, _ = protocol.evaluate_runs(pred_sf, runs_by_ep[ep], glab, gt, K,
                                             iou=iou, tolerances=tolerances)
            r_gru, _ = protocol.evaluate_runs(pred_gru, runs_by_ep[ep], glab, gt, K,
                                              iou=iou, tolerances=tolerances)
            ep_paired.append(dict(episode=ep,
                                  segF1_SF=float(r_sf["seg_f1"]),
                                  segF1_GRUctx=float(r_gru["seg_f1"]),
                                  dSegF1=float(r_gru["seg_f1"] - r_sf["seg_f1"])))
        o_res["episode_paired"] = ep_paired

        # visible-object reference on the same test frames (leave-one-date-out)
        em_lodo = probs_sf_lodo[test_frames].copy()
        gdt_tf = gdt[test_frames]
        for di, dname in enumerate([str(x) for x in g["dates"]]):
            m = gdt_tf == dname
            if m.any():
                em_lodo[m] = common.prior_correct(em_lodo[m], np.array(g["cweights"][di]))
        full_lodo = np.zeros((n, K))
        full_lodo[test_frames] = em_lodo
        pred_lodo = protocol.decode(full_lodo, epseq, A_o, "lag", lag=lag)
        r, _ = protocol.evaluate_runs(pred_lodo, runs, glab, gt, K, iou=iou,
                                      tolerances=tolerances)
        o_res["LODO_SF|lag3(seen)"] = dict(
            seg_f1=r["seg_f1"], macro_f1=r["macro_f1"],
            bnd_F1_0_5=r["bnd_F1@0.5"], bnd_F1_0_2=r["bnd_F1@0.2"],
            bnd_F1_full_0_5=protocol.boundary_f1_full(pred_lodo, epseq, glab, gt, 0.5),
            bnd_F1_full_0_2=protocol.boundary_f1_full(pred_lodo, epseq, glab, gt, 0.2))

        # volume-matched control: visible object, same number of training episodes
        n_train_ep = len(obj_episodes[o])
        # sorted, so the sample does not depend on set iteration order (hash seed)
        all_eps = sorted(set(gep[~is_bg].tolist()))
        vm_seg = []
        for sd in range(n_vol_seeds):
            rng_v = random.Random(sd)
            sample = set(rng_v.sample(all_eps, n_train_ep))
            tr_vm = np.where((~is_bg) & np.isin(gep, list(sample)))[0]
            sf_vm = models.fit_single_frame(ALL[tr_vm], glab[tr_vm], ALL[test_frames],
                                            K, C=C, seed=seed, max_iter=max_iter)[0]
            full_vm = np.zeros((n, K))
            full_vm[test_frames] = common.prior_correct(
                sf_vm, protocol.class_weights(glab[tr_vm], K))
            seqs_vm = protocol.training_sequences(gep, order, tr_vm, glab)
            pred_vm = protocol.decode(full_vm, epseq, protocol.transition_matrix(glab, seqs_vm, K), "lag", lag=lag)
            r, _ = protocol.evaluate_runs(pred_vm, runs, glab, gt, K, iou=iou,
                                          tolerances=tolerances)
            vm_seg.append(r["seg_f1"])
        volmatch[o] = dict(mean_seg_f1=float(np.mean(vm_seg)),
                           std_seg_f1=float(np.std(vm_seg)),
                           seeds=vm_seg, train_ep_size=n_train_ep)

        results[o] = o_res
        print(f"[{oi + 1:2d}/{len(real_objects)}] {o:14s} "
              f"SF={o_res['SF|lag3']['seg_f1']:.4f} "
              f"GRUctx={o_res['GRUctx|lag3']['seg_f1']:.4f} "
              f"Shuf={o_res['Shufctx|lag3']['seg_f1']:.4f} "
              f"SFAor={o_res['SFA_Oracle|lag3']['seg_f1']:.4f} "
              f"LODOseen={o_res['LODO_SF|lag3(seen)']['seg_f1']:.4f} "
              f"VolMatch={volmatch[o]['mean_seg_f1']:.4f}  "
              f"wall={time.time() - t0:.0f}s", flush=True)

    out = dict(
        protocol="leave-one-object-out: each object's labelled frames are removed from "
                 "training and only those frames are scored; SF, GRU context, shuffle "
                 "control, oracle, transitions and class weights are refitted per fold",
        objects=real_objects, K=K, A=A, n_frames=int(n),
        subset=int(args.max_objects) if args.max_objects else None,
        seeds=dict(global_seed=seed, volume_match_seeds=n_vol_seeds),
        hparams=dict(gru_hidden=hidden, layers=layers, history_frames=W,
                     epochs=epochs, batch=batch, subsample=subsample, lr=lr,
                     single_frame_C=C),
        decode=dict(lag_frames=int(cfg["decode"]["lag_frames"]), nominal_wait_s=wait),
        note_latency="actual output time = signed boundary error + fixed wait; "
                     "absolute error and wait are not added together",
        results=results, volume_match=volmatch,
    )
    path = os.path.join(args.work_dir, "l2_eval.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"saved {path}  total wall={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
