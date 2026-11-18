#!/usr/bin/env python
"""Step 11 - five-model comparison under an identical fold protocol.

All five schemes share the fold masks, the transition matrix, the class weights,
the decoder and the training budget; only the model and the way the parent
action enters the decision differ.

    SF|lag3        single-frame logistic regression, 72 dims, no history,
                   no action label
    GRUskill|lag3  causal GRU over three seconds, skill head only
    DualHead|lag3  same trunk, skill head plus an action head trained as an
                   auxiliary loss (weight 1.0), no explicit fusion
    GRUctx|lag3    causal GRU predicting the parent action, its posterior
                   appended to the 72-dim feature for a linear skill classifier
    GateCtx|lag3   SF and GRUctx posteriors fused by a two-parameter gate fitted
                   on the training fold only

Output: ``<work-dir>/algo_round.json``
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from crps import common, config as cfgmod, models, protocol  # noqa: E402
from crps.models import CausalGRU, DualHeadGRU  # noqa: E402


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
    aux_weight = float(cfg["dual_head"]["action_aux_weight"])
    gate_epochs = int(cfg["gate"]["epochs"])
    gate_lr = float(cfg["gate"]["lr"])
    C = float(cfg["single_frame"]["C"])
    max_iter = int(cfg["single_frame"]["max_iter"])
    iou = tuple(float(x) for x in cfg["evaluation"]["iou_thresholds"])
    lag = int(cfg["decode"]["lag_frames"])
    tolerances = tuple(float(x) for x in cfg["evaluation"]["boundary_tolerances"])
    W = int(cfg["data"]["history_frames"])

    g = np.load(os.path.join(args.work_dir, "grid.npz"), allow_pickle=True)
    gep = np.array([str(x) for x in g["grid_ep"]])
    gt = g["grid_t"].astype(float)
    glab = g["grid_label"].astype(int)
    is_bg = g["is_bg"].astype(bool)
    ALL = g["ALL"].astype(np.float64)
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

    print(f"grid={n}  K={K}  A={A}  objects={len(real_objects)}  seed={seed}")

    order = protocol.order_index(gep, gt)
    X_all = protocol.causal_history(ALL, gep, gt, W)
    grid_pact = np.array([str(x) for x in g["grid_pact"]])
    act_lab = np.array([act_names.index(a) if a in act_names else -1 for a in grid_pact])

    schemes = ["SF|lag3", "GRUskill|lag3", "DualHead|lag3", "GRUctx|lag3", "GateCtx|lag3"]
    results = {}
    t0 = time.time()
    for oi, o in enumerate(real_objects):
        runs = protocol.object_runs(gep, gt, obj_label, is_bg, obj_episodes, o)
        epseq = protocol.episode_sequences(gep, gt, obj_episodes, o)
        train, test_frames = protocol.fold_masks(gep, is_bg, obj_label, obj_episodes, o)
        trA = train[act_lab[train] >= 0]

        seqs_tr = protocol.training_sequences(gep, order, train, glab)
        A_o = protocol.transition_matrix(glab, seqs_tr, K)
        cw_o = protocol.class_weights(glab[train], K)

        Xtr = X_all[trA]
        y_skill = glab[trA]
        y_action = act_lab[trA]
        w_skill = models.balanced_weights(y_skill, K)
        w_action = models.balanced_weights(y_action, A)

        sf_te = models.fit_single_frame(ALL[train], glab[train], ALL[test_frames], K,
                                        C=C, seed=seed, max_iter=max_iter)[0]

        m_sk = CausalGRU(ALL.shape[1], hidden, K, layers=layers)
        models.train_gru(m_sk, Xtr, y_skill, weights=w_skill, epochs=epochs,
                         batch=batch, subsample=subsample, lr=lr)
        gsk_te = models.predict_proba(m_sk, X_all[test_frames])

        m_du = DualHeadGRU(ALL.shape[1], hidden, K, A, layers=layers)
        models.train_gru(m_du, Xtr, y_skill, y_aux=y_action, weights=w_skill,
                         aux_weights=w_action, epochs=epochs, batch=batch,
                         subsample=subsample, lr=lr, aux_weight=aux_weight)
        with torch.no_grad():
            hs, _ = m_du(torch.as_tensor(X_all[test_frames], dtype=torch.float32))
            gdu_te = torch.softmax(hs, 1).numpy().astype(np.float32)

        m_ac = CausalGRU(ALL.shape[1], hidden, A, layers=layers)
        models.train_gru(m_ac, Xtr, y_action, weights=w_action, epochs=epochs,
                         batch=batch, subsample=subsample, lr=lr)
        pact_te = models.predict_proba(m_ac, X_all[test_frames])
        pact_tr = models.predict_proba(m_ac, Xtr)

        gctx_te, clf_ctx, sc_ctx = models.fit_single_frame(
            np.hstack([ALL[trA], pact_tr]), glab[trA],
            np.hstack([ALL[test_frames], pact_te]), K, C=C, seed=seed, max_iter=max_iter)
        gctx_tr = models.predict_softmax(clf_ctx, sc_ctx,
                                         np.hstack([ALL[trA], pact_tr]), K)
        sf_tr = models.fit_single_frame(ALL[train], glab[train], ALL[trA], K,
                                        C=C, seed=seed, max_iter=max_iter)[0]
        w1, w2, b = models.gate_train(sf_tr, gctx_tr, pact_tr, glab[trA],
                                      epochs=gate_epochs, lr=gate_lr)
        gate_te = models.gate_apply(sf_te, gctx_te, pact_te, w1, w2, b)

        emissions = {"SF|lag3": sf_te, "GRUskill|lag3": gsk_te, "DualHead|lag3": gdu_te,
                     "GRUctx|lag3": gctx_te, "GateCtx|lag3": gate_te}

        o_res = {"gate_params": dict(w1=w1, w2=w2, b=b)}
        for name, em in emissions.items():
            full = np.zeros((n, K))
            full[test_frames] = common.prior_correct(em, cw_o)
            pred = protocol.decode(full, epseq, A_o, "lag", lag=lag)
            r, conf = protocol.evaluate_runs(pred, runs, glab, gt, K, iou=iou,
                                             tolerances=tolerances)
            o_res[name] = dict(seg_f1=r["seg_f1"], macro_f1=r["macro_f1"],
                               seg_f1_75=r["seg_f1_75"],
                               per_class_f1=r["per_class_f1"], conf=conf.tolist(),
                               bnd_F1_full_0_5=protocol.boundary_f1_full(
                                   pred, epseq, glab, gt, 0.5))

        runs_by_ep = defaultdict(list)
        for key, ridx in runs:
            runs_by_ep[key.split("_")[0]].append((key, ridx))
        ep_paired = []
        for ep, eidx in epseq:
            row = dict(episode=ep)
            for name, em in emissions.items():
                full = np.zeros((n, K))
                full[test_frames] = common.prior_correct(em, cw_o)
                pred = protocol.decode(full, [(ep, eidx)], A_o, "lag", lag=lag)
                r, _ = protocol.evaluate_runs(pred, runs_by_ep[ep], glab, gt, K,
                                              iou=iou, tolerances=tolerances)
                row[f"segF1_{name.split('|')[0]}"] = float(r["seg_f1"])
            ep_paired.append(row)
        o_res["episode_paired"] = ep_paired

        results[o] = o_res
        print(f"[{oi + 1:2d}/{len(real_objects)}] {o:14s} "
              f"SF={o_res['SF|lag3']['seg_f1']:.4f} "
              f"GRUsk={o_res['GRUskill|lag3']['seg_f1']:.4f} "
              f"Dual={o_res['DualHead|lag3']['seg_f1']:.4f} "
              f"GRUctx={o_res['GRUctx|lag3']['seg_f1']:.4f} "
              f"Gate={o_res['GateCtx|lag3']['seg_f1']:.4f}  "
              f"wall={time.time() - t0:.0f}s", flush=True)

    out = dict(
        protocol="five-model comparison under an identical leave-one-object-out "
                 "protocol; fold masks, transition matrix, class weights and "
                 "decoder are shared, only the model differs",
        objects=real_objects, K=K, A=A, n_frames=int(n),
        subset=int(args.max_objects) if args.max_objects else None,
        hparams=dict(gru_hidden=hidden, layers=layers, history_frames=W,
                     epochs=epochs, batch=batch, subsample=subsample, lr=lr,
                     dualhead_action_aux_weight=aux_weight,
                     single_frame_C=C, gate_epochs=gate_epochs, gate_lr=gate_lr,
                     global_seed=seed),
        results=results,
    )
    path = os.path.join(args.work_dir, "algo_round.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"saved {path}  total wall={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
