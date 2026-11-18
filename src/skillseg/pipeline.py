"""End-to-end orchestration.

One seed's pipeline is: train A on the development split, freeze the encoder, fit
P on its hidden state, fit S on raw features, score the untouched episodes once.
The split between :func:`train_and_predict` (which never reads test labels) and
:func:`score_predictions` (which never reads the model) is deliberate: the
predictions are written to disk and hashed before any label is opened.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from . import decode, metrics, models, objects
from .config import Config
from .data import Grid, build_grid
from .features import causal_stack, episode_sequences, frame_features


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


# --------------------------------------------------------------- split preparation
def prepare_split(cfg: Config, which: str, out_dir: str | Path,
                  verbose: bool = True) -> Dict[str, Any]:
    """Build grid, features and causal stack for one official split."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = cfg.dataset.split(which)
    grid = build_grid(episodes, cfg, verbose=verbose)
    feats = frame_features(grid, cfg, verbose=verbose)
    stack_path = out_dir / f"{which}_stack.npy"
    stack = causal_stack(feats, episode_sequences(grid), cfg.history_frames, stack_path)

    grid.save(out_dir / f"{which}_grid.npz")
    np.save(out_dir / f"{which}_frames.npy", feats)
    meta = {
        "split": which,
        "n_episodes": len(grid.episodes),
        "n_frames": len(grid),
        "n_annotated": int((~grid.is_bg).sum()),
        "n_background": int(grid.is_bg.sum()),
        "n_eval_frames": int(((~grid.is_bg) & (grid.obj != "action")).sum()),
        "n_objects": len(objects.real_objects(grid)),
        "dates": sorted(set(grid.date.tolist())),
        "stack_bytes": int(stack.nbytes),
    }
    if verbose:
        print(json.dumps(meta, indent=1, ensure_ascii=False))
    return meta


def load_split_artifacts(cfg: Config, out_dir: str | Path, which: str
                         ) -> tuple[Grid, np.ndarray, np.ndarray]:
    out_dir = Path(out_dir)
    grid = Grid.load(out_dir / f"{which}_grid.npz")
    frames = np.load(out_dir / f"{which}_frames.npy")
    stack = np.load(out_dir / f"{which}_stack.npy", mmap_mode="r")
    return grid, frames, stack


# --------------------------------------------------------------- one seed
def training_rows(grid: Grid, cfg: Config) -> np.ndarray:
    """Annotated frames whose parent action is one of the four known actions."""
    action_set = set(cfg.actions)
    keep_action = np.array([p in action_set for p in grid.pact.tolist()])
    return np.where((~grid.is_bg) & keep_action)[0]


def train_and_predict(cfg: Config, dev_dir: str | Path, test_dir: str | Path,
                      out_dir: str | Path, seed_index: int) -> Dict[str, Any]:
    """Train the arms on development data and freeze predictions for the test split.

    Test labels are not opened here: only the feature stack is read.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_grid, dev_frames, dev_stack = load_split_artifacts(cfg, dev_dir, "train")
    test_grid, test_frames, test_stack = load_split_artifacts(cfg, test_dir, "test")

    rows = training_rows(dev_grid, cfg)
    arms = models.build_arms(dev_stack, dev_frames, dev_grid.label, rows, cfg, seed_index)
    models.assert_paired(arms)

    weights = decode.class_weights(dev_grid.label[~dev_grid.is_bg], cfg.n_skills)
    dev_sequences = episode_sequences(dev_grid)
    transition = decode.transition_matrix(dev_grid.label, dev_sequences, cfg.n_skills)
    test_sequences = episode_sequences(test_grid)

    posteriors = models.arm_probabilities(arms, test_stack, test_frames, cfg.n_skills)
    predictions = {}
    for arm, p in posteriors.items():
        corrected = decode.prior_correct(p.astype(np.float64), weights)
        predictions[arm] = decode.decode_episodes(corrected, test_sequences, transition,
                                                  cfg.lag_frames)

    pred_path = out_dir / f"pred_seed{seed_index}.npz"
    np.savez_compressed(
        pred_path,
        ep=test_grid.ep, t=test_grid.t,
        pred_A=predictions["A"], pred_P=predictions["P"], pred_S=predictions["S"],
        prob_A=posteriors["A"].astype(np.float32),
        prob_P=posteriors["P"].astype(np.float32),
        prob_S=posteriors["S"].astype(np.float32),
    )

    info = {
        "seed_index": seed_index,
        "train_seed": arms.train_seed,
        "optimisation_steps": arms.steps,
        "n_train_rows": int(len(rows)),
        "n_test_frames": int(len(test_grid)),
        "encoder_sha256_16": models.params_fingerprint(arms.encoder.gru),
        "head_sha256_16": models.params_fingerprint(arms.encoder.head),
        "probe_sha256_16": models.params_fingerprint(arms.probe.fit),
        "encoder_sha_before_probe": arms.encoder_sha_before_probe,
        "encoder_sha_after_probe": arms.encoder_sha_after_probe,
        "train_rows_sha16": arms.probe.rows_sha,
        "n_rows_head": arms.probe.n_rows,
        "n_rows_probe": arms.probe.n_rows,
        "encoder_frozen": arms.probe.encoder_frozen,
        "probe_input_dim": arms.probe.n_features,
        "class_weights": weights.tolist(),
        "predictions": str(pred_path.name),
        "predictions_sha256": sha256_file(pred_path),
    }
    save_json(out_dir / f"seed{seed_index}.json", info)
    return info


def verify_seed_pairing(infos: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Reject a seed set unless the three seeds are genuinely distinct and comparable."""
    if len(infos) < 2:
        raise AssertionError("at least two seeds are required to check that seeds differ")
    seen = set()
    for info in infos:
        index = info["seed_index"]
        if index in seen:
            raise AssertionError(f"duplicate seed index {index}")
        seen.add(index)
        if info["encoder_sha_before_probe"] != info["encoder_sha_after_probe"]:
            raise AssertionError(f"seed {index}: encoder changed while fitting the probe")
        if info["n_rows_head"] != info["n_rows_probe"]:
            raise AssertionError(f"seed {index}: head and probe saw different row counts")
        if not info["encoder_frozen"]:
            raise AssertionError(f"seed {index}: encoder was not frozen")
    for key in ("encoder_sha256_16", "head_sha256_16", "probe_sha256_16"):
        values = [info[key] for info in infos]
        if len(set(values)) != len(values):
            raise AssertionError(f"{key} repeats across seeds: the seeds did not take effect")
    if len({info["train_rows_sha16"] for info in infos}) != 1:
        raise AssertionError("training rows differ across seeds: the arms are not comparable")
    return {"n_seeds": len(infos), "train_rows_sha16": infos[0]["train_rows_sha16"]}


# --------------------------------------------------------------- scoring
def score_predictions(cfg: Config, grid: Grid, prediction_files: Sequence[str | Path],
                      infos: Sequence[Mapping[str, Any]], iou_threshold: Optional[float] = None
                      ) -> Dict[str, Any]:
    """Score frozen predictions against the labels.  No model is loaded here."""
    threshold = cfg.iou_threshold if iou_threshold is None else iou_threshold
    pairing = verify_seed_pairing(infos)

    for info, path in zip(infos, prediction_files):
        if sha256_file(path) != info["predictions_sha256"]:
            raise RuntimeError(f"{path} does not match the hash recorded at prediction time")

    keep = (~grid.is_bg) & (grid.obj != "action")
    labels = grid.label
    cell_map = objects.cells(grid)
    episodes = sorted({episode for episode, _ in cell_map})
    runs = objects.all_runs(cell_map)

    per_seed: Dict[str, Any] = {}
    loaded: Dict[int, Dict[str, np.ndarray]] = {}
    for info, path in zip(infos, prediction_files):
        index = int(info["seed_index"])
        z = np.load(path, allow_pickle=True)
        loaded[index] = {"A": z["pred_A"], "P": z["pred_P"], "S": z["pred_S"]}
        block: Dict[str, Any] = {}
        for arm in ("A", "P", "S"):
            pred = loaded[index][arm]
            cell = metrics.cell_scores(pred, cell_map, labels, threshold)
            ep = metrics.episode_scores(cell)
            per_class = metrics.per_class_f1(pred, labels, keep, cfg.n_skills)
            block[arm] = {
                "cell_f1": {f"{e}|{o}": v for (e, o), v in cell.items()},
                "episode_f1": ep,
                "episode_mean": float(np.mean(list(ep.values()))),
                "cell_mean": float(np.mean(list(cell.values()))),
                "pooled": metrics.pooled_score(pred, runs, labels, threshold),
                "frame_accuracy": metrics.frame_accuracy(pred, labels, keep),
                "per_class_f1": per_class,
                "per_class_f1_named": {name: per_class[str(i)]
                                       for i, name in enumerate(cfg.skills)},
            }
        per_seed[str(index)] = block

    seeds = sorted(loaded)

    def seed_mean(arm: str, episode: str) -> float:
        return float(np.mean([per_seed[str(s)][arm]["episode_f1"][episode] for s in seeds]))

    a_ep = {ep: seed_mean("A", ep) for ep in episodes}
    p_ep = {ep: seed_mean("P", ep) for ep in episodes}
    s_ep = {ep: seed_mean("S", ep) for ep in episodes}
    paired = metrics.paired_effect(a_ep, p_ep, episodes)
    interval = metrics.bootstrap_interval(paired["delta"], cfg.bootstrap_resamples,
                                          cfg.bootstrap_seed)

    cell_delta = []
    for key in cell_map:
        a_cell = float(np.mean([per_seed[str(s)]["A"]["cell_f1"][f"{key[0]}|{key[1]}"] for s in seeds]))
        p_cell = float(np.mean([per_seed[str(s)]["P"]["cell_f1"][f"{key[0]}|{key[1]}"] for s in seeds]))
        cell_delta.append(p_cell - a_cell)

    per_seed_delta = {}
    for s in seeds:
        block = per_seed[str(s)]
        d = np.array([block["P"]["episode_f1"][ep] - block["A"]["episode_f1"][ep]
                      for ep in episodes])
        per_seed_delta[str(s)] = {
            "delta": float(d.mean()),
            "n_pos": int((d > 0).sum()),
            "A_mean": block["A"]["episode_mean"],
            "P_mean": block["P"]["episode_mean"],
        }

    return {
        "n_episodes": len(episodes),
        "n_cells": len(cell_map),
        "n_runs": len(runs),
        "objects_per_episode": len(cell_map) / max(len(episodes), 1),
        "n_eval_frames": int(keep.sum()),
        "cluster_unit": "episode",
        "n_clusters": len(episodes),
        "pairing": pairing,
        "threshold": threshold,
        "per_seed": per_seed,
        "per_seed_delta": per_seed_delta,
        "paired": paired,
        "bootstrap_ci": list(interval),
        "arm_means": {
            "A": float(np.mean(list(a_ep.values()))),
            "P": float(np.mean(list(p_ep.values()))),
            "S": float(np.mean(list(s_ep.values()))),
        },
        "cell_diagnostic": {
            "n": len(cell_delta),
            "mean": float(np.mean(cell_delta)),
            "sd": float(np.std(cell_delta, ddof=1)),
            "n_pos": int(sum(1 for x in cell_delta if x > 0)),
            "note": "diagnostic only; cells are not independent observations",
        },
        "seed_consistency": {
            "n_episodes_positive_in_every_seed": int(sum(
                1 for ep in episodes
                if all(per_seed[str(s)]["P"]["episode_f1"][ep] > per_seed[str(s)]["A"]["episode_f1"][ep]
                       for s in seeds))),
            "note": "lower bound on the effect under seed variation; not a substitute for the interval",
        },
    }


def decide(cfg: Config, scores: Mapping[str, Any]) -> Dict[str, Any]:
    verdict = metrics.decide(scores["paired"]["delta"], cfg, tuple(scores["bootstrap_ci"]))
    verdict["A_mean"] = scores["arm_means"]["A"]
    verdict["P_mean"] = scores["arm_means"]["P"]
    verdict["S_mean"] = scores["arm_means"]["S"]
    verdict["per_seed_delta"] = scores["per_seed_delta"]
    verdict["seed_consistency"] = scores["seed_consistency"]
    verdict["cluster_unit"] = scores["cluster_unit"]
    verdict["n_clusters"] = scores["n_clusters"]
    return verdict


# --------------------------------------------------------------- label-side diagnostics
def grid_censoring(grid: Grid, segments: Sequence[tuple], cfg: Config) -> Dict[str, Any]:
    """How many native segments survive the 5 Hz grid, per class.

    Reads labels only: no prediction, no model, so it cannot be tuned.
    """
    frames_by_episode: Dict[str, List[int]] = {}
    annotated = np.where(~grid.is_bg)[0]
    for i in annotated:
        frames_by_episode.setdefault(grid.ep[i], []).append(int(i))
    for key in frames_by_episode:
        frames_by_episode[key] = sorted(frames_by_episode[key], key=lambda i: grid.t[i])

    counts: List[tuple] = []
    for episode, skill, start, end in segments:
        idx = frames_by_episode.get(episode, [])
        times = grid.t[idx] if idx else np.array([])
        if len(times) == 0:
            counts.append((skill, 0, end - start)); continue
        inside = (times >= start - 1e-6) & (times <= end + 1e-6)
        counts.append((skill, int(inside.sum()), end - start))

    def block(rows: List[tuple]) -> Dict[str, Any]:
        if not rows:
            return {"n": 0}
        nf = np.array([r[1] for r in rows], dtype=float)
        duration = np.array([r[2] for r in rows], dtype=float)
        return {
            "n": int(len(rows)),
            "share_zero_frames": float((nf == 0).mean()),
            "share_one_frame": float((nf == 1).mean()),
            "share_at_most_two_frames": float((nf <= 2).mean()),
            "share_below_native_iou_half": float(((0.2 * nf / np.maximum(duration, 1e-9)) < 0.5).mean()),
            "median_frames": float(np.median(nf)),
            "median_duration_s": float(np.median(duration)),
        }

    by_class = {}
    for skill in cfg.skills:
        by_class[skill] = block([r for r in counts if r[0] == skill])
    return {"pooled": block(counts), "per_class": by_class, "n_segments": len(counts)}
