# Predicted Hierarchical Action Context for Low-Level Skill Recognition in Robotic Assembly

Reference implementation for the manuscript *Predicted Hierarchical Action Context for Low-Level
Skill Recognition in Robotic Assembly*. The manuscript is not published yet and its author list is
not final, so this repository does not cite it as a publication.

The question is narrow and testable. Low-level manipulation skills such as `Lift`, `Push` and
`Align` look almost identical inside the short observation window an online recogniser can
afford, so a single-frame classifier confuses them. The high-level assembly action the robot is
already executing — `pick`, `insert`, `remove`, `place` — restricts which skills are plausible at
that moment. This code asks whether using that context *predicted online from the same sensor
history*, rather than supplied by hand-written pre-conditions, improves low-level recognition.

The answer reported here is yes, but only partially and only at the label level:

* segmental F1 over held-out-object frames rises from **0.6717** to **0.7135**;
* full-episode boundary F1 does **not** improve (0.5948 vs 0.5894) — the gain is class
  disambiguation, not better temporal localisation;
* supplying the ground-truth parent action instead of predicting it reaches **0.8236**, so the
  remaining headroom is in the quality of the predicted context, not in the fusion step.

All numbers are development-set evidence. A 37-episode sealed holdout exists and was never read;
cross-seed robustness was not evaluated. See [Known limitations](#known-limitations).

---

## Method

```
                      3 s causal history of 72-dim frame features
                                     |
                          causal GRU (48 hidden units)
                                     |
                        parent-action posterior (4 classes)
                                     |
   72-dim frame feature  ---->  concat (76 dims)  ---->  logistic regression  ---->  skill posterior (9 classes)
                                                                                          |
                                                                     fixed-lag Viterbi (3 frames, 0.6 s wait)
                                                                                          |
                                                                                    skill sequence
```

The 72-dimensional frame feature describes a 0.4 s causal window and uses robot state only:

| Block | Channels | Statistics | Dims |
|---|---|---|---|
| Proprioception | joint positions (7), joint velocities (7), gripper positions (2), gripper velocities (2), gripper efforts (2) | mean, standard deviation, net displacement | 60 |
| Force/torque | measured force (3), measured torque (3) | mean, standard deviation | 12 |

No cameras, audio, event data or object identity are used at inference time.

Five variants are compared under an identical protocol (`scripts/11_run_algo_round.py`):

| Scheme | History | Action label | Fusion |
|---|---|---|---|
| `SF` | none | no | none — single-frame logistic regression on 72 dims |
| `GRUskill` | 3 s | no | none — GRU with a skill head |
| `DualHead` | 3 s | as auxiliary loss (weight 1.0) | none — shared trunk, two heads |
| `GRUctx` | 3 s | predicted, appended | feature-level (76 dims) |
| `GateCtx` | 3 s | predicted | score-level, two-parameter confidence gate |

`GRUctx` and `DualHead` are **not** a strict single-variable ablation of each other: they differ in
where the action information enters (input versus auxiliary loss) and in the decision rule (two
stages versus one). Both are reported so the comparison is not read as a controlled one.

## Results

Fold-averaged over the 17 held-out objects. Regenerate with steps 10, 20 and 21 below.

| Scheme | Segmental F1 | Macro F1 | Full-episode boundary F1@0.5 |
|---|---|---|---|
| `SF` | 0.6717 | 0.5341 | 0.5948 |
| `GRUctx` | **0.7135** | **0.5686** | 0.5894 |
| `Shufctx` (context permuted) | 0.4098 | 0.4726 | 0.4872 |
| `SFA_Oracle` (ground-truth context) | 0.8236 | 0.6334 | 0.6220 |
| `LODO_SF` (object seen, same frames) | 0.6683 | 0.5330 | 0.5889 |
| `VolMatch_SF` (object seen, matched volume) | 0.6839 | — | — |

Five-model round, same folds (`scripts/11_run_algo_round.py`):

| Scheme | Segmental F1 | Gain vs `SF` |
|---|---|---|
| `SF` | 0.6717 | — |
| `GRUskill` | 0.6514 | −0.0203 |
| `DualHead` | 0.6468 | −0.0249 |
| `GRUctx` | 0.7091 | +0.0374 |
| `GateCtx` | 0.7093 | +0.0376 |

The two runs are separate. `GRUctx` is 0.7135 in the main run and 0.7091 in the five-model round
because the second script refits every scheme, and rebuilds the parent-action model, in one process;
the difference (0.004) is run-to-run variation, not a change of method. Do not mix rows across the
two tables.

Statistics: the independent unit is the **episode**, not the fold. 97 of 111 episodes improve,
mean Δsegmental F1 = 0.0446, paired Wilcoxon p = 2.0 × 10⁻¹⁴, episode bootstrap 95 % CI
[0.036, 0.054], date-clustered 95 % CI [0.027, 0.071] over five recording dates (date-level Wilcoxon
p = 0.0625, five dates). The 17 folds share most of their test episodes (mean pairwise overlap 75.8
of 86.1 test episodes per fold), so they are correlated replications and not 17 independent tests.

`VolMatch` is the one number that does not match the reported run exactly: it is a five-seed random
control, and this repository draws those seeds from a sorted episode list so the sample does not
depend on `PYTHONHASHSEED`. That moves the mean from 0.6854 to 0.6839.

`results/reference_metrics.json` records these values as reference targets. They are transcribed
from the reported runs, not recomputed by this repository at authoring time — see
[Reproducibility status](#reproducibility-status).

## Dataset

[REASSEMBLE](https://tuwien-asl.github.io/REASSEMBLE_page/) — *a multimodal dataset for contact-rich
robotic assembly and disassembly*.

* Code and download instructions: <https://github.com/TUWIEN-ASL/REASSEMBLE>
* Paper: <https://arxiv.org/abs/2502.05086>

**This repository ships no data.** Download REASSEMBLE from the official source and follow its
licence; the dataset is not redistributed here. Do not commit `.h5` files.

Expected layout after cloning REASSEMBLE into `data/REASSEMBLE`:

```
data/REASSEMBLE/
├── data/
│   └── data/
│       ├── 2025-01-09-13-57-17.h5
│       └── ...                      # one file per recorded episode
└── splits/
    ├── train_split1.txt             # 111 development episodes (used here)
    └── test_split1.txt              # 37 sealed episodes (never read by this code)
```

Only `train_split1.txt` is ever opened. No script reads `test_split1.txt`; the sealed split is listed
above only so you can see it exists and is deliberately left alone.

## Environment

Python 3.10 or newer, CPU only. Tested with Python 3.13, numpy 2.5, scikit-learn 1.9, torch 2.15
(CPU), h5py 3.16, scipy 1.18.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Scripts do not need to be installed; they add `src/` to `sys.path` themselves.

## Pipeline

Every script takes `--work-dir` (default `work/`), and the scripts that read raw recordings also take
`--data-root` (default `data/REASSEMBLE/`), so the commands below run as printed once the dataset
sits in the default location. Derived artefacts stay in `work/`, which is git-ignored.

On a laptop CPU the preprocessing steps take roughly 2, 1.5 and 1 minutes, and the main run about
18 minutes for its 17 folds. The five-model round refits every scheme and takes longer.

| Step | Command | Reads | Writes |
|---|---|---|---|
| 00 | `python scripts/00_build_frames.py` | REASSEMBLE HDF5 | `work/frames.npz` |
| 01 | `python scripts/01_build_grid.py` | `work/frames.npz`, HDF5 | `work/grid.npz` |
| 02 | `python scripts/02_build_objects.py` | `work/frames.npz`, `work/grid.npz`, HDF5 | `work/objects.npz`, `work/objects.json` |
| 10 | `python scripts/10_run_l2.py` | `work/grid.npz`, `work/objects.*` | `work/l2_eval.json` |
| 11 | `python scripts/11_run_algo_round.py` | `work/grid.npz`, `work/objects.*` | `work/algo_round.json` |
| 20 | `python scripts/20_aggregate_l2.py` | `work/l2_eval.json` | `work/l2_analysis.json` |
| 21 | `python scripts/21_aggregate_algo_round.py` | `work/algo_round.json` | `work/algo_round_agg.json` |
| 22 | `python scripts/22_stats_audit.py` | `work/l2_eval.json`, `work/objects.json` | `work/stats_audit.json` |

### Preprocessing

```bash
python scripts/00_build_frames.py    # annotated frames + 72-dim features + leave-one-date-out SF
python scripts/01_build_grid.py      # continuous timeline, background frames, SF and oracle posteriors
python scripts/02_build_objects.py   # assembly-object labels and the 17 object partitions
```

Step 01 prints a cross-check of the single-frame posteriors against step 00. On the full development
split the two fits agree to a mean absolute difference of 8 × 10⁻⁵ and a maximum of 1.2 × 10⁻²,
because the multinomial fit is sensitive to training-frame order at that magnitude. A disagreement
an order of magnitude larger means the feature or the frame inventory changed.

### Training and evaluation

```bash
python scripts/10_run_l2.py          # main run: SF, GRUctx, shuffle control, oracle, two controls
python scripts/11_run_algo_round.py  # five-model comparison on the same folds
python scripts/20_aggregate_l2.py    # fold-averaged table of the main run
python scripts/21_aggregate_algo_round.py
python scripts/22_stats_audit.py     # fold overlap, episode-level significance, metric separation
```

To check that the pipeline runs before committing to a full pass, use a subset:

```bash
python scripts/00_build_frames.py --limit-episodes 6 --work-dir work_smoke
python scripts/01_build_grid.py --work-dir work_smoke
python scripts/02_build_objects.py --limit-episodes 6 --work-dir work_smoke
python scripts/10_run_l2.py --work-dir work_smoke --max-objects 2
python scripts/11_run_algo_round.py --work-dir work_smoke --max-objects 2
python scripts/20_aggregate_l2.py --work-dir work_smoke
python scripts/21_aggregate_algo_round.py --work-dir work_smoke
python scripts/22_stats_audit.py --work-dir work_smoke
```

Subset runs need at least two recording dates, otherwise the leave-one-date-out reference model has
nothing to train on and the script stops with an explicit error. Numbers from a subset are
meaningless; the point is only that every stage runs.

### Hyperparameters and seeds

`configs/default.json` holds every value that affects a reported number and is what the reported
runs used: 0.4 s feature window, 0.2 s stride, 15-frame (3 s) history, 48 hidden units, one GRU
layer, 10 epochs, batch 256, at most 60 000 windows per fold, Adam at 1 × 10⁻³, dual-head auxiliary
weight 1.0, fixed-lag Viterbi with 3 frames and a 0.6 s nominal wait, global seed 0. Pass
`--config path/to/other.json` to override. Each output JSON records the resolved settings so a run
is self-describing.

A single global seed is used everywhere. Cross-seed robustness was **not** evaluated; treat every
difference below ~0.005 as within run-to-run variation.

Determinism notes for anyone comparing against the reported numbers:

* the order in which the GRU and the subsample permutation consume random numbers matters and is
  preserved here, because PyTorch draws the permutation before initialising the model in the main
  run and after it in the five-model round;
* `torch.set_num_threads` changes floating-point reduction order, so pin it if you need agreement
  beyond the third decimal;
* episode and object collections are sorted before sampling, so results do not depend on
  `PYTHONHASHSEED`.

## Evaluation protocol

* **Fold.** One assembly object is held out: its labelled frames are removed from training and only
  those frames are scored. The object's episodes still contribute their other objects' frames, and
  the frames are decoded inside the full episode timeline. This is therefore *target-object frame
  exclusion*, not unseen-object generalisation, and the code does not claim the latter.
* **Refitting.** Scalers, class priors, transition counts, the GRU, the gate and every classifier are
  estimated inside the fold on the other sixteen objects. Nothing is shared across folds.
* **Metrics.**
  * *Segmental F1* — a ground-truth segment counts as a true positive when an unused predicted
    segment of the same class reaches IoU ≥ 0.5. The threshold of 0.75 is also reported.
  * *Macro F1* — unweighted mean of per-class frame-level F1 from the confusion matrix.
  * *Boundary F1* — predicted boundary timestamps matched to ground-truth boundaries within a
    tolerance (0.2 s and 0.5 s). Reported on the full episode timeline, because a held-out object's
    frames form isolated runs with almost no internal boundaries.
  * *Edit distance* — Levenshtein distance between the collapsed label sequences, normalised by the
    longer sequence.
* **Averaging.** Tables average over folds (one value per object), not over frames. A frame-pooled
  per-class view is produced separately by step 21; the two views differ, and the paper reports both
  where they disagree (for example `Push` is 0.102 → 0.152 fold-averaged but 0.420 → 0.585
  frame-pooled).
* **Latency.** Detection latency is decomposed into a signed boundary-time error and a fixed 0.6 s
  decision wait. The absolute error and the wait are *not* summed, because they are not independent.
* **Oracle.** `SFA_Oracle` conditions on the ground-truth parent action at every frame and is a
  conditional upper bound, not a deployable method.

## Reproducibility status

| Check | Status |
|---|---|
| Unit tests (metrics, decoders, features, models, protocol, config) | passing, `python -m pytest tests` |
| Feature builder vs the artefacts that produced the reported numbers | bit-identical on 22 608 frames across 12 episodes spanning all five recording dates |
| Timeline, labels, parent actions, object labels | identical on the same episodes |
| Full eight-stage pipeline on a 6-episode subset | runs end to end |
| Full pipeline, all 17 folds (`scripts/00`–`02`, then `scripts/10`, 1170 s) | **reproduced 0.6717 and 0.7135 exactly**; all five deterministic schemes match the reported per-fold values to the last digit (max difference 0.0 over 17 folds). Only the random volume-matched control differs, by 0.0015, for the reason given above |
| Five-model round (`scripts/11`) | re-run on **2 of the 17 folds** (172 s): all five schemes matched the reported per-fold values exactly. The other 15 folds were not re-run (`FULL_REPRO_NOT_RUN` for the complete round) |
| Sealed holdout confirmation | not executed; the sealed split was never read |

## Repository layout

```
configs/default.json      all hyperparameters and the global seed
src/crps/
  common.py               fixed-lag Viterbi, HMM filtering, metrics
  features.py             72-dimensional proprioceptive / force-torque features
  data.py                 REASSEMBLE HDF5 access, splits, frame enumeration
  grid.py                 continuous timeline with background frames
  objects.py              per-frame object labels, object partitions
  protocol.py             folds, causal history, transitions, decoding, evaluation
  models.py               GRU encoders, single-frame classifier, confidence gate
  config.py               configuration and seeding
scripts/                  00-02 preprocessing, 10-11 experiments, 20-22 aggregation
tests/                    pytest suite, no dataset required
results/                  reference metrics of the reported runs
```

## Tests

```bash
python -m pytest tests
```

The suite needs no dataset: it builds synthetic HDF5 episodes and checks feature widths and window
semantics, decoder causality and fixed-lag equivalence, metric definitions, fold masks, index
alignment and configuration consistency. 67 tests.

## Known limitations

1. **Development split only.** All numbers come from the 111-episode development split. The
   37-episode sealed holdout was not read, so nothing here is a confirmation on unseen recordings.
2. **Single seed.** Cross-seed variance was not measured.
3. **Correlated folds.** The 17 object folds overlap heavily in training and test episodes; they are
   not independent tests, and the audit in step 22 exists to keep that visible.
4. **Label-level gain only.** Boundary F1 does not improve. The effect is disambiguation between
   confusable skills, not better temporal localisation.
5. **Conditional oracle.** `SFA_Oracle` requires the true parent action at every frame and bounds
   what a perfect context signal could deliver; it is not a method.
6. **Not a controlled ablation.** `GRUctx` and `DualHead` differ in two respects at once.
7. **Parent actions are coarse.** They are the verb of the enclosing annotation segment. When that
   verb is absent or malformed the frame is excluded from parent-action training, and background
   frames inherit a sentinel that is never used as an input.

## Citing

See `CITATION.cff`. The author list and the paper reference are placeholders pending confirmation;
please do not cite this as a published paper yet.

## Licence

No licence has been chosen. The code authorship and the terms of the third-party dependencies have
not been confirmed, so this repository deliberately ships without a `LICENSE` file rather than
asserting terms it cannot substantiate. Contact the maintainers before reuse or redistribution.
