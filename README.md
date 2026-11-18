# Readout Sensitivity and Temporal Evaluation in Sensor-Based Robot Skill Segmentation for Contact-Rich Assembly

Reference implementation for the manuscript of that title. The manuscript is not published, so
this repository does not cite it as a publication; see [CITATION.cff](CITATION.cff).

This is a **measurement study**, not a new model. It asks two questions about a number that is
routinely reported as if it measured perception:

1. How much of the segmental F1 on this corpus is fixed by the annotation and the evaluation grid
   rather than by the model?
2. Of what remains, does changing the **readout** on a frozen encoder move the score more than
   adding **hierarchical context** to the input?

The answers, on the REASSEMBLE corpus, are: a substantial and quantifiable part; and yes, by a
wide margin, but with an interval wider than the pre-registration allowed, and with seed
variation large enough that the honest summary is a lower bound rather than a point estimate.

The readout recipe itself is not new. Decoupled classifier retraining, logit adjustment and
linear-probe-then-finetune are established; what is new here is the pre-registered, held-out
quantification of how much that layer moves this benchmark, and the audit of what the remaining
error is made of.

---

## Results

Official test split: 37 episodes, one pass, opened after the protocol was frozen. Scores are
episode-level segmental F1 (definition in [docs/SCORING.md](docs/SCORING.md)), averaged over three
seeds inside each episode.

| Arm | What it is | segF1 |
|---|---|---|
| **A** | causal GRU (72→48) trained jointly with a softmax skill head | 0.7001 |
| **P** | the same seed's trained GRU, **frozen**, with a class-balanced logistic regression on its 48-d hidden state | **0.7477** |
| **S** | single-frame class-balanced logistic regression on the 72-d frame features, no encoder | 0.6619 |

Paired episode-level effect, P − A:

| Quantity | Value |
|---|---|
| mean difference | **+4.7555 percentage points** |
| 95% bootstrap interval (4,000 episode resamples) | **[+3.4741, +5.9937] pp** |
| interval width | 2.5196 pp |
| episodes improved | **35 / 37** |
| t = 7.23, Wilcoxon p = 7.2e-8, sign test p = 5.1e-9 | |

Three points belong next to that table, not after it.

**The pre-registration had a width condition and the result violates it.** The protocol fixed a
maximum interval width of 2.0 percentage points (0.020 in fraction terms). The observed width is
2.52 pp (0.0252). The ordered decision rule that
was implemented tests effect size, interval sign and episode count — all three pass, so the code
records `CONFIRMED` — while the width condition sits in the prose of the "otherwise" branch. Treat
the width as a confirmation condition and the same data are `INCONCLUSIVE`. **Both readings are
reported; neither is suppressed.** `scripts/40_decide.py` prints both, and
`results/decision_record.json` stores both.

**Seed heterogeneity is not small.** Per-seed differences are +8.85, +1.47 and +3.95 pp. Arm A's
episode mean varies by 3.00 pp across seeds while P varies by 0.81 pp, so much of the effect is
the joint model being unstable rather than the probe being better. The defensible lower bound is
that **24 of 37 episodes improve under all three seeds**; the +8.85 figure must not be quoted as
the effect.

**This is a same-dataset result.** The 37 test episodes come from the same five recording days and
the same 17 objects as the training data. There is no claim of cross-day, cross-session or
cross-embodiment generalisation, and no claim of a new algorithm.

Full tables, per-class scores and the secondary diagnostics are in [docs/RESULTS.md](docs/RESULTS.md).

---

## What the numbers can and cannot mean

The audit in the manuscript rests on a handful of measurements, most of them label-side and none of
them requiring a trained model:

| Observation | Development split | Test split |
|---|---|---|
| Release segments occupying ≤ 2 grid frames | 54.9 % | 64.5 % |
| Lift segments occupying ≤ 2 grid frames | 38.7 % | 45.1 % |
| all segments, ≤ 2 grid frames | 14.9 % | 17.1 % |
| long Release segments (≥ 6 frames) matched at IoU 0.5 | 28.0 % | 7.3 % |
| frame share of long Release segments predicted as `Grasp` | 59.4 % | 66.3 % |

A segment shorter than the 0.4 s window contributes **no frame at all** to the 5 Hz grid. That is
not a model failure, and no readout can recover it. The last two rows are diagnostics: they show
that long `Release` segments are largely predicted as `Grasp`, but not why. Separating an
annotation effect from a model effect would need a double-annotated subset, which does not exist,
so the manuscript states this as a bound on what the score can mean rather than as a finding about
the annotator.

What this repository does **not** support: that `Release` annotations are wrong (no double-annotated
subset was collected), that the readout would help on another corpus, or that any number here is
comparable to a published F1@τ from a vision-based temporal action segmentation baseline — the
class inventory, the metric construction and the input modality all differ.

---

## Dataset

[REASSEMBLE](https://tuwien-asl.github.io/REASSEMBLE_page/) — a contact-rich assembly corpus with
9 low-level skills (`Align`, `Approach`, `Grasp`, `Lift`, `Nudge`, `Pull`, `Push`, `Release`,
`Twist`), 4 high-level actions, 17 assembly objects, 148 episodes over 5 recording days.

**The corpus is not redistributed here.** Download it from the official release
(`git clone https://github.com/TUWIEN-ASL/REASSEMBLE.git`, dataset paper arXiv:2502.05086) and
point the code at it:

```
REASSEMBLE/
  data/data/<episode>.h5          # per-episode HDF5: robot_state, timestamps, segments_info
  splits/train_split1.txt         # 111 episodes, shipped with the dataset
  splits/test_split1.txt          # 37 episodes
```

The official split files are part of the dataset release. The 37 test episodes used here are
exactly those in `test_split1.txt`; the holdout was not chosen by this project.

Only robot state is read: joint positions and velocities, gripper positions, velocities and
efforts (60 dims), and measured force and torque (12 dims). No camera, audio, event or
object-identity signal is used at inference time.

---

## Installation

Python 3.10 or newer, CPU only.

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                                   # optional, exposes the skillseg package
```

`torch` is needed for the recurrent arms only. The scoring, configuration and metric modules
import without it, so `pytest -k "not arms"` runs on a machine without torch.

---

## Running the pipeline

The chain is four steps plus a label-side diagnostic. `--dataset-root` may be replaced by
`SKILLSEG_DATASET_ROOT`.

```bash
# 0. environment and protocol sanity
python scripts/00_check_env.py --dataset-root /data/REASSEMBLE

# 1. build the 5 Hz grid, 72-d features and the causal stack for each split
python scripts/10_build_grid.py --split train --dataset-root /data/REASSEMBLE
python scripts/10_build_grid.py --split test  --dataset-root /data/REASSEMBLE

# 2. train A, freeze it, fit P and S, write frozen predictions (once per seed)
for s in 0 1 2; do
  python scripts/20_train_predict.py --seed-index $s --dataset-root /data/REASSEMBLE
done

# 3. score the frozen predictions (single pass over the holdout); reads artifacts only
python scripts/30_score.py

# 4. apply the pre-registered rule; prints both readings of the width clause
python scripts/40_decide.py

# label-side diagnostic used in the measurement section
python scripts/50_censoring.py --split test --dataset-root /data/REASSEMBLE
```

Generated grids, features, the memory-mapped stack and the predictions land in `./artifacts`, which
is not committed. Steps 3 and 4 write `results/scores.json` and `results/decision.json`; the two
JSON files already in `results/` are the frozen record, so a local run overwrites them — commit that
only if you intend to replace the record.

Runtime on one CPU core, development split of 111 episodes / 185,451 frames: grid and features
about 1 minute, one seed of training about 4 minutes, scoring under a minute.

---

## The three arms, and why A/P is a readout comparison

```
                      15 x 72 causal history
                                |
                     GRU(72 -> 48)  <-- trained once per seed (arm A)
                                |
             +------------------+-------------------+
             |                                      |
       Linear(48 -> 9)                    frozen 48-d hidden state
       trained jointly                            |
       with the encoder                  LogisticRegression (balanced)
             |                                      |
           arm A                                  arm P

      72-d frame features (no history, no encoder)
                    |
          LogisticRegression (balanced)
                    |
                  arm S
```

Arm P is **not** an independently trained model. It is arm A's encoder, frozen, with the readout
replaced. Both linear readouts are fitted on the same rows. The code enforces this rather than
asserting it:

* `freeze_encoder` sets `requires_grad_(False)` and evaluation mode;
* `ProbeFit` records the encoder fingerprint and the row fingerprint at fit time;
* `assert_paired` recomputes the encoder fingerprint afterwards and raises if it moved;
* `verify_seed_pairing` rejects a seed set whose encoders, heads or probes repeat, or whose
  training rows differ.

`tests/test_arms.py` corrupts the encoder after the probe is fitted and requires the pairing check
to fail.

---

## Scoring in one paragraph

A **cell** is one `(episode, object)` pair, cut into runs at object changes and background gaps.
Within a cell, ground-truth and predicted runs are matched one-to-one **within the same class** by
frame-index IoU, greedily by best IoU, and a match counts at IoU ≥ 0.5. The cell score is
`2TP / (2TP + FP + FN)`. An **episode score** is the unweighted mean over the cells that exist in
that episode — an episode is not penalised for objects it never touches — and the arm's score is
the mean over episodes after averaging the three seeds inside each episode. The paired effect is
the mean episode difference with a 4,000-resample percentile bootstrap over episodes. Episodes are
the statistical unit; the 473 cells and the 3 × 37 seed-episode pairs are diagnostics.

Precise definitions, the two aggregation choices that are easy to get wrong, and the defects
inherited from the frozen implementation are in [docs/SCORING.md](docs/SCORING.md).

---

## Reproducibility status

Not everything here was re-run, and the difference matters.

| Layer | Status | How it was checked |
|---|---|---|
| grid, labels, background frames, 72-d features | **REPRODUCED** | rebuilt from the corpus and compared bit-for-bit with the grid the frozen run consumed: 111 episodes, 185,451 frames, times, labels, background mask and features all identical (max abs diff 0.0) |
| cell/episode aggregation and the paired interval | **REPRODUCED** | recomputed from the frozen per-episode values: Δ, both interval bounds, width, improved count and t statistic all match to 0.0 |
| training, holdout inference, per-class and pooled scores | **FROZEN_VERIFIED** | values taken from the frozen run; the holdout was opened once and is now consumed, so it was not re-run |
| secondary diagnostics (grid censoring, long-`Release` confusion) | **FROZEN_VERIFIED** | values from the frozen run |
| the earlier hierarchical-context study in `legacy/` | **REPORTED_ONLY** | archived; its numbers are not reproduced by this code and are not part of the current claim |

Running `scripts/90_verify_dev_grid.py` against a reference grid reproduces the first row. The
remaining rows cannot be re-run without training again and re-opening a consumed holdout, and this
repository does not claim otherwise.

---

## Known defects and limitations

These are recorded because a reader who "fixes" them will no longer reproduce the published
numbers.

1. **Transition matrix and background frames.** The transition matrix is estimated from the full
   label sequence, in which background frames carry `-1`. Negative indexing maps those frames onto
   the **last skill row**, so that row accumulates the outgoing transitions of background frames.
   This is the behaviour of the frozen implementation and it is preserved deliberately. It is a
   defect, not a modelling choice; it is pinned by a test and documented in
   [docs/SCORING.md](docs/SCORING.md).
2. **Cells are equally weighted, not frame-weighted.** A one-frame visit counts as much as a
   ten-second one. The pooled score is reported alongside so the difference is visible.
3. **Object-level leave-one-out is not leave-one-domain-out.** The development protocol holds out
   one assembly object at a time, but the same objects appear across recording days, so a
   day-held-out generalisation claim cannot be identified from it. The manuscript says
   "object-level LOO" and never "LODO".
4. **One annotator, one corpus, one modality.** No inter-annotator agreement was measured, so
   statements about the annotation are statements about this annotation, not about the physical
   event.
5. **Seed variation is reported, not eliminated.** Three seeds is what the pre-registration fixed;
   the per-seed spread is part of the result.

---

## Legacy: the earlier hierarchical-context study

`legacy/context_study/` contains the previous version of this repository, which evaluated whether
a predicted parent-action posterior improves low-level skill recognition. Its headline claim — a
gain in segmental F1 from adding context — is **not** part of the present manuscript and is not a
positive result here. In the development protocol used for the current study, adding hierarchical
context changes the score by **−0.84 pp** (negative on all five recording days once seeds are
averaged, though not in every individual seed), while replacing the readout on the same folds
changes it by **+1.75 pp** (44 of 51 fold-seed units positive).

The archived code is kept for provenance. It is not maintained, it is not covered by the current
tests, and it should not be read as supporting the current paper's claims.

---

## Tests

```bash
pytest                     # 50 tests, no dataset needed
pytest -k "not arms"       # without torch
```

The suite covers run-length matching, the IoU threshold, one-to-one behaviour, cell and episode
aggregation, pooled-versus-episode divergence, bootstrap reproducibility, prior correction, the
fixed-lag decoder, the transition-matrix defect, the A/P pairing guarantees, the decision rule
including the width conflict, and the censoring diagnostic. It does not train on real data and it
does not touch the holdout.

---

## Licence

No licence has been declared. The code is published for inspection and research use; until the
author adds a licence, reuse requires permission. The REASSEMBLE corpus is governed by its own
terms and is not covered by anything in this repository.

## Contact

Issues and questions: open an issue on this repository.
