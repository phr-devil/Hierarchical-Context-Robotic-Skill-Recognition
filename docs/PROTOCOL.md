# Protocol

Everything here was fixed before the holdout was opened. The configuration file
[`configs/default.json`](../configs/default.json) is the machine-readable copy; this document
explains what each value is for.

---

## Corpus

| Property | Value |
|---|---|
| episodes | 148 (111 development, 37 test) |
| low-level skills | 9: `Align`, `Approach`, `Grasp`, `Lift`, `Nudge`, `Pull`, `Push`, `Release`, `Twist` |
| high-level actions | 4: `insert`, `pick`, `place`, `remove` |
| assembly objects | 17 |
| recording days | 5 |
| native low-level segments | 10,195 (7,743 development, 2,452 test) |
| development grid frames | 185,451 (169,728 annotated, 15,723 background, 8.48 %) |
| test evaluation frames | 48,644 across 473 cells and 2,419 runs (12.8 objects per episode) |

The split is the one shipped with the corpus (`splits/train_split1.txt` and
`splits/test_split1.txt`). Both pools cover the **same five recording days and the same 17
objects**, so the test split is an episode-level holdout, not a new domain. Nothing in this
repository supports a cross-day, cross-session or cross-embodiment claim.

---

## Two protocols, two different numbers

These are not interchangeable and the manuscript keeps them apart.

**Development protocol (object-level leave-one-out, 17 folds).** One assembly object is held
out at a time, the arms are trained on the remaining objects, and only the held-out object's
frames are scored. Used for exploratory comparisons, e.g. the readout gain of +1.75 pp and the
context change of −0.84 pp.

**Confirmation protocol (single full-development training, one test pass).** Arm A is trained
once on all 111 development episodes, frozen, P and S are fitted on the same rows, and the 37
test episodes are scored once. This is the endpoint: 0.7001 / 0.7477 / +4.76 pp.

A development number and a confirmation number never appear in the same comparison.

---

## Grid

A frame is placed at `start + 0.4 s` inside each annotated segment and then every `0.2 s` up
to `end`. Consequences:

* a segment shorter than 0.4 s contributes **no frame**;
* a segment of 0.6 s contributes two frames;
* frames falling between annotated segments are inserted as **background** with label `-1`;
* the grid runs at 5 Hz and is the only thing the model is trained or scored on.

The grid is rebuilt from the corpus by `scripts/10_build_grid.py` and was verified identical to
the grid the frozen run consumed — see [REPRODUCIBILITY.md](../REPRODUCIBILITY.md).

---

## Features

72 dimensions per frame, all from robot state, over a causal 0.4 s window:

| Block | Channels | Statistics | Dims |
|---|---|---|---|
| proprioception | joint positions (7), joint velocities (7), gripper positions (2), gripper velocities (2), gripper efforts (2) | mean, standard deviation, net displacement | 3 × 20 = 60 |
| force/torque | measured force (3), measured torque (3) | mean, standard deviation | 2 × 6 = 12 |

The recurrent arms consume the last **15** such frames (3 s at 5 Hz), left-padded with zeros at
episode start. The stack is built per episode in **temporal** order: the grid stores annotated
frames before background frames, so using the stored order silently misaligns every window.

---

## Arms

| Arm | Input | Encoder | Readout | Trained |
|---|---|---|---|---|
| A | 15 × 72 | GRU, 48 hidden, 1 layer | linear softmax head | encoder and head jointly |
| P | 15 × 72 → 48-d hidden | the same seed's arm-A GRU, **frozen** | class-balanced logistic regression on the 48-d hidden state | readout only |
| S | 72 (single frame) | none | class-balanced logistic regression | readout only |

Arm A: Adam, lr 1e-3, batch 256, 10 epochs, inverse-frequency cross-entropy. Arm P and S:
`LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=2000,
random_state=0)` on standardised inputs. Both readouts see exactly the same rows as each other
and as the arm-A head.

P is a readout replacement on a frozen encoder, not a second model. Pairing is enforced, not
asserted: see [README](../README.md#the-three-arms-and-why-ap-is-a-readout-comparison).

---

## Post-processing

Both steps are shared by all three arms and estimated on development data only.

1. **Inverse-frequency-weight correction.** From the non-background development
   counts `n_k`, the frozen code computes `w_k = N / (K * n_k)` (with
   absent-class safeguards). Posteriors are divided by `w_k` and renormalised:
   `q_k = (p_k / w_k) / sum_j(p_j / w_j)`, equivalent to reweighting
   by development class counts when all counts are positive. It is not a
   threshold and introduces no tuned parameter. The implementation direction
   is **not** division by class frequency.
2. **Fixed-lag Viterbi.** Each episode is decoded independently with a three-frame lag (0.6 s
   nominal wait), so a frame is committed only after three more frames have been observed.

Transition counts come from the full development episode sequences. The frozen
implementation has a documented negative-indexing defect: background `-1` can
contaminate the final skill **row or column**. The frozen decoder remains the
official-test default. A separately named background-masked matrix estimator is
available for explicitly labelled post hoc DEV sensitivity work only; no masked
result is substituted for the published numbers. See
[SCORING.md](SCORING.md#inherited-defects) and
[DECODER_SENSITIVITY.md](DECODER_SENSITIVITY.md).

---

## Seeds

Three seeds, fixed in advance. The training seed is derived, not chosen:

```
train_seed(i) = 20261110 + 1000 * i + 18        i = 0, 1, 2
              = 20261128, 20262128, 20263128
```

Seeds are averaged **inside** each episode before any inference is computed; the seed is not a
statistical unit and 3 × 37 is never treated as a sample size.

---

## Pre-registration

| Item | Value |
|---|---|
| endpoint | mean paired episode difference in segmental F1, P − A |
| statistical unit | episode, n = 37 |
| interval | 4,000-resample percentile bootstrap over episodes, seed 20261111 |
| effect floor | +1.0 pp |
| equivalence bound | +0.5 pp |
| improved episodes | ≥ 24 of 37 |
| interval width | ≤ 2.0 pp |
| tuning on test | forbidden; the holdout is scored once |

The width condition and its conflict with the ordered test are described in
[SCORING.md](SCORING.md#decision-rule), and both readings are reported in the results.

---

## Secondary diagnostics

Fixed at the same time as the endpoint, and not part of the decision.

* **Grid censoring (label side only).** For every native segment, the number of grid frames it
  occupies. Reported per class and pooled. It uses no prediction, so it cannot be tuned against
  a score.
* **Long `Release` segments.** For `Release` segments of at least 6 and at least 9 frames: the
  share matched at IoU 0.5 by a `Release` prediction, and the share of their frames predicted
  as `Grasp`. Uses the deterministic arm S as the reference decoder so that the diagnostic does
  not depend on a seed.
* A class with fewer than 20 segments in a split is reported but not judged.
