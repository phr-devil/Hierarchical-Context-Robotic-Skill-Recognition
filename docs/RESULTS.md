# Results

Every value below is frozen. The provenance tag on each block says how it was obtained:

* **REPRODUCED** — recomputed by this repository and matching to machine precision;
* **FROZEN_VERIFIED** — taken from the frozen run; the computation was not repeated because the
  holdout was opened once and is now consumed.

Percentages are percentage points (pp) where they describe a difference, and plain percentages
where they describe a share.

---

## Confirmation endpoint — test split, 37 episodes, one pass

`FROZEN_VERIFIED`

| Arm | Episode-level segF1 |
|---|---|
| A (GRU + softmax head, joint training) | 0.7001 |
| P (frozen GRU + logistic probe) | **0.7477** |
| S (single-frame logistic regression) | 0.6619 |

| Quantity | Value |
|---|---|
| mean paired difference P − A | +4.7555 pp |
| 95 % interval | [+3.4741, +5.9937] pp |
| interval width | 2.5196 pp (bound 2.0) |
| episodes improved | 35 / 37 |
| t statistic | 7.234 |
| t test p | 1.6e-8 |
| Wilcoxon signed-rank p | 7.2e-8 |
| one-sided sign test p | 5.1e-9 |
| cells | 473 |
| runs | 2,419 |
| evaluation frames | 48,644 |

**Rule conflict.** The ordered test passes on effect size, interval sign and episode count, so
the coded verdict is `CONFIRMED`. The pre-registered interval-width bound of 2.0 pp is exceeded,
so with the width treated as a confirmation condition the verdict is `INCONCLUSIVE`. Both are
reported; see [SCORING.md](SCORING.md#decision-rule).

---

## Seed variation

`FROZEN_VERIFIED`

| Seed index | training seed | A | P | P − A | episodes improved |
|---|---|---|---|---|---|
| 0 | 20261128 | 0.6686 | 0.7571 | +8.85 pp | 35 / 37 |
| 1 | 20262128 | 0.7284 | 0.7431 | +1.47 pp | 25 / 37 |
| 2 | 20263128 | 0.7033 | 0.7428 | +3.95 pp | 33 / 37 |

Across seeds, arm A's episode mean has a standard deviation of 3.00 pp and arm P's 0.81 pp.
Most of the point estimate therefore rides on the joint model's instability, not on the probe.
The defensible statement is the intersection: **24 of 37 episodes improve under all three
seeds**. The +8.85 pp figure is not the effect and is never quoted as one.

---

## Per-class, seed-averaged

`FROZEN_VERIFIED` — frame-level F1 over evaluation frames. Frame-level, so it is not comparable
with the segmental endpoint above and is never used in a paired test.

| Class | A | P | S | P − A |
|---|---|---|---|---|
| Align | 0.9183 | 0.9254 | 0.8811 | +0.71 |
| Approach | 0.9492 | 0.9582 | 0.9389 | +0.90 |
| Grasp | 0.9422 | 0.9477 | 0.9259 | +0.54 |
| Lift | 0.5592 | 0.6608 | 0.4178 | +10.16 |
| Nudge | 0.0182 | 0.1388 | 0.0000 | +12.06 |
| Pull | 0.9274 | 0.9307 | 0.8733 | +0.33 |
| Push | 0.7673 | 0.7773 | 0.5290 | +1.00 |
| Release | 0.5726 | 0.6541 | 0.5559 | +8.16 |
| Twist | 0.7691 | 0.7924 | 0.5197 | +2.33 |

P ≥ A in all nine classes. The gains concentrate in `Lift`, `Nudge` and `Release` — the classes
with few frames and short segments. `Nudge` remains near zero for every arm; it is present in
too few episodes to be learned and is reported rather than excluded.

Pooled (one count table over the whole split) and frame accuracy, seed-averaged:

| Arm | pooled segF1 | TP | FP | FN | frame accuracy |
|---|---|---|---|---|---|
| A | 0.6866 | 2,037 | 1,482 | 381 | 0.9133 |
| P | 0.7329 | 2,147 | 1,293 | 271 | 0.9224 |
| S | 0.6246 | 1,945 | 1,864 | 474 | 0.8813 |

From A to P: +110 true positives, −189 false positives, −110 misses. The readout buys precision
more than recall.

Cell-level diagnostic (`REPRODUCED` from frozen per-cell values): 473 cells, mean Δ +0.0492,
sd 0.0668, 344 positive. Diagnostic only — cells are not independent observations.

---

## Grid censoring

`FROZEN_VERIFIED` — label side only, no prediction involved.

| Class | segments (dev) | ≤ 2 frames (dev) | segments (test) | ≤ 2 frames (test) |
|---|---|---|---|---|
| Release | 1,377 | 54.9 % | 453 | **64.5 %** |
| Lift | 895 | 38.7 % | 233 | 45.1 % |
| Pull | — | 5.3 % | 299 | 6.4 % |
| Push | — | 0.0 % | 55 | 0.0 % |
| Twist | — | 0.5 % | 63 | 0.0 % |
| Grasp | — | 0.1 % | 580 | 0.0 % |
| Approach | — | 0.1 % | 526 | 0.2 % |
| Align | — | 1.3 % | 233 | 0.9 % |
| Nudge | — | 0.0 % | 10 | 0.0 % |
| **pooled** | 7,743 | 14.9 % | 2,452 | **17.1 %** |

A `Release` segment shorter than the 0.4 s window cannot appear on the grid at all. On the test
split nearly two thirds of `Release` segments are in that position, which caps what any model
can score on the class before it sees a single frame.

---

## Long `Release` segments

`FROZEN_VERIFIED` — arm S as the deterministic reference decoder; `Release` segments matched at
IoU 0.5 against a `Release` prediction.

| Segment length (frames) | dev n | dev match | test n | test match |
|---|---|---|---|---|
| 1 | 324 | 0.8704 | 127 | 0.7008 |
| 2 | 404 | 0.9406 | 148 | 0.8446 |
| 3–5 | 426 | 0.8568 | 106 | 0.6321 |
| **≥ 6** | 189 | 0.2804 | 55 | **0.0727** |
| ≥ 9 | 79 | — | 28 | — |

Share of frames inside long `Release` segments that are predicted as `Grasp`:

| | dev | test |
|---|---|---|
| ≥ 6 frames | 59.4 % (n = 1,882 frames) | **66.3 %** (n = 555) |
| ≥ 9 frames | 66.8 % (n = 1,125) | 69.0 % (n = 364) |

With the same arms under the confirmation protocol, the long-`Release` match rate is 0.339 for A
and 0.406 for P, against 0.073 for S — the recurrent arms do better, and still lose most long
`Release` segments. The `Grasp` share is 0.485 for A and 0.468 for P, so the confusion survives
a better readout.

---

## What these numbers do not show

* No comparison with a published F1@τ. The class inventory (9 skills here versus 69
  action-object pairs in the corpus's own baseline), the metric construction (per-class run
  matching versus thresholded segment matching) and the modality (proprioception and force
  versus wrist video) differ on all three axes.
* No cross-day, cross-session or cross-embodiment claim: the test episodes come from the same
  five recording days and the same 17 objects.
* No claim that the `Release` annotation is wrong. The long-segment behaviour is consistent with
  the annotation absorbing `Release` into `Grasp`; without a double-annotated subset it cannot
  be attributed to the annotator rather than to the model.
* No claim of a new algorithm. Decoupled readout retraining is established; the contribution is
  the pre-registered quantification and the measurement audit.
