# Scoring

This document defines every quantity that appears in the results, in the order the
manuscript introduces them. The three definitions numbered (7)–(9) in the manuscript are
reproduced here as (7), (8) and (9). If the final manuscript renumbers its equations, these
definitions remain authoritative for the code.

---

## Units

| Unit | What it is | Role |
|---|---|---|
| run | maximal stretch of consecutive grid frames inside one episode that belong to one object and are not background | segments matching |
| cell | one `(episode, object)` pair, holding that object's runs | builds the episode score |
| episode | one recording, 37 in the test split | **statistical unit** |
| seed | one training run of arm A, 3 in total | averaged inside an episode, never an observation |

A run is cut wherever the object changes or a background frame intervenes, so two separate
visits to the same object in one episode never share a match. Background frames carry label
`-1` and are excluded from every scored quantity. The `"action"` object is a frame-level
artefact of the annotation format, not an assembly object, and is excluded from cells.

---

## (7) Cell score

Let `g` and `p` be the ground-truth and predicted label sequences over the frames of a cell,
in temporal order. Run-length encode both:

```
runs(g) = (label, first_frame, last_frame) ...
runs(p) = ...
```

Match greedily: for each ground-truth run in order, find the unused predicted run **with the
same label** whose frame-index intersection-over-union is largest, and accept it if that IoU
is at least `tau = 0.5`.

```
IoU(a, b) = |[a0, a1] ∩ [b0, b1]| / |[a0, a1] ∪ [b0, b1]|        (inclusive frame indices)
```

Then

```
S_cell = 2 TP / (2 TP + FP + FN)                                    (7)
```

with `TP` the accepted matches, `FP` predicted runs left unused, `FN` ground-truth runs left
unmatched. Three properties are worth stating because they are easy to misread:

* the matching is **within a class**: a `Lift` prediction can never match a `Grasp` segment,
  it counts as one false alarm and one miss;
* the matching is **greedy, not optimal**: an assignment solver could in principle accept a
  different set of pairs. The frozen program is greedy and this repository is greedy;
* the IoU is over **grid frame indices**, i.e. a discretisation of temporal IoU at 5 Hz, not
  an IoU in seconds. A segment that occupies two frames cannot exceed IoU 2/2 only if the
  prediction occupies exactly those two frames.

---

## (8) Episode score

A cell exists only when the episode contains frames of that object. Therefore

```
S_episode = (1 / |O_e|) * sum_{o in O_e} S_cell(e, o)               (8)
```

where `O_e` is the set of objects **present** in episode `e`.

An empty cell cannot arise: cells are constructed from runs and a run has at least one frame.
An object the episode never touches is *absent from the denominator*, not scored as zero. If
it were scored as zero — the natural reading of "mean over the 17 objects" — every episode
would be diluted by the objects it does not use, and the reported values would be lower. That
variant was not used and is not reported anywhere.

Two consequences follow, and both are stated in the manuscript rather than left implicit:

* cells are weighted **equally**, so a one-frame visit counts as much as a ten-second one;
* the arm score is the mean of `S_episode` over episodes, so long episodes do not dominate.

The `pooled` score in the results is a different quantity — one count table over every run in
the split — and is reported only so that the difference between the two is visible.

---

## (9) Paired effect and interval

Per seed `s`, compute `S_episode` for each arm. Average the seeds **inside** the episode:

```
P_e = mean_s P_(e,s)        A_e = mean_s A_(e,s)

d_e = P_e - A_e             d = mean_e d_e                          (9)
```

The 95 % interval is the percentile interval of 4,000 bootstrap resamples of the 37 episodes
(`numpy.random.RandomState(20261111)`, resampling episodes with replacement, recomputing the
mean of `d_e`). The episode is the resampling unit.

Not statistical units, and never treated as such:

* the 473 cells — reported as a diagnostic with their own mean, sd and positive count;
* the 3 × 37 seed-episode pairs — averaging seeds first is what removes initialisation
  variance; treating 111 pairs as independent would inflate the sample size threefold;
* the 2,419 runs.

---

## Decision rule

Fixed before the holdout was opened.

| Condition | Threshold |
|---|---|
| effect floor | `d >= +1.0` pp |
| interval sign | lower bound `> 0` |
| improved episodes | `>= 24` of 37 |
| interval width | `<= 2.0` pp |
| equivalence bound (null branch) | upper bound `< +0.5` pp |

Ordered test, as implemented:

```
if d >= 1.0 and lower > 0 and improved >= 24:   CONFIRMED
elif upper < 0.5:                               NOT_CONFIRMED
else:                                           INCONCLUSIVE
```

The width condition is in the pre-registration but not in that ordered test — in the protocol
document it appears in the prose of the "otherwise" branch, as a reason a result may be
inconclusive. In the reported run the effect clears all three ordered conditions while the
width is 2.52 pp against the 2.0 pp bound.

Both readings are therefore computed and reported:

* `verdict` — the ordered test as coded: `CONFIRMED`;
* `verdict_strict` — with the width treated as a confirmation condition: `INCONCLUSIVE`.

`clause_conflict` is true exactly when these differ. `scripts/40_decide.py` prints all three
fields; `results/decision_record.json` stores them. Neither reading is presented alone.

Power, from a development-split resampling calibration performed before the holdout:

* a true effect of 1.92 pp was recovered as `CONFIRMED` in 100 % of resamples;
* a true effect of zero gave `INCONCLUSIVE` in 71 %, `NOT_CONFIRMED` in 29 %, and a false
  `CONFIRMED` in 0.2 %;
* the minimum detectable effect at 80 % power with 37 episodes is 0.99 pp.

So a null result at 37 episodes would have been informative in the equivalence direction only,
and an effect below roughly 1 pp could not have been confirmed at all.

---

## Inherited defects

These are properties of the frozen implementation. They are preserved so that the published
numbers remain reproducible, and they are pinned by tests so that they cannot change silently.

1. **Background frames in the transition matrix.** The matrix is estimated from the full
   label sequence, background included. Background frames carry `-1`, and negative indexing
   maps them onto the **last** skill row (`Twist`, in the alphabetical class order). That row
   therefore also accumulates the outgoing transitions of background frames — mostly
   transitions into segment starts. The decoder's transitions out of `Twist` are consequently
   contaminated. `tests/test_decode.py::test_background_label_lands_on_the_last_row` pins the
   behaviour. Fixing it would change the decoded sequences and invalidate comparison with the
   published numbers; it is left as a documented defect.
2. **Greedy matching.** A globally optimal assignment would accept a different set of pairs in
   ambiguous cases. No alternative was ever scored.
3. **Object-level leave-one-out is not leave-one-domain-out.** In the development protocol one
   object is held out at a time, but objects recur across recording days, so day-level
   generalisation is not identified. The test split is a recording-level holdout of the same
   five days and the same 17 objects; it is an episode-level holdout, not a new domain.
4. **Per-class F1 is frame-level.** The per-class table counts frames, so a class can look
   acceptable there while every one of its segment boundaries is wrong. It is never mixed with
   the segmental score.
