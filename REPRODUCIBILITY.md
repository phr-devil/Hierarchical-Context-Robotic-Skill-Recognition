# Reproducibility

What was re-run, what was not, and how to check either claim.

**Status at `v2.0.1`.** That release changes documentation, citation metadata and licence terms
only: the algorithm, the configuration, the tests and the frozen results are unchanged from
`v2.0.0`, so the evidence classes below carry over without re-verification, and nothing here was
re-run to produce it. The text that did change is the short-segment diagnostic in `README.md` and
`docs/RESULTS.md`, which now separates grid coverage from zero-frame censoring and no longer
implies a ceiling on segmental F1 that the measurement does not support. The commands under
"Reproducing the verification" are the ones actually executed; `scripts/20_train_predict.py` and
`scripts/30_score.py` were not executed against the corpus.

---

## Environment

| | |
|---|---|
| Python | 3.13 (3.10+ expected to work) |
| CPU only | no GPU used at any point |
| numpy / scipy / scikit-learn / torch / h5py | versions printed by `scripts/00_check_env.py` |
| wall clock, one seed | ≈ 4 min training and inference, ≈ 2 min grid and features for 111 episodes |

`pytest` runs the whole unit suite in well under a minute without the corpus.

---

## What was verified here

### 1. Grid, labels, background mask and features — REPRODUCED

`scripts/90_verify_dev_grid.py` rebuilds the development split from the corpus and compares it
to the grid the frozen scoring run consumed, element by element, with zero tolerance.

```
grid frames=185451 annotated=169728 background=15723 (8.48%) episodes=111
frame features (185451, 72) dtype=float32
{
 "split": "train",
 "reference_frames": 185451,
 "rebuilt_frames": 185451,
 "frames_equal": true,
 "episode_ids_equal": true,
 "times_max_abs_diff": 0.0,
 "labels_equal": true,
 "background_equal": true,
 "features_max_abs_diff": 0.0,
 "features_shape_equal": true,
 "identical": true
}
```

This covers the parts that are most likely to drift silently: the `start + 0.4 s` frame rule,
the `0.2 s` stride, the skip rule for short segments, the insertion of background frames, the
temporal ordering used for windows, and the 60 + 12 feature construction.

### 2. Aggregation and paired inference — REPRODUCED

Given the frozen per-episode values, the aggregation and decision code in `skillseg.metrics`
reproduces the frozen report exactly:

```
delta    0.0475553982   (difference 0.0)
ci95     [0.0347411538, 0.0599369398]   (difference 0.0 on both bounds)
width    0.0251957861   (difference 0.0)
episodes improved      35
t statistic            7.234272
verdict                CONFIRMED
verdict_strict         INCONCLUSIVE
clause_conflict        true
```

Seed averaging also matches: recomputing the seed-averaged episode arrays from the per-seed
values reproduces the frozen paired arrays with a maximum difference of 0.0.

### 3. Decision-rule behaviour — REPRODUCED

`tests/test_protocol.py` covers the four branches with synthetic inputs, including the width
conflict: a case that clears effect size, sign and episode count while exceeding the width bound
must return `CONFIRMED`, `underpowered`, `clause_conflict` and `verdict_strict == INCONCLUSIVE`.
`tests/test_arms.py` corrupts the encoder after the probe is fitted and requires
`assert_paired` to raise.

---

## What was not re-run

### Training, holdout inference, per-class and pooled scores — FROZEN_VERIFIED

The 37 test episodes were scored once, before this repository existed in its current form. The
holdout is consumed: re-running training or inference to obtain a slightly different number
would be a second look at the same data, and the pre-registration forbids it. The values are
therefore reported as they were produced, with their hashes, and the code that produced them is
this code.

For the same reason `scripts/20_train_predict.py` and `scripts/30_score.py` have **not** been
executed against the corpus in this checkout: doing so would mean building the test grid and
running inference over the consumed holdout. What was exercised here is `scripts/10_build_grid.py`
(development split, executed) and `scripts/90_verify_dev_grid.py` (executed, identical), plus the
components of steps 2 and 3 through the unit suite — pairing, decoding, matching, aggregation and
the decision rule — and the decision step end to end on the frozen per-episode values.

A reader who runs the pipeline end to end will get numbers close to, but not necessarily
identical to, those above. Sources of legitimate divergence, in order of size:

1. **torch version and thread count** — the batch loop draws from the global RNG; a different
   torch build changes the sampled order and therefore the trained encoder;
2. **BLAS and `lbfgs` convergence** in the logistic readouts;
3. **species of the CPU** for floating-point reductions.

The protocol is deterministic conditional on the environment, not across environments. A
bit-identical rerun requires the same torch build; no claim is made that it is portable.

### Secondary diagnostics — FROZEN_VERIFIED

Grid censoring and the long-`Release` confusion were computed from the same frozen scoring
pass. `scripts/50_censoring.py` recomputes the label-side half from the corpus and will agree
with the censoring table, because that half uses no prediction.

---

## Reproducing the verification

```bash
# unit tests, no dataset required
pytest

# grid equivalence (needs the corpus and a reference grid file)
python scripts/90_verify_dev_grid.py \
    --reference /path/to/reference_grid.npz \
    --dataset-root /path/to/REASSEMBLE
```

The reference file must contain `grid_ep`, `grid_t`, `grid_label`, `is_bg` and `ALL`, which is
the layout the frozen scoring program used. Exit code 0 means the rebuild is identical.

---

## Evidence classes

| Layer | Class |
|---|---|
| grid, labels, background mask, features | REPRODUCED |
| cell/episode aggregation, bootstrap, decision rule | REPRODUCED |
| training, test inference, per-class and pooled scores | FROZEN_VERIFIED |
| grid censoring, long-`Release` confusion | FROZEN_VERIFIED |
| `legacy/context_study/` results | REPORTED_ONLY — archived, not reproduced by this code, not part of the current claim |

No result in this repository is labelled REPRODUCED unless it was recomputed here and matched.
