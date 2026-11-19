# Post hoc background-masked decoder sensitivity (DEV only)

## Status

**METHOD AVAILABLE / EFFECT NOT RUN.** This is a post hoc robustness proposal, not part
of the original preregistered confirmation. There is no measured A/P effect under the
corrected transition matrix in this repository. No official-test result has been
recomputed, and the original 37-episode holdout remains consumed.

## Exact defect and two estimators

`skillseg.decode.transition_matrix()` is the original frozen estimator. It
accumulates transitions from episode timelines containing background `-1` labels.
Because NumPy accepts negative indices, a background *source* is counted in the
last skill row and a background *destination* in the last skill column. It is
the estimator used by `pipeline.train_and_predict()` and by all reported results.

`skillseg.decode.transition_matrix_masked()` uses the same episode boundaries,
Laplace smoothing and row normalisation, but **skips** a transition unless both
its source and destination are in `[0, n_skills)`. It does not bridge over
background frames. It is never selected automatically by the original pipeline.

Do **not** alter `transition_matrix()`, `configs/default.json`, any frozen
prediction, the existing `results/*.json`, or the `v2.0.1` release.

## Valid DEV-only sensitivity protocol (only when authentic artifacts exist)

1. Locate provenance-verified **development out-of-fold** A/P emissions for the
   original exploratory object-held-out comparisons, per fold and seed.
   Check episode/frame/order alignment, original A/P encoder pairing, and
   unchanged prior-correction configuration. **In-sample** emissions are not
   an acceptable substitute for held-out DEV evaluation.
2. For each corresponding DEV fold, estimate both transition matrices using
   *only that fold's permitted DEV training sequences*. Keep all other
   parameters and the saved emissions identical. Never estimate transitions
   from the fold's scoring labels.
3. Apply the identical frozen prior correction, 3-frame fixed-lag Viterbi and
   original object-cell/episode scoring to both matrices. Change **only**
   transition estimation; keep the same matched evaluation population.
4. Report each arm's original/corrected DEV segmental F1, their paired P−A
   differences, the change in P−A, the number of predictions changed and the
   impacted classes/episodes. Preserve folds, seeds, and episodes as correlated
   observations; do not treat fold×seed cells as independent confirmation.
5. Mark all results `POST_HOC_DEV`. Report adverse or null findings too.
   Do not translate them into a corrected official-test estimate, use them
   for test-driven model selection, or revise the frozen confirmation verdict.

**Stop condition:** if the authentic DEV out-of-fold class probabilities or
their alignment/provenance are missing, record `NOT_RUN / ARTIFACT_MISSING`.
Do not train new models, infer over official test episodes or invent results
just to fill the report.

## Required low-cost validation

Run `pytest tests/test_decode.py` (synthetic only) to pin both estimators:
a background-containing sequence must reproduce the frozen contamination,
the masked alternative must discard both row- and column-side contamination,
and both estimators must agree when all labels are valid. The tests neither
require REASSEMBLE nor establish empirical robustness of the A/P effect.

## Interpretation

A shared erroneous transition matrix makes the A/P decoder *configuration*
the same, but it does not make the error cancel: different emission posteriors
can respond differently to the transition perturbation. The published effect
is conditional on the frozen decoder. Its corrected-decoder robustness is
**unknown** until an appropriately bounded analysis is completed.
