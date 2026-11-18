# Archived: the hierarchical-context study

This directory is the previous version of this repository, kept for provenance.

**What it studied.** Whether an online posterior over the parent assembly action, predicted from
a three-second causal history of robot-state features, improves low-level skill recognition on
REASSEMBLE under leave-one-object-out cross-validation. Its headline claim was a segmental-F1
gain from adding that context.

**Why it is archived.** The current manuscript asks a different question — how much of the score
is readout and annotation rather than perception — and the context result does not support it. In
the development protocol used for the current study, adding hierarchical context changes the
segmental score by −0.84 pp, while replacing the readout on the same folds changes it by
+1.75 pp. The context mechanism is therefore an exploratory negative result in the present work,
not a positive claim.

**Status.** Not maintained, not covered by the current test suite, not imported by
`src/skillseg`, and not part of any published number in `results/`. Its own numbers are
`REPORTED_ONLY`: they are transcribed from the runs that produced them and are not reproduced by
any code in this repository.

**Layout.** Unchanged from the original release:

```
src/crps/        the context-study package
scripts/         00-02 build frames and grid, 10-11 run arms, 20-22 aggregate
tests/           its test suite (kept as-is)
configs/         its configuration
results/         its reference_metrics.json
README.md        its original README, unedited
```

Running it requires `PYTHONPATH=legacy/context_study/src` and the same corpus; the paths inside
are those of the machine the study ran on. Expect to adjust them. Do not treat a successful run
as a reproduction of the current paper.
