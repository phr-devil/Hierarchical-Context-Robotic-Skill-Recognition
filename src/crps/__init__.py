"""CRPS: predicted hierarchical context for low-level robotic skill recognition.

Reference implementation for the paper that evaluates whether an online posterior
over the parent assembly action improves low-level skill recognition on the
REASSEMBLE dataset under leave-one-object-out cross-validation.

Modules
-------
common    causal decoders (HMM filtering, fixed-lag Viterbi) and metrics
features  72-dimensional proprioceptive / force-torque frame features
data      REASSEMBLE HDF5 parsing, split handling, annotated frame enumeration
grid      continuous timeline construction (annotated frames + background frames)
objects   per-frame assembly-object labels and object episode sets
protocol  leave-one-object-out folds, causal history tensors, transition counts
models    causal GRU encoders, single-frame classifier, confidence gate
"""

__version__ = "1.0.0"
