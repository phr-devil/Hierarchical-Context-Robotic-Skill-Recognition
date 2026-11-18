"""Readout-sensitivity study for proprioceptive robot skill segmentation.

Reference implementation for the manuscript *Readout Sensitivity and Temporal
Evaluation in Sensor-Based Robot Skill Segmentation for Contact-Rich Assembly*.

The study asks a measurement question rather than proposing an architecture: on
the REASSEMBLE corpus, how much of the reported segmental F1 is attributable to
the readout on top of a frozen encoder, and how much of the residual error is a
property of the annotation layer rather than of the model.

Three arms share one encoder, one input and one evaluation grid:

``A``  causal GRU (72 -> 48) trained jointly with a softmax skill head.
``P``  the same seed's trained GRU, frozen, with a class-balanced logistic
       regression fitted on its 48-dimensional hidden state.
``S``  a single-frame class-balanced logistic regression on the 72-dimensional
       frame features, with no recurrent encoder at all.

Modules
-------
config    configuration loading and seed derivation
data      REASSEMBLE HDF5 parsing, official splits, 5 Hz evaluation grid
features  60-dimensional proprioceptive + 12-dimensional force/torque features
objects   assembly-object labels, object runs and (episode, object) cells
models    the three arms, encoder fingerprints, pairing assertions
decode    class-prior correction and fixed-lag Viterbi
metrics   run-level one-to-one matching, cell/episode scores, bootstrap, rules
pipeline  end-to-end orchestration for one seed and for the decision step
"""

__version__ = "2.0.1"

__all__ = ["config", "data", "features", "objects", "models", "decode", "metrics", "pipeline"]
