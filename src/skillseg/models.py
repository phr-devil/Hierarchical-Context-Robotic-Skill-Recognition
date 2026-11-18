"""The three arms and the pairing guarantees between them.

``A`` and ``P`` are not two independent models.  Arm P is arm A's trained encoder
with the readout replaced: the encoder is frozen, both readouts are fitted on
exactly the same rows, and the only difference is the decision surface.  That is
what makes the A/P difference interpretable as a readout effect rather than as a
difference between two training runs.

The guarantees are structural rather than claimed:

* :func:`freeze_encoder` sets ``requires_grad_(False)`` and evaluation mode;
* :func:`fit_probe` records a fingerprint of the encoder it consumed, and
  :func:`assert_paired` recomputes it afterwards -- a probe fitted on a changed
  encoder cannot pass;
* :func:`fit_single_frame` receives no encoder object at all, so arm S cannot use
  one.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import Config


# --------------------------------------------------------------- torch model
def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_gru(input_dim: int, hidden: int, n_classes: int):
    """Causal GRU encoder + linear softmax head, as used by arm A."""
    _, nn = _torch()

    class CausalSkillGRU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden, num_layers=1, batch_first=True)
            self.head = nn.Linear(hidden, n_classes)

        def forward(self, x):
            out, _ = self.gru(x)
            return self.head(out[:, -1])

        def encode(self, x):
            out, _ = self.gru(x)
            return out[:, -1]

    return CausalSkillGRU()


def set_seed(seed: int) -> None:
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)


def inverse_frequency_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    return len(labels) / (n_classes * np.maximum(counts, 1.0))


# --------------------------------------------------------------- arm A
def train_joint_arm(stacked: np.ndarray, labels: np.ndarray, rows: np.ndarray,
                    cfg: Config, seed: int) -> Tuple[Any, int]:
    """Train encoder and softmax head jointly.  Returns ``(model, optimisation_steps)``.

    The call order (seed, model construction, optimiser, batch loop) is the order
    the frozen run used; changing it changes the sampled batches and therefore the
    numbers.
    """
    torch, nn = _torch()
    set_seed(seed)
    model = build_gru(stacked.shape[2], int(cfg.gru["hidden"]), cfg.n_skills)

    x = torch.tensor(np.asarray(stacked[rows])).float()
    y = torch.tensor(labels[rows]).long()
    weight = torch.tensor(inverse_frequency_weights(labels[rows], cfg.n_skills),
                          dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.gru["lr"]))

    epochs, batch = int(cfg.gru["epochs"]), int(cfg.gru["batch"])
    model.train()
    steps = 0
    for _ in range(epochs):
        order = torch.randperm(len(x))
        for start in range(0, len(x), batch):
            idx = order[start:start + batch]
            optimizer.zero_grad()
            loss = loss_fn(model(x[idx]), y[idx])
            loss.backward()
            optimizer.step()
            steps += 1
    model.eval()
    return model, steps


def freeze_encoder(model) -> None:
    """Make the encoder a fixed feature extractor.  Precondition for arm P."""
    model.eval()
    for parameter in model.gru.parameters():
        parameter.requires_grad_(False)


def encoder_is_frozen(model) -> bool:
    return (not model.training) and all(not p.requires_grad for p in model.gru.parameters())


# --------------------------------------------------------------- inference
def predict_proba(model, stacked: np.ndarray, rows: np.ndarray, chunk: int = 4000) -> np.ndarray:
    torch, _ = _torch()
    out: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(rows), chunk):
            batch = torch.tensor(np.asarray(stacked[rows[start:start + chunk]])).float()
            out.append(torch.softmax(model(batch), 1).numpy())
    return np.vstack(out).astype(np.float32)


def encode_hidden(model, stacked: np.ndarray, rows: np.ndarray, chunk: int = 12000) -> np.ndarray:
    """48-dimensional hidden state of the last recurrent step."""
    torch, _ = _torch()
    out: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(rows), chunk):
            batch = torch.tensor(np.asarray(stacked[rows[start:start + chunk]])).float()
            out.append(model.encode(batch).numpy())
    return np.vstack(out).astype(np.float32)


# --------------------------------------------------------------- linear readouts
def fit_logistic(x: np.ndarray, y: np.ndarray, cfg: Config) -> Tuple[Any, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(**cfg.readout).fit(scaler.transform(x), y)
    return scaler, clf


def apply_logistic(fit: Tuple[Any, Any], x: np.ndarray, n_classes: int) -> np.ndarray:
    scaler, clf = fit
    proba = clf.predict_proba(scaler.transform(x))
    full = np.zeros((len(x), n_classes), dtype=np.float64)
    for column, cls in enumerate(clf.classes_):
        full[:, int(cls)] = proba[:, column]
    return full


def fit_probe(hidden: np.ndarray, labels: np.ndarray, cfg: Config) -> Tuple[Any, Any]:
    """Class-balanced logistic regression on the frozen encoder's hidden state."""
    return fit_logistic(hidden, labels, cfg)


def fit_single_frame(features: np.ndarray, labels: np.ndarray, cfg: Config) -> Tuple[Any, Any]:
    """Class-balanced logistic regression on instantaneous features; no encoder."""
    return fit_logistic(features, labels, cfg)


# --------------------------------------------------------------- fingerprints
def _flatten(obj: Any) -> List[Tuple[str, np.ndarray]]:
    if hasattr(obj, "state_dict"):
        return [(k, v.detach().cpu().numpy()) for k, v in obj.state_dict().items()]
    out: List[Tuple[str, np.ndarray]] = []
    for name in ("coef_", "intercept_", "classes_", "mean_", "scale_", "var_"):
        if hasattr(obj, name):
            out.append((name, np.asarray(getattr(obj, name))))
    return out


def params_fingerprint(obj: Any) -> str:
    """Structural hash of a fitted object; used only for pairing checks."""
    digest = hashlib.sha256()
    items: List[Tuple[str, np.ndarray]] = []
    if isinstance(obj, (tuple, list)):
        for part in obj:
            items.extend(_flatten(part))
    else:
        items.extend(_flatten(obj))
    for name, array in items:
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()[:16]


def rows_fingerprint(rows: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest()[:16]


@dataclass
class ProbeFit:
    """A linear readout plus the provenance needed to prove what it was fitted on."""

    scaler: Any
    clf: Any
    encoder_sha: str
    rows_sha: str
    n_rows: int
    n_features: int
    encoder_frozen: bool

    @property
    def fit(self) -> Tuple[Any, Any]:
        return self.scaler, self.clf


@dataclass
class TrainedArms:
    """One seed's arm triple.  ``encoder`` is shared by A and P by construction."""

    seed: int
    train_seed: int
    encoder: Any
    probe: ProbeFit
    single_frame: ProbeFit
    steps: int
    encoder_sha_before_probe: str
    encoder_sha_after_probe: str
    meta: Dict[str, Any] = field(default_factory=dict)


def build_arms(stacked: np.ndarray, frame_features: np.ndarray,
               labels: np.ndarray, rows: np.ndarray, cfg: Config, seed_index: int) -> TrainedArms:
    """Train arm A, freeze it, fit P on its hidden state, and fit S independently."""
    train_seed = cfg.train_seed(seed_index)
    model, steps = train_joint_arm(stacked, labels, rows, cfg, train_seed)

    sha_trained = params_fingerprint(model.gru)
    freeze_encoder(model)
    sha_frozen = params_fingerprint(model.gru)
    if sha_trained != sha_frozen:
        raise RuntimeError("encoder changed during freezing; refusing to fit a probe")

    hidden = encode_hidden(model, stacked, rows)
    if hidden.shape[1] != int(cfg.gru["hidden"]):
        raise RuntimeError(f"hidden state has {hidden.shape[1]} dims, expected {cfg.gru['hidden']}")
    probe_scaler, probe_clf = fit_probe(hidden, labels[rows], cfg)

    sha_after = params_fingerprint(model.gru)
    if sha_after != sha_frozen:
        raise RuntimeError("encoder changed while fitting the probe; the readout is not paired")

    single_scaler, single_clf = fit_single_frame(frame_features[rows], labels[rows], cfg)

    rows_sha = rows_fingerprint(rows)
    return TrainedArms(
        seed=seed_index,
        train_seed=train_seed,
        encoder=model,
        probe=ProbeFit(probe_scaler, probe_clf, sha_after, rows_sha, len(rows),
                       hidden.shape[1], encoder_is_frozen(model)),
        single_frame=ProbeFit(single_scaler, single_clf, "no-encoder", rows_sha, len(rows),
                              frame_features.shape[1], True),
        steps=steps,
        encoder_sha_before_probe=sha_trained,
        encoder_sha_after_probe=sha_after,
    )


def assert_paired(arms: TrainedArms) -> None:
    """Raise unless A and P provably share one frozen encoder and one row set."""
    current = params_fingerprint(arms.encoder.gru)
    if not encoder_is_frozen(arms.encoder):
        raise AssertionError("encoder is not frozen")
    if arms.encoder_sha_before_probe != current:
        raise AssertionError("encoder differs from the one the probe was fitted on")
    if arms.probe.encoder_sha != current:
        raise AssertionError("probe provenance does not match the encoder")
    if arms.probe.rows_sha != arms.single_frame.rows_sha:
        raise AssertionError("A/P and S were fitted on different rows")
    if arms.probe.n_rows != arms.single_frame.n_rows:
        raise AssertionError("A/P and S were fitted on a different number of rows")
    if arms.probe.n_features != arms.encoder.head.in_features:
        raise AssertionError("probe does not read the recurrent hidden state")


def arm_probabilities(arms: TrainedArms, stacked: np.ndarray, frame_features: np.ndarray,
                      n_classes: int, rows: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Class posteriors for arms A, P and S over the requested rows."""
    rowwise = np.arange(stacked.shape[0]) if rows is None else np.asarray(rows)
    hidden = encode_hidden(arms.encoder, stacked, rowwise)
    return {
        "A": predict_proba(arms.encoder, stacked, rowwise),
        "P": apply_logistic(arms.probe.fit, hidden, n_classes),
        "S": apply_logistic(arms.single_frame.fit, np.asarray(frame_features[rowwise]), n_classes),
    }
