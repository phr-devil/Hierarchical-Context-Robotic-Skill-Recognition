"""Models: causal GRU encoders, single-frame classifier, confidence gate.

Two model families are compared in the paper.

Single-frame (SF)
    A multinomial logistic regression on the 72-dimensional frame feature with
    no history and no action label. This is the context-free baseline.

GRU-based
    A one-layer causal GRU over a three-second window of the same features,
    with 48 hidden units and a linear head. What the head predicts distinguishes
    the variants:

    GRUskill      skill head only            (temporal modelling, no action label)
    DualHead      skill head + action head   (action label as auxiliary target)
    GRUctx        action head only, whose posterior is appended to the frame
                  feature to form the 76-dimensional skill classifier input
    GateCtx       GRUctx and SF fused by a two-parameter confidence gate

GRUctx and DualHead are not a strict single-variable ablation of each other:
they differ in where the action information enters (input fusion versus an
auxiliary loss) and in the decision rule (two stages versus one). Both are
reported so that the comparison is not read as a controlled one.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class CausalGRU(nn.Module):
    """One-layer GRU over a fixed-length window; predicts from the last step."""

    def __init__(self, n_features, hidden, n_out, layers=1):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1])


class DualHeadGRU(nn.Module):
    """Shared GRU trunk with a skill head and an auxiliary action head."""

    def __init__(self, n_features, hidden, n_skill, n_action, layers=1):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=layers, batch_first=True)
        self.hs = nn.Linear(hidden, n_skill)
        self.ha = nn.Linear(hidden, n_action)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.hs(out[:, -1]), self.ha(out[:, -1])


def balanced_weights(y, n_classes):
    """Inverse-frequency weights as a float tensor (torch counterpart of the numpy helper)."""
    y = np.asarray(y).astype(int)
    cnt = np.bincount(y, minlength=n_classes).astype(float)
    w = np.maximum(cnt, 1)
    w = w.sum() / (n_classes * w)
    return torch.tensor(w, dtype=torch.float32)


def train_gru(model, X, y, y_aux=None, weights=None, aux_weights=None,
              epochs=10, batch=256, subsample=60000, lr=1e-3, aux_weight=1.0,
              n_classes=None, n_aux_classes=None):
    """Mini-batch training with the fixed budget used in all reported runs.

    At most ``subsample`` windows are used per fold; the cap applies to every
    variant, so the comparison is budget-matched. ``y_aux`` switches on the
    auxiliary head with unit loss weight. When ``weights`` is not supplied, the
    balanced weights are estimated from the retained windows.

    ``model`` may be a module or a zero-argument factory. A factory is called
    *after* the subsample permutation has been drawn, which reproduces the
    random-number order of the reported main run (draw first, initialise second).
    The five-model comparison initialises first, so it passes a built module.
    """
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.long)
    if y_aux is not None:
        y_aux = torch.as_tensor(y_aux, dtype=torch.long)
    if len(X) > subsample:
        perm = torch.randperm(len(X))[:subsample]
        X = X[perm]
        y = y[perm]
        if y_aux is not None:
            y_aux = y_aux[perm]
    if not isinstance(model, nn.Module):
        model = model()
    if weights is None:
        n_classes = int(y.max()) + 1 if n_classes is None else int(n_classes)
        weights = balanced_weights(y.numpy(), n_classes)
    if y_aux is not None and aux_weights is None:
        n_aux_classes = int(y_aux.max()) + 1 if n_aux_classes is None else int(n_aux_classes)
        aux_weights = balanced_weights(y_aux.numpy(), n_aux_classes)

    loss_skill = nn.CrossEntropyLoss(weight=weights)
    loss_action = nn.CrossEntropyLoss(weight=aux_weights) if y_aux is not None else None
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(X))
        for s in range(0, len(X), batch):
            idx = order[s:s + batch]
            opt.zero_grad()
            if y_aux is None:
                loss = loss_skill(model(X[idx]), y[idx])
            else:
                hs, ha = model(X[idx])
                loss = loss_skill(hs, y[idx]) + aux_weight * loss_action(ha, y_aux[idx])
            loss.backward()
            opt.step()
    model.eval()
    return model


def predict_proba(model, X, batch=2000):
    """Class posteriors for every window, in fixed-size chunks."""
    model.eval()
    X = torch.as_tensor(X, dtype=torch.float32)
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            out.append(torch.softmax(model(X[s:s + batch]), 1).numpy())
    return np.vstack(out).astype(np.float32)


def fit_single_frame(X_train, y_train, X_test, n_classes, C=1.0, seed=0, max_iter=2000):
    """Fit the SF classifier and return posteriors aligned to the full class order.

    Folds may miss a class; the returned matrix always has ``n_classes``
    columns so that emissions stay comparable across folds.
    """
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(C=C, class_weight="balanced", solver="lbfgs",
                             max_iter=max_iter, random_state=seed)
    clf.fit(scaler.transform(X_train), y_train)
    pr = clf.predict_proba(scaler.transform(X_test))
    full = np.zeros((len(X_test), n_classes))
    for ci, c in enumerate(clf.classes_):
        full[:, int(c)] = pr[:, ci]
    return full, clf, scaler


def predict_softmax(clf, scaler, X, n_classes):
    """Posteriors from an already fitted classifier, aligned to the full class order."""
    pr = clf.predict_proba(scaler.transform(X))
    full = np.zeros((len(X), n_classes))
    for ci, c in enumerate(clf.classes_):
        full[:, int(c)] = pr[:, ci]
    return full


def fit_lodo_softmax(X, y, dates, n_classes, C=1.0, seed=0, max_iter=2000, train_mask=None):
    """Leave-one-date-out SF posteriors used as the visible-object reference.

    ``dates`` is the per-frame recording date; the function folds over its
    distinct values. ``train_mask`` restricts which frames may be used for
    fitting -- needed on the continuous timeline, where background frames carry
    the label ``-1`` and are decoded but never trained on. Returns the posterior
    matrix and the per-date class weights, which are later used to undo the
    balanced training prior before decoding.
    """
    dates = np.asarray([str(d) for d in dates])
    trainable = np.ones(len(y), dtype=bool) if train_mask is None else np.asarray(train_mask, dtype=bool)
    probs = np.zeros((len(y), n_classes))
    cweights = {}
    for d in sorted(set(dates.tolist())):
        te = np.where(dates == d)[0]
        tr = np.where((dates != d) & trainable)[0]
        if len(tr) == 0:
            raise ValueError(
                f"leave-one-date-out fold for {d} has no training frames; "
                "at least two recording dates are required")
        ytr = y[tr]
        cnt = np.bincount(ytr, minlength=n_classes).astype(float)
        cweights[d] = len(ytr) / (n_classes * np.maximum(cnt, 1))
        full, _, _ = fit_single_frame(X[tr], ytr, X[te], n_classes, C=C,
                                      seed=seed, max_iter=max_iter)
        probs[te] = full
    return probs, cweights


def gate_train(sf_e, ctx_e, pact_e, y, epochs=300, lr=0.05):
    """Fit the two-parameter confidence gate on training emissions.

    The gate weight is ``sigmoid(w1 * action_confidence + w2 * skill_uncertainty
    + b)`` and interpolates the two posteriors in log space. It is fitted on
    training-fold emissions only, so no test-object information reaches it.
    """
    eps = 1e-12
    act_conf = (pact_e * np.log(pact_e + eps)).sum(1)
    skill_ent = -(sf_e * np.log(sf_e + eps)).sum(1)
    f1 = torch.tensor(act_conf, dtype=torch.float32)
    f2 = torch.tensor(skill_ent, dtype=torch.float32)
    y_t = torch.tensor(y).long()
    logb = torch.tensor(np.log(sf_e + eps))
    logc = torch.tensor(np.log(ctx_e + eps))
    w1 = torch.zeros(1, requires_grad=True)
    w2 = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w1, w2, b], lr=lr)
    for _ in range(epochs):
        g = torch.sigmoid(w1 * f1 + w2 * f2 + b).unsqueeze(1)
        logits = (1 - g) * logb + g * logc
        loss = nn.CrossEntropyLoss()(torch.softmax(logits, 1), y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w1.item(), w2.item(), b.item()


def gate_apply(sf_e, ctx_e, pact_e, w1, w2, b):
    eps = 1e-12
    act_conf = (pact_e * np.log(pact_e + eps)).sum(1)
    skill_ent = -(sf_e * np.log(sf_e + eps)).sum(1)
    g = 1.0 / (1.0 + np.exp(-(w1 * act_conf + w2 * skill_ent + b)))
    logb = np.log(sf_e + eps)
    logc = np.log(ctx_e + eps)
    logits = (1 - g)[:, None] * logb + g[:, None] * logc
    logits = logits - logits.max(1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(1, keepdims=True)
