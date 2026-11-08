import numpy as np
import pytest
import torch

from crps import models


def _data(n=64, window=4, dim=6, classes=3, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, window, dim).astype(np.float32)
    y = rng.randint(0, classes, n)
    return X, y


def test_causal_gru_output_shape():
    m = models.CausalGRU(6, 8, 3)
    out = m(torch.zeros(5, 4, 6))
    assert out.shape == (5, 3)


def test_dual_head_outputs():
    m = models.DualHeadGRU(6, 8, 3, 2)
    hs, ha = m(torch.zeros(5, 4, 6))
    assert hs.shape == (5, 3)
    assert ha.shape == (5, 2)


def test_lockstep_hidden_size_is_forty_eight():
    """The reported runs use 48 hidden units in every GRU; keep the default path honest."""
    m = models.CausalGRU(72, 48, 4)
    assert m.gru.hidden_size == 48
    assert m.gru.num_layers == 1


def test_predict_proba_rows_are_distributions():
    X, y = _data()
    torch.manual_seed(0)
    m = models.CausalGRU(6, 8, 3)
    models.train_gru(m, X, y, epochs=1, batch=16, subsample=1000)
    p = models.predict_proba(m, X)
    assert p.shape == (len(X), 3)
    assert np.allclose(p.sum(1), 1.0, atol=1e-5)


def test_training_is_deterministic_given_the_seed():
    X, y = _data()
    torch.manual_seed(7)
    m1 = models.CausalGRU(6, 8, 3)
    models.train_gru(m1, X, y, epochs=2, batch=16, subsample=1000)
    p1 = models.predict_proba(m1, X)

    torch.manual_seed(7)
    m2 = models.CausalGRU(6, 8, 3)
    models.train_gru(m2, X, y, epochs=2, batch=16, subsample=1000)
    p2 = models.predict_proba(m2, X)
    assert np.allclose(p1, p2)


def test_subsample_cap_is_respected():
    X, y = _data(n=200)
    torch.manual_seed(0)
    m = models.CausalGRU(6, 8, 3)
    models.train_gru(m, X, y, epochs=1, batch=16, subsample=32)
    # training on 32 of 200 windows still predicts for all of them
    assert models.predict_proba(m, X).shape[0] == 200


def test_auxiliary_head_trains():
    X, y = _data()
    y_aux = (y % 2).astype(int)
    torch.manual_seed(0)
    m = models.DualHeadGRU(6, 8, 3, 2)
    models.train_gru(m, X, y, y_aux=y_aux, epochs=1, batch=16, subsample=1000)
    hs, ha = m(torch.as_tensor(X[:3]))
    assert hs.shape == (3, 3)
    assert ha.shape == (3, 2)


def test_balanced_weights_upweight_rare_classes():
    y = np.array([0] * 10 + [1] * 2)
    w = models.balanced_weights(y, 2).numpy()
    assert w[0] < w[1]


def test_single_frame_keeps_full_class_order():
    rng = np.random.RandomState(0)
    X = rng.randn(60, 4)
    y = np.array([0] * 30 + [2] * 30)          # class 1 absent from training
    p, clf, scaler = models.fit_single_frame(X, y, X[:5], 3)
    assert p.shape == (5, 3)
    assert np.allclose(p[:, 1], 0.0)
    assert set(clf.classes_) == {0, 2}
    again = models.predict_softmax(clf, scaler, X[:5], 3)
    assert np.allclose(p, again)


def test_gate_output_is_a_distribution():
    rng = np.random.RandomState(0)
    sf = rng.dirichlet(np.ones(3), size=20)
    ctx = rng.dirichlet(np.ones(3), size=20)
    pact = rng.dirichlet(np.ones(2), size=20)
    y = rng.randint(0, 3, 20)
    w1, w2, b = models.gate_train(sf, ctx, pact, y, epochs=5)
    assert all(np.isfinite([w1, w2, b]))
    out = models.gate_apply(sf, ctx, pact, w1, w2, b)
    assert out.shape == (20, 3)
    assert np.allclose(out.sum(1), 1.0)


def test_gate_can_reproduce_either_extreme():
    """The gate weight is learned, including its sign: a large negative weight on
    the action term drives the output to the context posterior."""
    sf = np.tile(np.array([[0.9, 0.1]]), (4, 1))
    ctx = np.tile(np.array([[0.1, 0.9]]), (4, 1))
    pact = np.tile(np.array([[0.99, 0.01]]), (4, 1))
    out = models.gate_apply(sf, ctx, pact, -1e3, 0.0, 0.0)
    assert np.allclose(out, ctx, atol=1e-6)
    out_sf = models.gate_apply(sf, ctx, pact, 1e3, 0.0, 0.0)
    assert np.allclose(out_sf, sf, atol=1e-6)


@pytest.mark.parametrize("n_classes", [2, 9])
def test_predict_proba_handles_class_counts(n_classes):
    X, y = _data(classes=n_classes, n=80)
    torch.manual_seed(0)
    m = models.CausalGRU(6, 8, n_classes)
    models.train_gru(m, X, y, epochs=1, batch=16, subsample=1000)
    assert models.predict_proba(m, X).shape == (80, n_classes)


def test_train_gru_accepts_a_model_factory():
    """A factory lets the model be initialised after the subsample draw."""
    X, y = _data(n=120)
    torch.manual_seed(11)
    m = models.train_gru(lambda: models.CausalGRU(6, 8, 3), X, y,
                         epochs=1, batch=16, subsample=32)
    assert isinstance(m, torch.nn.Module)
    assert models.predict_proba(m, X).shape == (120, 3)


def test_factory_path_is_deterministic():
    X, y = _data(n=120)
    torch.manual_seed(12)
    a = models.train_gru(lambda: models.CausalGRU(6, 8, 3), X, y, epochs=1, batch=16, subsample=32)
    pa = models.predict_proba(a, X)
    torch.manual_seed(12)
    b = models.train_gru(lambda: models.CausalGRU(6, 8, 3), X, y, epochs=1, batch=16, subsample=32)
    pb = models.predict_proba(b, X)
    assert np.allclose(pa, pb)
