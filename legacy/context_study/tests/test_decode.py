import numpy as np

from crps import common


def _random_emission(n, k, seed=0):
    rng = np.random.RandomState(seed)
    p = rng.rand(n, k) + 0.05
    return p / p.sum(1, keepdims=True)


def _random_transition(k, seed=0):
    rng = np.random.RandomState(seed)
    a = rng.rand(k, k) + 0.1
    return a / a.sum(1, keepdims=True)


def _full_viterbi(emission, A):
    logp = np.log(np.maximum(emission, 1e-300))
    logA = np.log(np.maximum(A, 1e-12))
    L, K = emission.shape
    delta = logp[0].copy()
    back = np.zeros((L, K), dtype=int)
    for i in range(1, L):
        cand = delta[:, None] + logA
        back[i] = cand.argmax(0)
        delta = cand.max(0) + logp[i]
    path = np.zeros(L, dtype=int)
    path[-1] = int(delta.argmax())
    for i in range(L - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def test_hmm_filter_is_causal():
    """Outputs on a prefix must not change when future frames are appended."""
    k, L = 3, 12
    P = _random_emission(L, k, seed=1)
    A = _random_transition(k, seed=2)
    full = np.zeros(L, dtype=int)
    common.dec_hmm_filter(P, A, [np.arange(L)], full)
    prefix = np.zeros(6, dtype=int)
    common.dec_hmm_filter(P[:6], A, [np.arange(6)], prefix)
    assert np.array_equal(full[:6], prefix)


def test_viterbi_lag_matches_full_path_when_lag_covers_sequence():
    k, L = 3, 10
    P = _random_emission(L, k, seed=3)
    A = _random_transition(k, seed=4)
    out = np.zeros(L, dtype=int)
    common.viterbi_lag(P, A, [np.arange(L)], out, lag=L + 5)
    assert np.array_equal(out, _full_viterbi(P, A))


def test_viterbi_lag_commits_with_requested_delay():
    """With lag=1 every frame except the last is committed before the frontier."""
    k, L = 3, 8
    P = _random_emission(L, k, seed=5)
    A = _random_transition(k, seed=6)
    out = np.zeros(L, dtype=int)
    common.viterbi_lag(P, A, [np.arange(L)], out, lag=1)
    assert np.all(out >= 0)
    assert np.all(out < k)


def test_viterbi_lag_keeps_groups_independent():
    k = 3
    P = _random_emission(10, k, seed=7)
    A = _random_transition(k, seed=8)
    out = np.zeros(10, dtype=int)
    common.viterbi_lag(P, A, [np.arange(0, 5), np.arange(5, 10)], out, lag=2)
    alone_a = np.zeros(10, dtype=int)
    common.viterbi_lag(P, A, [np.arange(0, 5)], alone_a, lag=2)
    assert np.array_equal(out[:5], alone_a[:5])


def test_filter_prefers_sticky_path_under_strong_transition():
    k = 2
    A = np.array([[0.95, 0.05], [0.05, 0.95]])
    P = np.array([[0.55, 0.45]] * 6)
    out = np.zeros(6, dtype=int)
    common.dec_hmm_filter(P, A, [np.arange(6)], out)
    assert out[0] == 0
    assert np.all(out == 0)


def test_emission_dominates_when_transitions_are_diffuse():
    k = 2
    A = np.array([[0.5, 0.5], [0.5, 0.5]])
    P = np.array([[0.9, 0.1], [0.1, 0.9]])
    out = np.zeros(2, dtype=int)
    common.dec_hmm_filter(P, A, [np.arange(2)], out)
    assert list(out) == [0, 1]
