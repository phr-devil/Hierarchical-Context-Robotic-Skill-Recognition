"""Pre-registered decision rule, including the interval-width clause."""
import numpy as np
import pytest

from skillseg import metrics


def _decision(config, deltas):
    return metrics.decide(deltas, config)


def test_clear_effect_is_confirmed_without_a_width_flag(config):
    deltas = [0.05 + 0.005 * ((i % 5) - 2) for i in range(37)]
    out = _decision(config, deltas)
    assert out["verdict"] == "CONFIRMED"
    assert out["underpowered"] is False
    assert out["clause_conflict"] is False
    assert out["verdict_strict"] == "CONFIRMED"
    assert out["width"] < config.prereg["max_ci_width"]


def test_null_effect_is_not_confirmed(config):
    out = _decision(config, [0.0] * 37)
    assert out["verdict"] == "NOT_CONFIRMED"
    assert out["ci95"][1] < config.prereg["equiv_upper"]


def test_symmetric_noise_is_inconclusive(config):
    deltas = [0.05] * 18 + [-0.05] * 19
    out = _decision(config, deltas)
    assert out["verdict"] == "INCONCLUSIVE"
    assert out["ci95"][0] < 0 < out["ci95"][1]


def test_width_conflict_returns_both_readings(config):
    """The reported run's situation: all ordered conditions met, interval too wide."""
    deltas = [0.05] * 36 + [0.90]
    out = _decision(config, deltas)
    assert out["verdict"] == "CONFIRMED"          # ordered test as coded
    assert out["underpowered"] is True            # width above 2.0 pt
    assert out["clause_conflict"] is True
    assert out["verdict_strict"] == "INCONCLUSIVE"  # width treated as a condition


def test_effect_below_the_floor_cannot_be_confirmed(config):
    deltas = [0.004 + 0.001 * ((i % 3) - 1) for i in range(37)]
    out = _decision(config, deltas)
    assert out["verdict"] != "CONFIRMED"


def test_too_few_improved_episodes_blocks_confirmation(config):
    deltas = [0.30] * 20 + [-0.01] * 17
    out = _decision(config, deltas)
    assert out["sign_pos"] == 20
    assert out["verdict"] != "CONFIRMED"


def test_decision_reports_seed_and_cluster_metadata():
    deltas = np.linspace(0.01, 0.08, 37).tolist()
    out = metrics.decide(deltas, _config_stub())
    assert out["n_episodes"] == 37
    assert "verdict_strict" in out and "clause_conflict" in out


def _config_stub():
    """Minimal stand-in so the metadata test does not need a config file."""
    from types import SimpleNamespace
    return SimpleNamespace(
        prereg={"confirm_delta": 0.01, "equiv_upper": 0.005,
                "min_improved": 24, "max_ci_width": 0.02},
        bootstrap_resamples=1000, bootstrap_seed=20261111,
    )
