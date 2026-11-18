"""The three arms: shared encoder, frozen probe, encoder-free single-frame arm."""
import numpy as np
import pytest

from skillseg import models

torch = pytest.importorskip("torch", reason="the recurrent arms need torch")


def _toy(n=96, dim=72, k=9, seed=0):
    rng = np.random.RandomState(seed)
    stacked = rng.randn(n, 15, dim).astype(np.float32)
    frames = stacked[:, -1, :].copy()
    labels = rng.randint(0, k, n).astype(np.int64)
    return stacked, frames, labels, np.arange(n)


def test_arms_share_one_frozen_encoder(config):
    stacked, frames, labels, rows = _toy()
    arms = models.build_arms(stacked, frames, labels, rows, config, 0)

    models.assert_paired(arms)
    assert arms.encoder_sha_before_probe == arms.encoder_sha_after_probe
    assert models.encoder_is_frozen(arms.encoder)
    assert all(not p.requires_grad for p in arms.encoder.gru.parameters())


def test_probe_reads_the_hidden_state_not_the_input(config):
    stacked, frames, labels, rows = _toy()
    arms = models.build_arms(stacked, frames, labels, rows, config, 0)
    hidden = int(config.gru["hidden"])
    assert arms.probe.clf.coef_.shape == (config.n_skills, hidden)
    assert arms.probe.n_features == hidden
    assert arms.single_frame.clf.coef_.shape == (config.n_skills, config.feature_dim)


def test_single_frame_arm_has_no_encoder(config):
    stacked, frames, labels, rows = _toy()
    arms = models.build_arms(stacked, frames, labels, rows, config, 0)
    assert arms.single_frame.encoder_sha == "no-encoder"
    assert not hasattr(arms.single_frame, "encoder")


def test_probe_and_head_see_the_same_rows(config):
    stacked, frames, labels, rows = _toy()
    arms = models.build_arms(stacked, frames, labels, rows, config, 0)
    assert arms.probe.rows_sha == arms.single_frame.rows_sha
    assert arms.probe.n_rows == len(rows) == arms.single_frame.n_rows


def test_seeds_produce_different_encoders(config):
    stacked, frames, labels, rows = _toy()
    a = models.build_arms(stacked, frames, labels, rows, config, 0)
    b = models.build_arms(stacked, frames, labels, rows, config, 1)
    assert a.train_seed != b.train_seed
    assert models.params_fingerprint(a.encoder.gru) != models.params_fingerprint(b.encoder.gru)


def test_corrupting_the_encoder_breaks_pairing(config):
    stacked, frames, labels, rows = _toy()
    arms = models.build_arms(stacked, frames, labels, rows, config, 0)
    with torch.no_grad():
        for parameter in arms.encoder.gru.parameters():
            parameter.add_(0.01)
    with pytest.raises(AssertionError):
        models.assert_paired(arms)


def test_arm_probabilities_are_distributions(config):
    stacked, frames, labels, rows = _toy()
    arms = models.build_arms(stacked, frames, labels, rows, config, 0)
    probs = models.arm_probabilities(arms, stacked, frames, config.n_skills)
    for arm, p in probs.items():
        assert p.shape == (len(stacked), config.n_skills), arm
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5), arm


def test_freeze_is_idempotent_and_does_not_change_values(config):
    stacked, frames, labels, rows = _toy()
    model, _ = models.train_joint_arm(stacked, labels, rows, config, config.train_seed(0))
    before = models.params_fingerprint(model.gru)
    models.freeze_encoder(model)
    models.freeze_encoder(model)
    assert models.params_fingerprint(model.gru) == before
