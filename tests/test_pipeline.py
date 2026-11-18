"""Pipeline-level invariants: seed pairing checks and the censoring diagnostic."""
import hashlib

import numpy as np
import pytest

from skillseg import pipeline


def _info(index, **over):
    base = {
        "seed_index": index,
        "encoder_sha_before_probe": f"enc{index}",
        "encoder_sha_after_probe": f"enc{index}",
        "n_rows_head": 100,
        "n_rows_probe": 100,
        "encoder_frozen": True,
        "encoder_sha256_16": f"encoder-{index}",
        "head_sha256_16": f"head-{index}",
        "probe_sha256_16": f"probe-{index}",
        "train_rows_sha16": "same-rows",
    }
    base.update(over)
    return base


def test_pairing_accepts_distinct_seeds_on_the_same_rows():
    out = pipeline.verify_seed_pairing([_info(0), _info(1), _info(2)])
    assert out["n_seeds"] == 3
    assert out["train_rows_sha16"] == "same-rows"


def test_pairing_rejects_repeated_encoders():
    with pytest.raises(AssertionError, match="repeats across seeds"):
        pipeline.verify_seed_pairing([_info(0), _info(1, encoder_sha256_16="encoder-0")])


def test_pairing_rejects_a_moving_encoder():
    with pytest.raises(AssertionError, match="changed while fitting the probe"):
        pipeline.verify_seed_pairing([_info(0, encoder_sha_after_probe="other"), _info(1)])


def test_pairing_rejects_unfrozen_encoders():
    with pytest.raises(AssertionError, match="not frozen"):
        pipeline.verify_seed_pairing([_info(0, encoder_frozen=False), _info(1)])


def test_pairing_rejects_different_training_rows():
    with pytest.raises(AssertionError, match="training rows differ"):
        pipeline.verify_seed_pairing([_info(0), _info(1, train_rows_sha16="other")])


def test_pairing_rejects_duplicate_seed_index():
    with pytest.raises(AssertionError, match="duplicate seed index"):
        pipeline.verify_seed_pairing([_info(0), _info(0)])


def test_sha256_file_is_stable(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc")
    assert pipeline.sha256_file(path) == hashlib.sha256(b"abc").hexdigest()


def test_censoring_counts_frames_inside_the_grid(config, toy_grid):
    segments = [
        ("E1", "Grasp", 0.0, 0.3),     # shorter than the window: no grid frame at all
        ("E1", "Grasp", 0.3, 1.5),     # five grid frames
        ("E2", "Lift", 0.2, 1.5),      # six grid frames
    ]
    stats = pipeline.grid_censoring(toy_grid, segments, config)
    assert stats["n_segments"] == 3
    assert stats["per_class"]["Grasp"]["n"] == 2
    assert stats["per_class"]["Grasp"]["share_zero_frames"] == pytest.approx(0.5)
    assert stats["pooled"]["share_at_most_two_frames"] == pytest.approx(1 / 3)
    assert stats["per_class"]["Lift"]["median_frames"] == pytest.approx(6.0)


def test_censoring_returns_unjudged_classes(config, toy_grid):
    stats = pipeline.grid_censoring(toy_grid, [("E1", "Grasp", 0.3, 1.5)], config)
    # a class with no segment in the split is reported as empty, not as a zero rate
    assert stats["per_class"]["Push"] == {"n": 0}
