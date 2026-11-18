"""Configuration loading and the frozen seed derivation."""
import json

import pytest

from skillseg.config import load_config


def test_protocol_values_match_the_frozen_run(config):
    assert config.n_skills == 9
    assert config.feature_dim == 72
    assert config.window_s == 0.4 and config.stride_s == 0.2
    assert config.history_frames == 15
    assert config.lag_frames == 3
    assert config.iou_threshold == 0.5
    assert config.bootstrap_resamples == 4000
    assert tuple(config.seeds) == (0, 1, 2)
    assert config.prereg["max_ci_width"] == 0.02


def test_seed_derivation_is_fixed(config):
    assert [config.train_seed(i) for i in range(3)] == [20261128, 20262128, 20263128]


def test_split_files_are_read_verbatim(config, tmp_path):
    split_dir = tmp_path / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train_split1.txt").write_text("a\nb\n\n", encoding="utf-8")
    (split_dir / "test_split1.txt").write_text("c\n", encoding="utf-8")
    assert config.dataset.split("train") == ["a", "b"]
    assert config.dataset.split("test") == ["c"]


def test_missing_split_file_names_the_expected_path(config):
    with pytest.raises(FileNotFoundError) as exc:
        config.dataset.split("train")
    assert "train_split1.txt" in str(exc.value)


def test_bad_feature_dim_is_rejected(tmp_path, config):
    raw = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    raw["features"]["force_dims"] = [3]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="72"):
        load_config(bad)
