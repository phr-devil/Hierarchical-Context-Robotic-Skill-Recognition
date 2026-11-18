import json
import os

import numpy as np

from crps import config as cfgmod, features


def _reference(repo_root):
    with open(os.path.join(repo_root, "results", "reference_metrics.json"), "r") as fh:
        return json.load(fh)


def test_default_config_is_loadable():
    cfg = cfgmod.load()
    for key in ["data", "single_frame", "gru", "gate", "decode", "evaluation", "seeds"]:
        assert key in cfg


def test_default_config_matches_the_reported_run():
    """The shipped config must describe the run whose numbers are in results/."""
    cfg = cfgmod.load()
    assert cfg["data"]["window"] == 0.4
    assert cfg["data"]["stride"] == 0.2
    assert cfg["data"]["history_frames"] == 15          # 3 s at a 0.2 s stride
    assert cfg["gru"]["hidden"] == 48
    assert cfg["gru"]["layers"] == 1
    assert cfg["gru"]["epochs"] == 10
    assert cfg["gru"]["batch"] == 256
    assert cfg["gru"]["subsample"] == 60000
    assert cfg["gru"]["lr"] == 0.001
    assert cfg["dual_head"]["action_aux_weight"] == 1.0
    assert cfg["decode"]["lag_frames"] == 3
    assert cfg["decode"]["nominal_wait_seconds"] == 0.6
    assert cfg["seeds"]["global"] == 0
    assert cfg["seeds"]["volume_match_seeds"] == 5


def test_history_covers_the_declared_seconds():
    cfg = cfgmod.load()
    seconds = cfg["data"]["history_frames"] * cfg["data"]["stride"]
    assert abs(seconds - 3.0) < 1e-9


def test_fused_dimension_matches_the_paper(repo_root):
    ref = _reference(repo_root)
    assert ref["protocol"]["frame_feature_dim"] == features.FRAME_FEATURE_DIM
    assert ref["protocol"]["fused_feature_dim"] == features.FRAME_FEATURE_DIM + ref["protocol"]["parent_actions"]


def test_reference_metrics_are_internally_consistent(repo_root):
    ref = _reference(repo_root)
    main = ref["main_run"]["segmental_f1_mean_over_17_object_folds"]
    five = ref["five_model_round"]["segmental_f1_mean_over_17_object_folds"]
    # the two runs are separate, but both share the same context-free baseline
    assert main["SF"] == five["SF"] == 0.6717
    # the main run and the five-model round report different GRUctx numbers
    assert main["GRUctx"] != five["GRUctx"]
    for name, value in five.items():
        assert 0.0 <= value <= 1.0, name


def test_reference_metrics_state_their_limitations(repo_root):
    ref = _reference(repo_root)
    text = " ".join(ref["known_limitations"]).lower()
    assert "sealed" in text
    assert "seed" in text
    assert "boundary" in text


def test_set_seeds_is_reproducible():
    cfg = cfgmod.load()
    cfgmod.set_seeds(cfg)
    a = np.random.rand(3)
    cfgmod.set_seeds(cfg)
    b = np.random.rand(3)
    assert np.allclose(a, b)
