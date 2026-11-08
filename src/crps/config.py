"""Configuration loading and global seeding."""
import json
import os
import random

import numpy as np

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "configs", "default.json")


def load(path=None):
    """Read a JSON config; ``None`` uses ``configs/default.json``."""
    path = path or DEFAULT_CONFIG
    with open(path, "r") as fh:
        return json.load(fh)


def set_seeds(cfg):
    """Seed python, numpy and torch from the config.

    A single global seed is used for every reported run; cross-seed robustness
    was not evaluated and the README states so.
    """
    s = cfg.get("seeds", {})
    seed = int(s.get("global", 0))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    return seed
