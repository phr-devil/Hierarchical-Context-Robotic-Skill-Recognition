import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from skillseg.config import Config, DatasetConfig, load_config  # noqa: E402
from skillseg.data import Grid  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture()
def config(tmp_path: Path) -> Config:
    """Protocol configuration with a small network and a throwaway dataset root."""
    raw = json.loads((REPO_ROOT / "configs" / "default.json").read_text(encoding="utf-8"))
    raw["dataset"]["root"] = str(tmp_path)
    raw["arms"]["a"].update({"hidden": 8, "epochs": 1, "batch": 32})
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return load_config(path)


@pytest.fixture()
def toy_grid() -> Grid:
    """Two episodes, two objects, one background gap, one repeated visit."""
    ep = ["E1"] * 10 + ["E2"] * 6
    t = [0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 1.8, 2.0, 2.2, 2.4,
         0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    label = [0, 0, 1, 1, 1, -1, 2, 2, 2, 2,
             1, 1, 1, 1, 3, 3]
    obj = ["gear"] * 10 + ["gear"] * 2 + ["bolt"] * 4
    return Grid(
        ep=np.array(ep, dtype=object),
        t=np.array(t, dtype=np.float64),
        label=np.array(label, dtype=np.int64),
        pact=np.array(["insert"] * 16, dtype=object),
        date=np.array(["2024-01-01"] * 16, dtype=object),
        obj=np.array(obj, dtype=object),
    )
