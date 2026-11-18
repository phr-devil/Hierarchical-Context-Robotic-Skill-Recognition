"""Configuration loading.

Every numeric constant that the frozen protocol fixed is read from
``configs/default.json`` rather than hard-coded, so that a reader can see the
full protocol in one place.  Two conventions matter:

* rates are stored as fractions in ``[0, 1]`` (0.01 = 1 percentage point);
* the class order is fixed by the corpus, not by the data, so that a label index
  means the same thing across splits, arms and seeds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

SKILLS: Tuple[str, ...] = (
    "Align", "Approach", "Grasp", "Lift", "Nudge", "Pull", "Push", "Release", "Twist",
)
ACTIONS: Tuple[str, ...] = ("insert", "pick", "place", "remove")

ARTIFACT_OBJECT = "action"   # frame-level pseudo-object, excluded from scoring
BACKGROUND_OBJECT = "UNKNOWN"


@dataclass(frozen=True)
class DatasetConfig:
    """Location of the third-party corpus and of its official split files."""

    root: Path
    data_subdir: str = "data/data"
    split_subdir: str = "splits"
    split_suffix: str = "_split1.txt"

    @property
    def data_dir(self) -> Path:
        return self.root / self.data_subdir

    @property
    def split_dir(self) -> Path:
        return self.root / self.split_subdir

    def episode_path(self, episode: str) -> Path:
        return self.data_dir / f"{episode}.h5"

    def split(self, which: str) -> List[str]:
        """Official split membership.  ``which`` is ``train`` or ``test``."""
        path = self.split_dir / f"{which}{self.split_suffix}"
        if not path.exists():
            raise FileNotFoundError(
                f"official split file not found: {path}. Point dataset.root at the "
                f"REASSEMBLE release, which ships {self.split_subdir}/."
            )
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass(frozen=True)
class Config:
    dataset: DatasetConfig
    skills: Tuple[str, ...]
    actions: Tuple[str, ...]
    window_s: float
    stride_s: float
    history_frames: int
    proprio_channels: Tuple[str, ...]
    force_channels: Tuple[str, ...]
    proprio_dims: Tuple[int, ...]
    force_dims: Tuple[int, ...]
    gru: Dict[str, Any]
    readout: Dict[str, Any]
    lag_frames: int
    iou_threshold: float
    bootstrap_resamples: int
    bootstrap_seed: int
    prereg: Dict[str, float]
    seeds: Tuple[int, ...]
    seed_base: int
    seed_stride: int
    seed_offset: int
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    # -- derived ---------------------------------------------------------
    @property
    def n_skills(self) -> int:
        return len(self.skills)

    @property
    def skill_index(self) -> Dict[str, int]:
        return {name: i for i, name in enumerate(self.skills)}

    @property
    def feature_dim(self) -> int:
        return 3 * sum(self.proprio_dims) + 2 * sum(self.force_dims)

    def train_seed(self, index: int) -> int:
        """Training seed for the ``index``-th repeat.

        The offset is not cosmetic: it is the value the frozen run used, and it
        is part of the protocol.  Reproducing the published numbers requires this
        exact derivation.
        """
        return self.seed_base + self.seed_stride * int(index) + self.seed_offset


def _tuple(x: Sequence[Any]) -> Tuple[Any, ...]:
    return tuple(x)


def load_config(path: str | Path) -> Config:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    ds = raw["dataset"]
    root = Path(ds.get("root") or "").expanduser()
    cfg = Config(
        dataset=DatasetConfig(
            root=root,
            data_subdir=ds.get("data_subdir", "data/data"),
            split_subdir=ds.get("split_subdir", "splits"),
            split_suffix=ds.get("split_suffix", "_split1.txt"),
        ),
        skills=_tuple(raw["grid"].get("skills", SKILLS)),
        actions=_tuple(raw["grid"].get("actions", ACTIONS)),
        window_s=float(raw["grid"]["window_s"]),
        stride_s=float(raw["grid"]["stride_s"]),
        history_frames=int(raw["features"]["history_frames"]),
        proprio_channels=_tuple(raw["features"]["proprio_channels"]),
        force_channels=_tuple(raw["features"]["force_channels"]),
        proprio_dims=_tuple(int(v) for v in raw["features"]["proprio_dims"]),
        force_dims=_tuple(int(v) for v in raw["features"]["force_dims"]),
        gru=dict(raw["arms"]["a"]),
        readout=dict(raw["arms"]["p"]),
        lag_frames=int(raw["decode"]["lag_frames"]),
        iou_threshold=float(raw["evaluation"]["iou_threshold"]),
        bootstrap_resamples=int(raw["evaluation"]["bootstrap_resamples"]),
        bootstrap_seed=int(raw["evaluation"]["bootstrap_seed"]),
        prereg={k: float(v) for k, v in raw["prereg"].items()},
        seeds=_tuple(int(s) for s in raw["seeds"]["values"]),
        seed_base=int(raw["seeds"]["base"]),
        seed_stride=int(raw["seeds"]["stride"]),
        seed_offset=int(raw["seeds"]["offset"]),
        raw=raw,
    )
    if cfg.feature_dim != 72:
        raise ValueError(f"feature_dim={cfg.feature_dim}, expected 72 (60 proprioceptive + 12 force/torque)")
    if cfg.n_skills != 9:
        raise ValueError(f"{cfg.n_skills} skill classes configured, the corpus defines 9")
    return cfg
