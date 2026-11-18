"""Corpus access and construction of the 5 Hz evaluation grid.

The grid is not the raw annotation.  A frame is placed at ``start + window`` and
then every ``stride`` seconds, so a segment shorter than the 0.4 s window
contributes no frame at all, and a segment of 0.6 s contributes two.  Frames that
fall between annotated segments are inserted as background.  This construction is
what the reported scores are computed on; it is a property of the evaluation, not
of the model, and Section 4 of the manuscript is about its consequences.

The implementation below is a path-parameterised port of the frozen scoring code.
It was verified to rebuild the development grid bit-identically: 111 episodes,
185,451 frames, identical labels, frame times and 72-dimensional features
(max absolute difference 0.0).  See ``REPRODUCIBILITY.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from .config import BACKGROUND_OBJECT, Config


def _text(value) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", "replace")
    return str(value)


@dataclass
class Grid:
    """Evaluation grid for one split.

    ``label`` is the skill index, or ``-1`` for background.  ``obj`` is the
    assembly object the frame belongs to, ``"UNKNOWN"`` for background, and
    ``"action"`` for the frame-level pseudo-object present in the corpus.
    """

    ep: np.ndarray
    t: np.ndarray
    label: np.ndarray
    pact: np.ndarray
    date: np.ndarray
    obj: np.ndarray

    def __len__(self) -> int:
        return int(self.t.shape[0])

    @property
    def is_bg(self) -> np.ndarray:
        return self.label < 0

    @property
    def episodes(self) -> List[str]:
        return sorted(set(self.ep.tolist()))

    def subset(self, episodes: Sequence[str]) -> "Grid":
        keep = np.isin(self.ep, list(episodes))
        return Grid(self.ep[keep], self.t[keep], self.label[keep],
                    self.pact[keep], self.date[keep], self.obj[keep])

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, ep=self.ep, t=self.t, label=self.label,
                            pact=self.pact, date=self.date, obj=self.obj)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Grid":
        z = np.load(path, allow_pickle=True)
        return cls(
            ep=np.array([str(x) for x in z["ep"]]),
            t=z["t"].astype(np.float64),
            label=z["label"].astype(np.int64),
            pact=np.array([str(x) for x in z["pact"]]),
            date=np.array([str(x) for x in z["date"]]),
            obj=np.array([str(x) for x in z["obj"]]),
        )


def episode_dates(episodes: Iterable[str]) -> Dict[str, str]:
    """Recording date of an episode, taken from its identifier ``YYYY-MM-DD-...``."""
    return {ep: "-".join(ep.split("-")[:3]) for ep in episodes}


def annotated_frames(episodes: Sequence[str], cfg: Config) -> List[Tuple[str, int, float, int, str, str]]:
    """Enumerate grid frames contributed by annotated segments.

    A frame is emitted at ``start + window_s`` and then every ``stride_s`` seconds
    up to ``end``.  Segments shorter than ``window_s`` are skipped entirely -- this
    is the mechanism behind the grid censoring analysed in the manuscript.
    """
    index = cfg.skill_index
    out: List[Tuple[str, int, float, int, str, str]] = []
    for ep in episodes:
        date = "-".join(ep.split("-")[:3])
        with h5py.File(cfg.dataset.episode_path(ep), "r") as h:
            segments = h["segments_info"]
            for key in segments.keys():
                group = segments[key]
                if "low_level" not in group:
                    continue
                pact = _text(group["text"][()]).strip().split()[0].lower()
                for j in group["low_level"].keys():
                    seg = group["low_level"][j]
                    start = float(seg["start"][()])
                    end = float(seg["end"][()])
                    skill = _text(seg["text"][()]).strip()
                    if end - start < cfg.window_s + 1e-6:
                        continue
                    if skill not in index:
                        raise ValueError(f"unknown skill label {skill!r} in {ep}: the class table is fixed")
                    for t in np.arange(start + cfg.window_s, end + 1e-9, cfg.stride_s):
                        out.append((ep, int(key), float(t), index[skill], pact, date))
    return out


def object_map(episodes: Sequence[str], cfg: Config) -> Dict[Tuple[str, int], str]:
    """Map ``(episode, parent-segment index)`` to the assembly object in its text.

    The corpus names the object inside the parent action text, e.g.
    ``"insert the small gear"``; the label is the remainder after the verb.
    """
    from .objects import parse_object

    mapping: Dict[Tuple[str, int], str] = {}
    for ep in episodes:
        with h5py.File(cfg.dataset.episode_path(ep), "r") as h:
            segments = h["segments_info"]
            for key in segments.keys():
                group = segments[key]
                if "low_level" not in group:
                    continue
                mapping[(ep, int(key))] = parse_object(_text(group["text"][()]))
    return mapping


def build_grid(episodes: Sequence[str], cfg: Config, verbose: bool = False) -> Grid:
    """Annotated frames plus inter-segment background frames, in episode order."""
    frames = annotated_frames(episodes, cfg)
    objects = object_map(episodes, cfg)

    by_episode: Dict[str, List[Tuple[str, int, float, int, str, str]]] = {}
    for row in frames:
        by_episode.setdefault(row[0], []).append(row)

    ep_out: List[str] = []
    t_out: List[float] = []
    label_out: List[int] = []
    pact_out: List[str] = []
    date_out: List[str] = []
    obj_out: List[str] = []

    for ep in sorted(by_episode):
        rows = sorted(by_episode[ep], key=lambda r: r[2])
        for row in rows:
            ep_out.append(row[0]); t_out.append(row[2]); label_out.append(row[3])
            pact_out.append(row[4]); date_out.append(row[5])
            obj_out.append(objects.get((row[0], row[1]), BACKGROUND_OBJECT))
        # background frames fill the gaps between consecutive annotated frames
        for k in range(1, len(rows)):
            gap = rows[k][2] - rows[k - 1][2]
            if gap > cfg.stride_s + 1e-6:
                for tb in np.arange(rows[k - 1][2] + cfg.stride_s, rows[k][2], cfg.stride_s):
                    ep_out.append(ep); t_out.append(float(tb)); label_out.append(-1)
                    pact_out.append(rows[k - 1][4] if rows[k - 1][4] == rows[k][4] else "none")
                    date_out.append(rows[k][5]); obj_out.append(BACKGROUND_OBJECT)

    grid = Grid(
        ep=np.array(ep_out, dtype=object),
        t=np.array(t_out, dtype=np.float64),
        label=np.array(label_out, dtype=np.int64),
        pact=np.array(pact_out, dtype=object),
        date=np.array(date_out, dtype=object),
        obj=np.array(obj_out, dtype=object),
    )
    if verbose:
        n_bg = int(grid.is_bg.sum())
        print(f"grid frames={len(grid)} annotated={len(grid) - n_bg} background={n_bg} "
              f"({100.0 * n_bg / max(len(grid), 1):.2f}%) episodes={len(grid.episodes)}")
    return grid


def native_segments(episodes: Sequence[str], cfg: Config) -> List[Tuple[str, str, float, float]]:
    """Raw low-level segments ``(episode, skill, start, end)``, independent of the grid.

    Used only by the grid-censoring diagnostic, which reads the label side alone.
    """
    rows: List[Tuple[str, str, float, float]] = []
    for ep in episodes:
        with h5py.File(cfg.dataset.episode_path(ep), "r") as h:
            segments = h["segments_info"]
            for key in segments.keys():
                group = segments[key]
                if "low_level" not in group:
                    continue
                for j in group["low_level"].keys():
                    seg = group["low_level"][j]
                    rows.append((ep, _text(seg["text"][()]).strip(),
                                 float(seg["start"][()]), float(seg["end"][()])))
    return rows
