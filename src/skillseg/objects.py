"""Assembly objects, object runs and scoring cells.

The corpus annotates one assembly object per parent action segment.  A scoring
cell is a pair ``(episode, object)``; inside a cell, the frames are split into
*runs* -- maximal stretches of consecutive grid frames that belong to that object
and are not background.  Two separate visits to the same object in one episode
therefore form two runs, and segment matching never bridges them.

The object is a scoring device only.  It is never an input to any arm, and it is
not a statistical unit: inference is episode-level (see ``metrics``).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import ARTIFACT_OBJECT, BACKGROUND_OBJECT
from .data import Grid
from .features import episode_sequences

Run = Tuple[str, np.ndarray]


def parse_object(text: str) -> str:
    """Object name from a parent action string.

    The corpus writes e.g. ``"insert the small gear"``; the object is the text
    after the leading verb.  A leading article is kept because the corpus uses it
    consistently and distinct strings are distinct objects.
    """
    text = text.strip()
    match = re.match(r"^[A-Za-z]+\s+(.+?)\.?$", text)
    return match.group(1).strip() if match else text


def runs_for(grid: Grid, target: str,
             ep_index: Optional[Dict[str, np.ndarray]] = None) -> List[Run]:
    """Runs of one object, as time-ordered frame index arrays."""
    if ep_index is None:
        ep_index = {ep: idx for ep, idx in episode_sequences(grid)}
    runs: List[Run] = []
    for ep, idx in ep_index.items():
        keep = (grid.obj[idx] == target) & (~grid.is_bg[idx])
        i = 0
        n = len(idx)
        while i < n:
            if keep[i]:
                j = i
                while j < n and keep[j]:
                    j += 1
                runs.append((f"{ep}_{i}", idx[i:j]))
                i = j
            else:
                i += 1
    return runs


def cells(grid: Grid,
          ep_index: Optional[Dict[str, np.ndarray]] = None,
          exclude_artifacts: bool = True) -> Dict[Tuple[str, str], List[Run]]:
    """``(episode, object) -> runs`` for every scored cell.

    The ``"action"`` pseudo-object is a frame-level artefact of the annotation
    format, not an assembly object; it is dropped from scoring by default.  The
    same flag is exposed so that the diagnostic code can reproduce the frozen
    behaviour, which also drops it.
    """
    if ep_index is None:
        ep_index = {ep: idx for ep, idx in episode_sequences(grid)}
    wanted = sorted(set(grid.obj[~grid.is_bg].tolist()))
    if exclude_artifacts:
        wanted = [o for o in wanted if o != ARTIFACT_OBJECT]
    out: Dict[Tuple[str, str], List[Run]] = {}
    for obj in wanted:
        for key, idx in runs_for(grid, obj, ep_index):
            episode = key.split("_")[0]
            out.setdefault((episode, obj), []).append((key, idx))
    return out


def all_runs(cell_map: Dict[Tuple[str, str], List[Run]]) -> List[Run]:
    """Flattened run list, used for the pooled (frame-pooled) score."""
    runs: List[Run] = []
    for key in sorted(cell_map):
        runs.extend(cell_map[key])
    return runs


def real_objects(grid: Grid) -> List[str]:
    """Objects that actually carry robot interaction, excluding background and artefacts."""
    return [o for o in sorted(set(grid.obj[~grid.is_bg].tolist()))
            if o not in (ARTIFACT_OBJECT, BACKGROUND_OBJECT)]
