"""Per-frame assembly-object labels and the leave-one-object-out partition.

The object of a frame is the object named in the text of its enclosing parent
segment (``Pick square peg 4.`` -> ``square peg 4``). Frames belonging to the
``action`` label are an artefact of the annotation (generic action segments that
name no object) and are excluded from the object folds; background frames are
unknown by construction.
"""
import numpy as np

UNKNOWN = "UNKNOWN"
ARTIFACT_OBJECT = "action"


def build_object_grid(frames, grid, table):
    """Align object names of annotated frames onto the continuous grid.

    Returns
    -------
    obj_grid : ndarray of str, shape (n_grid,)
    episodes : dict object -> set of episode ids (annotated frames only)
    objects  : list of object names, sorted, artefact and unknown removed
    """
    seg = frames["seg"]
    ep = frames["ep"]
    lookup = {}
    for i in range(len(ep)):
        lookup[(ep[i], round(float(frames["t"][i]), 6))] = table.get((ep[i], int(seg[i])), UNKNOWN)

    n = len(grid["t"])
    obj_grid = np.array([UNKNOWN] * n, dtype=object)
    matched = 0
    for i in range(n):
        if grid["is_bg"][i]:
            continue
        key = (grid["ep"][i], round(float(grid["t"][i]), 6))
        if key in lookup:
            obj_grid[i] = lookup[key]
            matched += 1

    episodes = {}
    for i in range(n):
        if grid["is_bg"][i]:
            continue
        episodes.setdefault(obj_grid[i], set()).add(grid["ep"][i])

    objects = sorted(o for o in episodes if o not in (UNKNOWN, ARTIFACT_OBJECT))
    return obj_grid.astype(str), episodes, objects, matched


def object_frame_counts(obj_grid):
    counts = {}
    for o in obj_grid:
        counts[o] = counts.get(o, 0) + 1
    return counts
