"""REASSEMBLE dataset access.

Directory layout expected under ``--data-root`` (a clone of the official
REASSEMBLE repository, see README):

    <data-root>/data/data/<episode>.h5
    <data-root>/splits/train_split1.txt
    <data-root>/splits/test_split1.txt

An episode file contains ``timestamps/<channel>``, ``robot_state/<channel>`` and
``segments_info/<k>``. Each ``segments_info`` entry carries a natural-language
``text`` for the parent assembly action and, when annotated at the low level, a
``low_level`` group of child segments with their own ``start``, ``end`` and
``text``.

Only the development split (``train_split1.txt``, 111 episodes) is read here.
The sealed holdout (``test_split1.txt``) is intentionally never opened.
"""
import os
import re

import h5py
import numpy as np

_PARENT_OBJECT_RE = re.compile(r"^[A-Za-z]+\s+(.+?)\.?$")


def read_split(path):
    """Episode identifiers listed in a split file, one per line."""
    with open(path, "r") as fh:
        return [line.strip() for line in fh if line.strip()]


def episode_path(data_root, episode):
    return os.path.join(data_root, "data", "data", episode + ".h5")


def episode_date(episode):
    """Recording date encoded in the episode identifier (``YYYY-MM-DD-...``)."""
    return "-".join(episode.split("-")[:3])


def _dec(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", "replace")
    return str(value)


def parent_action(text):
    """Parent assembly action label: the leading verb, lower-cased.

    REASSEMBLE parent segments are named e.g. ``Pick square peg 4.``, so the
    first whitespace-delimited token is the action (pick / insert / remove /
    place).
    """
    return _dec(text).strip().split()[0].lower()


def object_of(text):
    """Assembly object named in a parent segment: the noun phrase after the verb."""
    txt = _dec(text).strip()
    m = _PARENT_OBJECT_RE.match(txt)
    return m.group(1).strip() if m else txt


def open_episode(data_root, episode):
    return h5py.File(episode_path(data_root, episode), "r")


def parent_segments(h5):
    """Yield ``(index, text)`` for parent segments that carry low-level labels."""
    info = h5["segments_info"]
    for k in info.keys():
        g = info[k]
        if "low_level" not in g:
            continue
        yield int(k), _dec(g["text"][()])


def low_level_segments(h5, k):
    """Yield ``(text, start, end)`` for the child segments of parent ``k``."""
    for j in h5["segments_info"][k]["low_level"].keys():
        node = h5["segments_info"][k]["low_level"][j]
        yield (_dec(node["text"][()]).strip(),
               float(node["start"][()]),
               float(node["end"][()]))


def annotated_frames(data_root, episodes, window, stride):
    """Enumerate every annotated frame of ``episodes``.

    A low-level segment contributes one frame every ``stride`` seconds starting
    ``window`` seconds after its onset, because that is the earliest time at
    which a full causal window lies inside the segment.

    Returns a dict of parallel arrays plus the sorted skill vocabulary.
    """
    ep_col, seg_col, t_col, skill_col, parent_col, date_col = [], [], [], [], [], []
    for ep in episodes:
        date = episode_date(ep)
        with open_episode(data_root, ep) as h5:
            info = h5["segments_info"]
            for k in info.keys():
                g = info[k]
                if "low_level" not in g:
                    continue
                pact = parent_action(g["text"][()])
                for j in g["low_level"].keys():
                    node = g["low_level"][j]
                    s = float(node["start"][()])
                    e = float(node["end"][()])
                    skill = _dec(node["text"][()]).strip()
                    if e - s < window + 1e-6:
                        continue
                    for t in np.arange(s + window, e + 1e-9, stride):
                        ep_col.append(ep)
                        seg_col.append(k)
                        t_col.append(float(t))
                        skill_col.append(skill)
                        parent_col.append(pact)
                        date_col.append(date)

    classes = sorted(set(skill_col))
    label = np.array([classes.index(x) for x in skill_col])
    return dict(
        ep=np.array(ep_col),
        seg=np.array(seg_col, dtype=np.int32),
        t=np.array(t_col),
        y=label.astype(np.int16),
        skill=np.array(skill_col),
        pact=np.array(parent_col),
        date=np.array(date_col),
        classes=np.array(classes),
        n=len(t_col),
        stride=stride,
        window=window,
    )


def object_table(data_root, episodes):
    """Map ``(episode, parent segment)`` to the assembly object it manipulates."""
    table = {}
    for ep in episodes:
        with open_episode(data_root, ep) as h5:
            for k, text in parent_segments(h5):
                table[(ep, k)] = object_of(text)
    return table
