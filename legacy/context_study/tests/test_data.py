import h5py

from crps import data


def _write_episode(path, segments):
    """segments: list of (parent_text, [(skill_text, start, end), ...])."""
    with h5py.File(path, "w") as fh:
        si = fh.create_group("segments_info")
        for i, (parent_text, children) in enumerate(segments):
            g = si.create_group(str(i))
            g.create_dataset("text", data=parent_text)
            ll = g.create_group("low_level")
            for j, (skill, s, e) in enumerate(children):
                c = ll.create_group(str(j))
                c.create_dataset("text", data=skill)
                c.create_dataset("start", data=float(s))
                c.create_dataset("end", data=float(e))


def test_parent_action_and_object_parsing():
    assert data.parent_action("Pick square peg 4.") == "pick"
    assert data.parent_action(b"Insert BNC.") == "insert"
    assert data.object_of("Pick square peg 4.") == "square peg 4"
    assert data.object_of("Insert BNC.") == "BNC"
    assert data.object_of("Remove large gear") == "large gear"


def test_episode_date():
    assert data.episode_date("2025-01-13-09-43-29") == "2025-01-13"


def test_read_split(tmp_path):
    p = tmp_path / "train_split1.txt"
    p.write_text("a-1\n\nb-2\n")
    assert data.read_split(str(p)) == ["a-1", "b-2"]


def test_annotated_frames_grid(tmp_path):
    root = tmp_path
    (root / "data" / "data").mkdir(parents=True)
    _write_episode(str(root / "data" / "data" / "ep-1.h5"), [
        ("Pick square peg 4.", [("Approach", 0.0, 2.0), ("Grasp", 2.0, 3.0)]),
        ("Place square peg 4.", [("Release", 3.0, 3.2)]),   # shorter than the window
    ])
    frames = data.annotated_frames(str(root), ["ep-1"], window=0.4, stride=0.2)
    # segment 0: t = 0.4 .. 2.0 -> 9 frames; segment 1: t = 2.4 .. 3.0 -> 4 frames
    # the third child is only 0.2 s long and cannot host a 0.4 s causal window,
    # so the "place" parent contributes no frames at all
    assert frames["n"] == 13
    assert frames["classes"].tolist() == ["Approach", "Grasp"]
    assert frames["t"].min() == 0.4
    assert set(frames["pact"].tolist()) == {"pick"}
    assert frames["date"][0] == "ep-1"  # short id, date parsing is pass-through
    assert int((frames["y"] == 0).sum()) == 9


def test_annotated_frames_uses_episode_date(tmp_path):
    root = tmp_path
    (root / "data" / "data").mkdir(parents=True)
    _write_episode(str(root / "data" / "data" / "2025-01-13-09-43-29.h5"),
                   [("Pick BNC.", [("Grasp", 0.0, 1.0)])])
    frames = data.annotated_frames(str(root), ["2025-01-13-09-43-29"], 0.4, 0.2)
    assert frames["date"][0] == "2025-01-13"


def test_segments_without_low_level_are_skipped(tmp_path):
    root = tmp_path
    (root / "data" / "data").mkdir(parents=True)
    p = str(root / "data" / "data" / "ep-1.h5")
    with h5py.File(p, "w") as fh:
        si = fh.create_group("segments_info")
        g = si.create_group("0")
        g.create_dataset("text", data="Pick BNC.")  # no low_level group
        g2 = si.create_group("1")
        g2.create_dataset("text", data="Place BNC.")
        ll = g2.create_group("low_level")
        c = ll.create_group("0")
        c.create_dataset("text", data="Release")
        c.create_dataset("start", data=0.0)
        c.create_dataset("end", data=1.0)
    frames = data.annotated_frames(str(root), ["ep-1"], 0.4, 0.2)
    assert frames["classes"].tolist() == ["Release"]
    assert set(frames["seg"].tolist()) == {1}


def test_object_table(tmp_path):
    root = tmp_path
    (root / "data" / "data").mkdir(parents=True)
    _write_episode(str(root / "data" / "data" / "ep-1.h5"),
                   [("Pick square peg 4.", [("Grasp", 0.0, 1.0)])])
    table = data.object_table(str(root), ["ep-1"])
    assert table[("ep-1", 0)] == "square peg 4"
