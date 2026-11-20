import pytest
from conftest import box, make_fake_bdd, make_frame

from adapres.data import bdd_mot


def test_parse_basic():
    frames = [make_frame("v1", 1, [box("a", "car", 0, 0, 10, 10), box("b", "trailer", 1, 1, 5, 5)], "night"),
              make_frame("v1", 0, [box("c", "rider", 0, 0, 3, 3, crowd=True)], "night")]
    out = bdd_mot.parse_video_labels(frames)
    assert [f.frame_idx for f in out] == [0, 1]  # sorted
    assert out[0].boxes[0].eval_id == 1 and out[0].boxes[0].crowd
    assert out[1].boxes[1].kind == "ignore"
    assert bdd_mot.sequence_timeofday(out) == "night"


def test_parse_skips_non_box_labels():
    fr = make_frame("v1", 0, [{"id": "p", "category": "car", "poly2d": []}], None)
    assert bdd_mot.parse_video_labels([fr])[0].boxes == []


@pytest.mark.parametrize("bad", [
    [{"videoName": "v", "frameIndex": 0}],                                      # no name
    [make_frame("v", 0, [box("a", "car", 10, 0, 0, 10)], None)],                # inverted box
    [make_frame("v", 0, [], None), make_frame("w", 1, [], None)],              # two videos
    [make_frame("v", 0, [], None), make_frame("v", 0, [], None)],              # duplicate index
    {"not": "a list"},
])
def test_parse_rejects_bad_input(bad):
    with pytest.raises(bdd_mot.LabelFormatError):
        bdd_mot.parse_video_labels(bad)


def test_unknown_category_raises():
    with pytest.raises(KeyError):
        bdd_mot.parse_video_labels([make_frame("v", 0, [box("a", "traffic sign", 0, 0, 1, 1)], None)])


@pytest.mark.parametrize("raw,norm", [("daytime", "daytime"), ("night", "night"), ("dawn/dusk", "dawn_dusk"),
                                      ("undefined", None), (None, None)])
def test_normalise_timeofday(raw, norm):
    assert bdd_mot.normalise_timeofday(raw) == norm


def test_timeofday_majority():
    fr = [make_frame("v", i, [], t) for i, t in enumerate(["night", "night", "daytime"])]
    assert bdd_mot.sequence_timeofday(bdd_mot.parse_video_labels(fr)) == "night"
    assert bdd_mot.sequence_timeofday(bdd_mot.parse_video_labels([make_frame("v", 0, [], None)])) is None


def test_discover_layout(tmp_path):
    make_fake_bdd(tmp_path, {"s1": "daytime", "s2": "night"}, n_frames=3)
    lay = bdd_mot.discover_layout(tmp_path, "track_images_val", "box_track_labels_trainval", "val")
    assert lay.label_dir.name == "val" and lay.n_label_files == 2
    assert lay.image_root.name == "val" and lay.n_image_dirs == 2


def test_discover_layout_fails_loudly(tmp_path):
    (tmp_path / "something_else").mkdir()
    with pytest.raises(bdd_mot.LayoutError, match="track_images_val.*\n?.*Tree seen"):
        bdd_mot.discover_layout(tmp_path, "track_images_val", "box_track_labels_trainval", "val")
