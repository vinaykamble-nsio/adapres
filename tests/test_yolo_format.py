import yaml
from conftest import box, make_frame

from adapres.data.bdd_mot import parse_video_labels
from adapres.data.yolo_format import flat_name, write_yolo_dataset, yolo_label_lines


def frame():
    return parse_video_labels([make_frame("v", 0, [
        box("a", "truck", 0, 0, 100, 50), box("b", "rider", 90, 40, 110, 60),
        box("i", "trailer", 1, 1, 9, 9), box("c", "car", 5, 5, 15, 15, crowd=True)], None)])[0]


def test_label_lines():
    lines = yolo_label_lines(frame(), 100, 50)
    # truck -> COCO 7; rider -> person 0, clipped to x<=100, y<=50; ignore + crowd omitted
    assert lines == ["7 0.500000 0.500000 1.000000 1.000000", "0 0.950000 0.900000 0.100000 0.200000"]


def test_write_dataset(tmp_path):
    img_root = tmp_path / "imgs"
    (img_root / "v").mkdir(parents=True)
    (img_root / "v" / "v-0000001.jpg").write_bytes(b"x")
    y = write_yolo_dataset(tmp_path / "ds", [("v/v-0000001.jpg", frame(), 100, 50)], img_root, {0: "person"}, copy=True)
    data = yaml.safe_load(y.read_text())
    assert data["val"] == "images" and data["names"] == {0: "person"}
    name = flat_name("v/v-0000001.jpg")
    assert (tmp_path / "ds" / "images" / name).read_bytes() == b"x"
    assert (tmp_path / "ds" / "labels" / "v__v-0000001.txt").read_text().startswith("7 ")
