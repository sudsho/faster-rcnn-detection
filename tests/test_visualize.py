"""Tests for the box-drawing helper."""
from PIL import Image

from src.visualize import draw_boxes


def test_draw_boxes_returns_same_size_image():
    img = Image.new("RGB", (200, 100), (200, 200, 200))
    out = draw_boxes(
        img,
        boxes=[[10, 10, 80, 80]],
        labels=[1],
        scores=[0.9],
        classes=["x"],
    )
    assert out.size == img.size
    assert out is not img  # we copied


def test_draw_boxes_skips_below_threshold():
    img = Image.new("RGB", (50, 50))
    # 0.1 score, threshold 0.5 -> nothing should be drawn but call must not crash
    out = draw_boxes(img, [[5, 5, 40, 40]], [1], [0.1], score_thresh=0.5)
    assert out.size == img.size


def test_draw_boxes_handles_label_without_classes():
    img = Image.new("RGB", (60, 60))
    # no classes provided -> label should be stringified
    out = draw_boxes(img, [[5, 5, 50, 50]], [3], [0.95])
    assert out is not None


def test_edge_box_at_top_does_not_throw():
    img = Image.new("RGB", (100, 100))
    # box right at y=0 used to clip negative -- should still work
    out = draw_boxes(img, [[5, 0, 80, 50]], [1], [0.9], classes=["x"])
    assert out.size == img.size
