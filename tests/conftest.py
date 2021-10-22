"""Shared fixtures for tests.

Builds a tiny synthetic COCO-format dataset on-disk so the data loader
test does not need real image files.
"""
import json
import os

import pytest
from PIL import Image


@pytest.fixture
def tiny_dataset(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    images = []
    annotations = []
    ann_id = 1
    for i in range(3):
        path = img_dir / f"im_{i}.jpg"
        Image.new("RGB", (64, 64), (i * 30, 100, 200 - i * 20)).save(path)
        images.append({
            "id": i + 1,
            "file_name": f"images/im_{i}.jpg",
            "width": 64,
            "height": 64,
        })
        annotations.append({
            "id": ann_id,
            "image_id": i + 1,
            "category_id": 1,
            "bbox": [10.0, 10.0, 30.0, 30.0],
            "area": 900.0,
            "iscrowd": 0,
        })
        ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "thing"}],
    }
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    ann_file = ann_dir / "tiny.json"
    with open(ann_file, "w") as f:
        json.dump(coco, f)
    return {"root": str(tmp_path), "ann": str(ann_file)}
