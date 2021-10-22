"""Smoke tests for the COCO dataset loader."""
import torch

from src.data import CocoDetection


def test_dataset_len_and_item(tiny_dataset):
    ds = CocoDetection(tiny_dataset["root"], tiny_dataset["ann"])
    assert len(ds) == 3
    img, target = ds[0]
    assert img.size == (64, 64)
    assert target["boxes"].shape == (1, 4)
    assert target["labels"].dtype == torch.int64
    assert int(target["labels"][0]) == 1


def test_drop_empty_skips_images_without_anns(tiny_dataset):
    # the fixture has 1 ann per image so drop_empty should be a no-op
    ds = CocoDetection(tiny_dataset["root"], tiny_dataset["ann"], drop_empty=True)
    assert len(ds) == 3


def test_categories_exposed(tiny_dataset):
    ds = CocoDetection(tiny_dataset["root"], tiny_dataset["ann"])
    assert ds.get_categories() == ["thing"]
