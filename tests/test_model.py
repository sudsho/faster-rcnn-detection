"""Smoke tests for build_model and its output structure."""
import pytest
import torch

from src.model import build_model


@pytest.mark.parametrize("num_classes", [2, 5, 21])
def test_build_model_swaps_predictor(num_classes):
    model = build_model(num_classes, pretrained=False)
    # final cls scores should be sized (num_classes,)
    assert model.roi_heads.box_predictor.cls_score.out_features == num_classes


def test_model_output_structure_eval_mode():
    model = build_model(num_classes=3, pretrained=False)
    model.eval()
    img = torch.zeros(3, 200, 200)
    with torch.no_grad():
        out = model([img])
    assert isinstance(out, list)
    assert len(out) == 1
    keys = set(out[0].keys())
    assert {"boxes", "labels", "scores"} <= keys


def test_model_returns_loss_dict_train_mode():
    model = build_model(num_classes=3, pretrained=False)
    model.train()
    img = torch.rand(3, 200, 200)
    target = {
        "boxes": torch.tensor([[10.0, 10.0, 60.0, 60.0]]),
        "labels": torch.tensor([1], dtype=torch.int64),
    }
    losses = model([img], [target])
    assert isinstance(losses, dict)
    assert all(torch.is_tensor(v) for v in losses.values())
    assert "loss_classifier" in losses
