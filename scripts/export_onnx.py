"""Export a trained checkpoint to ONNX (best-effort).

torchvision's Faster R-CNN ONNX export has caveats around dynamic shapes
and the post-processing ops, but a fixed-input shape exports cleanly enough
for downstream serving stacks.
"""
import argparse

import torch
import yaml

from src.model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--width", type=int, default=800)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model = build_model(cfg["model"]["num_classes"], pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    dummy = torch.zeros(3, args.height, args.width)
    torch.onnx.export(
        model,
        ([dummy],),
        args.out,
        opset_version=11,
        input_names=["images"],
        output_names=["boxes", "labels", "scores"],
        do_constant_folding=True,
    )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
