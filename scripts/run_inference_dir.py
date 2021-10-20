"""Run inference over a directory of images and dump JSON + annotated images."""
import argparse
import json
import os

import torch
import yaml
from PIL import Image

from src.api.inference import Inferencer
from src.visualize import draw_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--score-thresh", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    inferencer = Inferencer(args.config, args.weights)

    json_out = {}
    for fname in sorted(os.listdir(args.input_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        path = os.path.join(args.input_dir, fname)
        img = Image.open(path).convert("RGB")
        result = inferencer.detect(img, args.score_thresh)
        json_out[fname] = result["detections"]
        out = result["raw"]
        drawn = draw_boxes(
            img,
            out["boxes"].cpu().tolist(),
            out["labels"].cpu().tolist(),
            out["scores"].cpu().tolist(),
            classes=inferencer.classes,
            score_thresh=args.score_thresh,
        )
        drawn.save(os.path.join(args.output_dir, fname))

    with open(os.path.join(args.output_dir, "detections.json"), "w") as f:
        json.dump(json_out, f, indent=2)
    print("done. processed", len(json_out), "images")


if __name__ == "__main__":
    main()
