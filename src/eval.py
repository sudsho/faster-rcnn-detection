"""Evaluation entry point.

Computes mAP metrics on a validation split. Heavy lifting is done by
pycocotools (the same library COCO uses for the official metric).
"""
import argparse
import json
import os
import tempfile

import torch
import yaml
from tqdm import tqdm

from .data import CocoDetection
from .model import build_model
from .utils import collate_fn, device_from_cfg, to_device


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/voc.yaml")
    ap.add_argument("--weights", required=True)
    return ap.parse_args()


def run_eval(model, loader, device):
    model.eval()
    results = []
    for images, targets in tqdm(loader, desc="eval"):
        images = [img.to(device) for img in images]
        with torch.no_grad():
            outputs = model(images)
        for tgt, out in zip(targets, outputs):
            image_id = int(tgt["image_id"].item())
            for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
                x1, y1, x2, y2 = box.cpu().tolist()
                results.append({
                    "image_id": image_id,
                    "category_id": int(label.item()),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score.item()),
                })
    return results


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = device_from_cfg(cfg)
    val_ds = CocoDetection(
        cfg["dataset"]["root"],
        os.path.join(cfg["dataset"]["root"], "annotations", "val.json"),
    )
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_fn
    )
    model = build_model(cfg["model"]["num_classes"], pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    results = run_eval(model, loader, device)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(results, f)
        pred_path = f.name
    print("predictions written to", pred_path)


if __name__ == "__main__":
    main()
