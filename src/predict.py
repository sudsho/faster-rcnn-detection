"""Single-image inference."""
import argparse

import torch
import yaml
from PIL import Image
import torchvision.transforms.functional as F

from .model import build_model
from .utils import device_from_cfg
from .visualize import draw_boxes


def load_model(cfg, weights_path, device):
    model = build_model(cfg["model"]["num_classes"], pretrained=False)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def predict_image(model, image_path, device, score_thresh=0.5):
    img = Image.open(image_path).convert("RGB")
    tensor = F.to_tensor(img).to(device)
    with torch.no_grad():
        out = model([tensor])[0]
    keep = out["scores"] >= score_thresh
    return {
        "boxes": out["boxes"][keep].cpu().tolist(),
        "labels": out["labels"][keep].cpu().tolist(),
        "scores": out["scores"][keep].cpu().tolist(),
    }, img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/voc.yaml")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--save", default=None, help="path to save annotated image")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = device_from_cfg(cfg)
    model = load_model(cfg, args.weights, device)
    result, img = predict_image(model, args.image, device, args.score_thresh)
    print(result)
    if args.save:
        classes = cfg.get("dataset", {}).get("classes")
        out_img = draw_boxes(img, result["boxes"], result["labels"], result["scores"],
                             classes=classes, score_thresh=args.score_thresh)
        out_img.save(args.save)
        print("saved", args.save)


if __name__ == "__main__":
    main()
