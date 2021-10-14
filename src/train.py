"""Training entry point for Faster R-CNN.

Loads config, builds model + dataset, runs the training loop and saves
the best checkpoint.
"""
import argparse
import math
import os

import torch
import yaml

from .coco_eval import evaluate as coco_eval
from .data import CocoDetection
from .engine import train_one_epoch
from .model import build_model
from .transforms import AlbumentationsAdapter
from .utils import collate_fn, ensure_dir, set_seed, device_from_cfg, logger, to_device


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = device_from_cfg(cfg)
    ensure_dir(cfg["paths"]["ckpt_dir"])
    ensure_dir(cfg["paths"]["output_dir"])

    use_mlflow = cfg.get("mlflow", {}).get("enable", False)
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(cfg["mlflow"].get("tracking_uri", "file:./mlruns"))
            mlflow.set_experiment(cfg["mlflow"].get("experiment", "faster-rcnn"))
            mlflow.start_run()
            mlflow.log_params({
                "lr": cfg["train"]["lr"],
                "batch_size": cfg["train"]["batch_size"],
                "backbone": cfg["model"]["backbone"],
                "num_classes": cfg["model"]["num_classes"],
            })
        except ImportError:
            logger.warning("mlflow requested but not installed")
            use_mlflow = False

    train_ds = CocoDetection(
        cfg["dataset"]["root"],
        os.path.join(cfg["dataset"]["root"], "annotations", "train.json"),
        transforms=AlbumentationsAdapter(train=True),
    )
    val_ds = CocoDetection(
        cfg["dataset"]["root"],
        os.path.join(cfg["dataset"]["root"], "annotations", "val.json"),
        transforms=AlbumentationsAdapter(train=False),
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_fn,
    )

    model = build_model(cfg["model"]["num_classes"]).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg["train"]["lr"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["train"]["lr_step_size"],
        gamma=cfg["train"]["lr_gamma"],
    )
    scaler = torch.cuda.amp.GradScaler() if cfg["train"].get("amp") else None

    best_map = -math.inf
    for epoch in range(cfg["train"]["epochs"]):
        train_one_epoch(
            model, optimizer, train_loader, device, epoch,
            print_freq=cfg["train"]["print_freq"], scaler=scaler,
        )
        lr_scheduler.step()
        ckpt_path = os.path.join(cfg["paths"]["ckpt_dir"], f"epoch_{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)

        # validate at the end of each epoch
        model.eval()
        preds = []
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            with torch.no_grad():
                outputs = model(images)
            for tgt, out in zip(targets, outputs):
                image_id = int(tgt["image_id"].item())
                for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
                    x1, y1, x2, y2 = box.cpu().tolist()
                    preds.append({
                        "image_id": image_id,
                        "category_id": int(label.item()),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score.item()),
                    })
        gt_path = os.path.join(cfg["dataset"]["root"], "annotations", "val.json")
        metrics = coco_eval(gt_path, preds)
        m = metrics["mAP@0.5:0.95"]
        logger.info("epoch %d val mAP@.5:.95=%.4f", epoch, m)
        if use_mlflow:
            import mlflow
            mlflow.log_metrics({k: v for k, v in metrics.items()}, step=epoch)
        if m > best_map:
            best_map = m
            best_path = os.path.join(cfg["paths"]["ckpt_dir"], "best.pt")
            torch.save(model.state_dict(), best_path)
            logger.info("new best %.4f, saved %s", m, best_path)

    if use_mlflow:
        import mlflow
        mlflow.end_run()


if __name__ == "__main__":
    main()
