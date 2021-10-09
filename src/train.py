"""Training entry point for Faster R-CNN.

Loads config, builds model + dataset, runs the training loop and saves
the best checkpoint.
"""
import argparse
import os

import torch
import yaml

from .data import CocoDetection
from .model import build_model
from .transforms import AlbumentationsAdapter
from .utils import collate_fn, ensure_dir, set_seed, device_from_cfg, to_device


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

    train_ds = CocoDetection(
        cfg["dataset"]["root"],
        os.path.join(cfg["dataset"]["root"], "annotations", "train.json"),
        transforms=AlbumentationsAdapter(train=True),
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn,
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

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for i, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [to_device(t, device) for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % cfg["train"]["print_freq"] == 0:
                print(f"epoch {epoch} iter {i} loss {loss.item():.4f}")
        lr_scheduler.step()
        ckpt_path = os.path.join(cfg["paths"]["ckpt_dir"], f"epoch_{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)


if __name__ == "__main__":
    main()
