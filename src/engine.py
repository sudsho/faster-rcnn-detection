"""Per-iteration training/eval helpers (loosely modelled on torchvision's
detection reference engine).
"""
import math
import time

import torch

from .utils import logger, to_device


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1.0
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=50, scaler=None):
    model.train()
    lr_scheduler = None
    if epoch == 0:
        lr_scheduler = warmup_lr_scheduler(optimizer, min(1000, len(data_loader) - 1), 0.001)

    running = {"loss": 0.0, "n": 0}
    start = time.time()
    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [to_device(t, device) for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        if not math.isfinite(loss.item()):
            logger.warning("non-finite loss, skipping iter %d", i)
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        running["loss"] += loss.item()
        running["n"] += 1
        if i % print_freq == 0:
            avg = running["loss"] / max(running["n"], 1)
            logger.info("epoch %d iter %d/%d loss %.4f avg %.4f", epoch, i, len(data_loader), loss.item(), avg)
    elapsed = time.time() - start
    avg = running["loss"] / max(running["n"], 1)
    logger.info("epoch %d done in %.1fs avg loss %.4f", epoch, elapsed, avg)
    return avg
