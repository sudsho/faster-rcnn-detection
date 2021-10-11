"""Small misc helpers shared across modules."""
import logging
import os
import random
import numpy as np
import torch


logger = logging.getLogger("faster_rcnn")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


def collate_fn(batch):
    return tuple(zip(*batch))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_cfg(cfg):
    want = cfg.get("device", "cuda")
    if want == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(want)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(x, device) for x in obj)
    return obj
