"""Inference helper for the FastAPI app.

Wraps model loading + a `detect()` callable so main.py only deals with the
HTTP layer.
"""
import os
from typing import Any, Dict, List, Optional

import torch
import torchvision.transforms.functional as F
import yaml
from PIL import Image

from ..model import build_model
from ..utils import device_from_cfg, logger


class Inferencer:
    def __init__(self, cfg_path: str, weights_path: Optional[str] = None):
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)
        self.device = device_from_cfg(self.cfg)
        self.model = build_model(self.cfg["model"]["num_classes"], pretrained=False)
        if weights_path and os.path.exists(weights_path):
            logger.info("loading weights from %s", weights_path)
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        else:
            logger.warning("no weights found at %s, running with random init", weights_path)
        self.model.to(self.device).eval()
        self.classes = self.cfg.get("dataset", {}).get("classes")

    def detect(self, img: Image.Image, score_thresh: float = 0.5) -> Dict[str, Any]:
        tensor = F.to_tensor(img).to(self.device)
        with torch.no_grad():
            out = self.model([tensor])[0]
        return self._format(out, score_thresh)

    def _format(self, out, score_thresh) -> Dict[str, Any]:
        items: List[Dict[str, Any]] = []
        for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
            s = float(score.item())
            if s < score_thresh:
                continue
            l = int(label.item())
            name = self.classes[l - 1] if self.classes and 1 <= l <= len(self.classes) else str(l)
            items.append({
                "box": [float(v) for v in box.cpu().tolist()],
                "label": l,
                "label_name": name,
                "score": s,
            })
        return {"detections": items, "raw": out}
