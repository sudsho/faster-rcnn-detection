"""FastAPI app for serving Faster R-CNN inference.

Endpoints:
    GET  /health         liveness probe
    POST /predict        upload an image, get JSON detections
    POST /predict_image  upload an image, get a PNG with boxes drawn
"""
import io
import os

import torch
import torchvision.transforms.functional as F
import yaml
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from ..model import build_model
from ..utils import device_from_cfg
from ..visualize import draw_boxes
from .schemas import PredictResponse


CFG_PATH = os.environ.get("FRCNN_CONFIG", "configs/voc.yaml")
WEIGHTS = os.environ.get("FRCNN_WEIGHTS", "checkpoints/best.pt")


app = FastAPI(title="faster-rcnn-detection")
_state = {"model": None, "cfg": None, "device": None, "classes": None}


@app.on_event("startup")
def _load():
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    device = device_from_cfg(cfg)
    model = build_model(cfg["model"]["num_classes"], pretrained=False)
    if os.path.exists(WEIGHTS):
        model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.to(device).eval()
    _state["model"] = model
    _state["cfg"] = cfg
    _state["device"] = device
    _state["classes"] = cfg.get("dataset", {}).get("classes")


@app.get("/health")
def health():
    if _state["model"] is None:
        return {"status": "starting"}
    return {"status": "ok", "device": str(_state["device"]), "num_classes": _state["cfg"]["model"]["num_classes"]}


@app.get("/classes")
def classes():
    return {"classes": _state["classes"] or []}


def _detect(img):
    tensor = F.to_tensor(img).to(_state["device"])
    with torch.no_grad():
        out = _state["model"]([tensor])[0]
    return out


def _to_dets(out, score_thresh, classes):
    items = []
    for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
        s = float(score.item())
        if s < score_thresh:
            continue
        l = int(label.item())
        name = classes[l - 1] if classes and 1 <= l <= len(classes) else str(l)
        items.append({
            "box": [float(v) for v in box.cpu().tolist()],
            "label": l,
            "label_name": name,
            "score": s,
        })
    return items


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), score_thresh: float = 0.5):
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    out = _detect(img)
    dets = _to_dets(out, score_thresh, _state["classes"])
    return PredictResponse(detections=dets, width=img.width, height=img.height)


@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...), score_thresh: float = 0.5):
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    out = _detect(img)
    drawn = draw_boxes(
        img,
        out["boxes"].cpu().tolist(),
        out["labels"].cpu().tolist(),
        out["scores"].cpu().tolist(),
        classes=_state["classes"],
        score_thresh=score_thresh,
    )
    buf = io.BytesIO()
    drawn.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
