"""FastAPI app for serving Faster R-CNN inference.

Endpoints:
    GET  /health         liveness probe
    POST /predict        upload an image, get JSON detections
    POST /predict_image  upload an image, get a PNG with boxes drawn
"""
import io
import os

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from ..visualize import draw_boxes
from .inference import Inferencer
from .schemas import PredictResponse


CFG_PATH = os.environ.get("FRCNN_CONFIG", "configs/voc.yaml")
WEIGHTS = os.environ.get("FRCNN_WEIGHTS", "checkpoints/best.pt")


app = FastAPI(title="faster-rcnn-detection")
_state = {"infer": None}


@app.on_event("startup")
def _load():
    _state["infer"] = Inferencer(CFG_PATH, WEIGHTS)


@app.get("/health")
def health():
    inf = _state["infer"]
    if inf is None:
        return {"status": "starting"}
    return {"status": "ok", "device": str(inf.device), "num_classes": inf.cfg["model"]["num_classes"]}


@app.get("/classes")
def classes():
    inf = _state["infer"]
    return {"classes": (inf.classes if inf else None) or []}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), score_thresh: float = 0.5):
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    result = _state["infer"].detect(img, score_thresh)
    return PredictResponse(detections=result["detections"], width=img.width, height=img.height)


@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...), score_thresh: float = 0.5):
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    result = _state["infer"].detect(img, score_thresh)
    out = result["raw"]
    drawn = draw_boxes(
        img,
        out["boxes"].cpu().tolist(),
        out["labels"].cpu().tolist(),
        out["scores"].cpu().tolist(),
        classes=_state["infer"].classes,
        score_thresh=score_thresh,
    )
    buf = io.BytesIO()
    drawn.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
