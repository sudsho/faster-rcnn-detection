"""FastAPI app for serving Faster R-CNN inference."""
from fastapi import FastAPI

app = FastAPI(title="faster-rcnn-detection")


@app.get("/health")
def health():
    return {"status": "ok"}
