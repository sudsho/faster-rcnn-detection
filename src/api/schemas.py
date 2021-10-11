"""Pydantic models for the FastAPI service."""
from typing import List
from pydantic import BaseModel, Field


class Detection(BaseModel):
    box: List[float] = Field(..., description="x1, y1, x2, y2")
    label: int
    label_name: str
    score: float


class PredictResponse(BaseModel):
    detections: List[Detection]
    width: int
    height: int


class HealthResponse(BaseModel):
    status: str
    device: str = "unknown"
    num_classes: int = 0
