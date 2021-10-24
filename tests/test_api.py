"""End-to-end smoke test for the FastAPI service.

Uses a tiny model (pretrained=False, a couple of classes) so the test
fits comfortably in CI memory.
"""
import io
import os
from unittest import mock

import pytest
from PIL import Image


@pytest.fixture
def client(monkeypatch, tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "device: cpu\n"
        "model:\n  num_classes: 2\n  backbone: resnet50_fpn\n"
        "dataset:\n  classes: [thing]\n"
    )
    monkeypatch.setenv("FRCNN_CONFIG", str(cfg))
    monkeypatch.setenv("FRCNN_WEIGHTS", str(tmp_path / "missing.pt"))

    # delay imports until env is set
    from fastapi.testclient import TestClient
    from src.api.main import app

    with TestClient(app) as c:
        yield c


def _png_bytes(size=(64, 64)):
    img = Image.new("RGB", size, (180, 180, 180))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_classes(client):
    r = client.get("/classes")
    assert r.status_code == 200
    assert "classes" in r.json()


def test_predict_returns_schema(client):
    r = client.post(
        "/predict",
        files={"file": ("a.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200
    body = r.json()
    assert "detections" in body
    assert body["width"] == 64
    assert body["height"] == 64


def test_predict_image_returns_png(client):
    r = client.post(
        "/predict_image",
        files={"file": ("a.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
