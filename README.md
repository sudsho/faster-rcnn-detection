# faster-rcnn-detection

Object detection on a custom dataset using Faster R-CNN with a ResNet50-FPN
backbone (torchvision). Configured for Pascal VOC 2007 by default; the
loader also reads any COCO-format annotation set (Open Images, Roboflow
exports, your own labels).

## Quick start (tiny-CPU smoke, no GPU/download)

Prove the train and inference plumbing end to end in a couple of minutes on
a laptop CPU, with no dataset, no checkpoints and no network access. The
smoke builds the torchvision Faster R-CNN from random init (`pretrained=False`,
so no COCO or ImageNet weights are downloaded), synthesizes a few 128x128
images by drawing colored rectangles and recording their boxes, takes a few
SGD steps so the training loss falls, then runs eval-mode inference and
checks the output structure.

```bash
python scripts/smoke.py     # or: make smoke
```

Real output from a CPU run (torchvision 0.20.1, Python 3.11):

```
torch 2.5.1+cu121  device=cpu
synthetic batch: 4 images 128x128, 7 boxes, classes [1, 2]
loss components: {'loss_classifier': 1.3989, 'loss_box_reg': 0.0971, 'loss_objectness': 0.7153, 'loss_rpn_box_reg': 0.0223}
  step  0  total_loss 2.2337
  step  5  total_loss 0.7586
  step 10  total_loss 0.4393
  step 15  total_loss 0.4139
  step 20  total_loss 0.3147
  step 25  total_loss 0.2934
  step 29  total_loss 0.2377
train-mode loss: 2.2337 -> 0.2377
eval-mode inference: 0 raw detections, boxes(0, 4) labels(0,) scores(0,)
SMOKE OK
```

(Zero raw detections is expected: a randomly initialized model trained for
30 steps has no confident proposals to keep after NMS; what the smoke checks
is that the loss decreases and the eval output has correctly shaped
`boxes`/`labels`/`scores` tensors.)

The unit tests exercise the same code paths (model head swap, dataset loader,
box drawing, FastAPI endpoints) and run in seconds:

```bash
python -m pytest -q     # 16 passed
```

This smoke only proves the code runs. The real headline (Pascal VOC / COCO
mAP) needs a GPU and a real dataset plus the COCO/ImageNet pretrained
weights that `build_model(pretrained=True)` downloads; those paths are
guarded so the smoke never touches the network.

## Problem

Given an image, draw a tight bounding box around each object of interest and
predict its class. The model is the classic two-stage Faster R-CNN: an FPN
on top of a ResNet50 backbone produces multi-scale features, an RPN proposes
regions, and a box head classifies each proposal and refines its
coordinates. We swap torchvision's default 91-class COCO predictor for a
fresh head sized to the target dataset.

## Dataset

Default: **Pascal VOC 2007** (20 classes + background).

To use a different dataset, point `dataset.root` and the class list in a new
config to a folder containing your images plus a COCO-format annotation
json under `annotations/train.json` and `annotations/val.json`. A recipe
for converting an Open Images class-export (via the OIDv4 toolkit) lives at
`scripts/oi_to_coco.py`. A Roboflow export "COCO JSON" works as-is.

## Pipeline

```
voc_to_coco.py / oi_to_coco.py  ->  data/<ds>/annotations/{train,val}.json
                                              |
                              CocoDetection (src/data.py)
                                              |
                          AlbumentationsAdapter augments
                                              |
              torchvision fasterrcnn_resnet50_fpn (src/model.py)
                                              |
                  train_one_epoch + StepLR (src/engine.py)
                                              |
                pycocotools mAP per-epoch (src/coco_eval.py)
                                              |
                  best.pt -> FastAPI inference (src/api/main.py)
```

## Quickstart

The training loop expects a COCO-format `annotations/train.json` and
`annotations/val.json` under `dataset.root`. The VOC converter and split
helper under `scripts/` / `src/voc_to_coco.py` produce a single COCO json
per VOC split, and `scripts/make_voc_splits.py` filters that into
train/val subsets; you may need to rename or copy the outputs to match the
hardcoded names above. Once the annotations are in place:

```bash
pip install -r requirements.txt
python -m src.train  --config configs/voc.yaml
python -m src.predict --config configs/voc.yaml --weights checkpoints/voc/best.pt \
    --image samples/dog.jpg --save out.jpg
```

## Serving

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# JSON detections
curl -F "file=@samples/dog.jpg" http://localhost:8000/predict
# Annotated PNG
curl -F "file=@samples/dog.jpg" http://localhost:8000/predict_image -o out.png
```

`/predict` returns `{detections: [{box, label, label_name, score}, ...], width, height}`.
`/predict_image` returns the input image with boxes drawn on it.

A `Dockerfile` and `docker-compose.yml` are included so the service plus an
MLflow tracking server can come up with `docker-compose up`.

## Results

Not benchmarked in this repo. The training loop prints per-epoch mAP via
pycocotools, but predictions are collected in the resized tensor frame
while the ground-truth json is in original pixel coordinates, so the
printed mAP is not directly comparable to published VOC/COCO numbers. Use
it only as a relative signal across epochs of the same run.

## Layout

- `src/data.py`        COCO-format dataset wrapper
- `src/transforms.py`  albumentations augmentations
- `src/model.py`       builds Faster R-CNN with a swappable box predictor head
- `src/engine.py`      train_one_epoch with warmup
- `src/train.py`       training loop, mlflow logging, best-checkpoint saving
- `src/predict.py`     single-image inference, optional annotated save
- `src/visualize.py`   draws boxes with class name + score
- `src/api/main.py`    FastAPI `/predict` (JSON) and `/predict_image` (PNG)
- `configs/`           voc.yaml, openimages.yaml, coco_subset.yaml
- `tests/`             smoke tests for data, model output shape, drawing, API
- `notebooks/eda.ipynb`  class distribution, image dim histograms, sample boxes
- `scripts/`           voc download, voc/oi -> coco conversion, split helpers
- `Dockerfile`, `docker-compose.yml`, `ci/test.yml.example`, `Makefile`

## License

MIT.
