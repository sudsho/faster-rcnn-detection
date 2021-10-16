# faster-rcnn-detection

Object detection on a custom dataset using Faster R-CNN with a ResNet50-FPN
backbone (torchvision). Trains on Pascal VOC by default; the loader also
supports any COCO-format custom dataset (Open Images, Roboflow exports, your
own labels).

## Problem

Given an image, draw a tight bounding box around each object of interest and
predict its class. The model is the classic two-stage Faster R-CNN: an FPN
on top of an ImageNet-pretrained ResNet50 produces multi-scale features, an
RPN proposes regions, and a box head classifies each proposal and refines
its coordinates. We swap torchvision's default 91-class COCO predictor for a
fresh head sized to the target dataset.

## Dataset

Default: **Pascal VOC 2007 + 2012** (20 classes + background).

To use a different dataset, point the `dataset.root` and class list in a new
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

```bash
pip install -r requirements.txt
bash scripts/download_voc.sh data/voc
python -m src.voc_to_coco --voc-root data/voc/VOCdevkit/VOC2007 --split trainval \
    --out data/voc/annotations/voc2007_trainval.json
python -m src.train  --config configs/voc.yaml
python -m src.eval   --config configs/voc.yaml --weights checkpoints/voc/best.pt
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

Numbers from a 12-epoch run on VOC 07+12 trainval, evaluated on VOC 2007
test:

| metric        | value  |
|---------------|--------|
| mAP @ 0.5     | 0.764  |
| mAP @ 0.5:0.95| 0.451  |

(Will rerun once the loader changes settle.)

## Layout

- `src/data.py`        COCO-format dataset wrapper
- `src/transforms.py`  albumentations augmentations
- `src/model.py`       builds Faster R-CNN with a swappable box predictor head
- `src/engine.py`      train_one_epoch with warmup
- `src/train.py`       training loop, mlflow logging, best-checkpoint saving
- `src/eval.py`        mAP@0.5 and mAP@0.5:0.95 via pycocotools
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
