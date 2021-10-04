# faster-rcnn-detection

Object detection on a custom dataset using Faster R-CNN with a ResNet50-FPN
backbone (torchvision). Trains on Pascal VOC by default; the loader also
supports any COCO-format custom dataset.

## What is in here

- `src/data.py` COCO-format dataset wrapper plus albumentations augmentations
- `src/model.py` builds Faster R-CNN with a swappable box predictor head
- `src/train.py` training loop on top of the torchvision detection reference
- `src/eval.py` mAP@0.5 and mAP@0.5:0.95 via pycocotools
- `src/predict.py` single-image inference, returns boxes + labels + scores
- `src/visualize.py` draws boxes with class name and score on the image
- `src/api/main.py` FastAPI service with `/predict` (JSON) and `/predict_image` (PNG)
- `configs/` YAML configs for VOC, OpenImages and a small default
- `tests/` smoke tests (data, model output, drawing, API)

## Status

WIP. README will fill out as the pipeline lands.
