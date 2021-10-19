# Architecture notes

## Two-stage detector

Faster R-CNN is the original two-stage detector and still a strong baseline.
There are two networks that share a backbone:

1. Region Proposal Network (RPN). At each spatial location of the FPN
   features it predicts an objectness score and a 4-d offset for a set of
   anchors. ROI proposals are NMS-filtered.
2. Box head. For each surviving proposal it pools a fixed-size feature
   (RoIAlign, 7x7), then predicts class probabilities and a per-class
   refinement of the bbox.

We use the torchvision implementation
(`torchvision.models.detection.fasterrcnn_resnet50_fpn`) with COCO-pretrained
weights, then swap the box predictor head to match `num_classes`. The RPN
keeps its weights and tends to converge fast on a new dataset.

## Backbone: ResNet50-FPN

`fasterrcnn_resnet50_fpn` builds:
- ResNet50 trunk
- Feature Pyramid Network on top of conv2..conv5 outputs (P2..P5) with a
  P6 pooled level for very large objects.
- 256-d feature maps at every level.

`trainable_backbone_layers=3` freezes the first stages and the BN running
stats, which reduces VRAM and stabilises early training when the new head
is randomly initialised.

## Loss

Box head: cross entropy + smooth L1 (per-class offsets).
RPN: BCE on objectness + smooth L1 on bbox offsets.
torchvision returns the four scalars in a dict; we sum them and call
`.backward()`.

## Augmentations

albumentations is convenient because it understands bounding boxes natively.
The pipeline stays simple on purpose:
- LongestMaxSize + PadIfNeeded keeps aspect ratio
- horizontal flip
- light brightness/contrast jitter
- occasional motion blur
- ToTensorV2

Stronger augs (mosaic, mixup, cutmix) often hurt for two-stage detectors
in our experience and are skipped here.

## Eval

pycocotools is the COCO-style metric. We feed it the standard predictions
list `[{image_id, category_id, bbox=[x,y,w,h], score}, ...]` and read
`coco_eval.stats[0]` for mAP@0.5:0.95 and `stats[1]` for mAP@0.5.

## Serving

FastAPI app exposes:
- `GET  /health` -- returns model status
- `GET  /classes` -- returns the class list from the active config
- `POST /predict` -- multipart upload, JSON detections back
- `POST /predict_image` -- multipart upload, annotated PNG back

The model loads once at startup. Threshold can be passed per-request as a
query param.
