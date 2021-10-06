"""Faster R-CNN model construction.

Wraps torchvision's reference detection model. We swap the box predictor
head to match `num_classes` (background counts as one class, so a 20-class
dataset uses num_classes=21).
"""
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(num_classes, pretrained=True, trainable_backbone_layers=3):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=pretrained,
        trainable_backbone_layers=trainable_backbone_layers,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
