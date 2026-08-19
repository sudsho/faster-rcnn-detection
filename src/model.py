"""Faster R-CNN model construction.

Wraps torchvision's reference detection model. We swap the box predictor
head to match `num_classes` (background counts as one class, so a 20-class
dataset uses num_classes=21).
"""
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(num_classes, pretrained=True, trainable_backbone_layers=3):
    """Build Faster R-CNN with a fresh box predictor sized to num_classes.

    pretrained=True downloads COCO detection weights plus the ImageNet
    backbone (needs network access). pretrained=False builds the whole
    model from random init and touches the network for nothing, which is
    what the offline tiny-CPU smoke and the unit tests rely on.
    """
    if pretrained:
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        weights_backbone = torchvision.models.ResNet50_Weights.DEFAULT
    else:
        weights = None
        weights_backbone = None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=weights_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
