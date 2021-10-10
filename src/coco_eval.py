"""Wrapper around pycocotools to compute mAP@0.5, mAP@0.5:0.95, etc."""
import json
import tempfile

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate(gt_json, predictions):
    """predictions: list of {"image_id", "category_id", "bbox" [x,y,w,h], "score"}."""
    coco_gt = COCO(gt_json)
    if not predictions:
        return {"mAP@0.5:0.95": 0.0, "mAP@0.5": 0.0, "mAP@0.75": 0.0}

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(predictions, f)
        tmp = f.name

    coco_dt = coco_gt.loadRes(tmp)
    e = COCOeval(coco_gt, coco_dt, "bbox")
    e.evaluate()
    e.accumulate()
    e.summarize()

    stats = e.stats
    return {
        "mAP@0.5:0.95": float(stats[0]),
        "mAP@0.5": float(stats[1]),
        "mAP@0.75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "AR@1": float(stats[6]),
        "AR@10": float(stats[7]),
        "AR@100": float(stats[8]),
    }
