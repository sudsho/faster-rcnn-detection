"""Dataset utilities for object detection.

Reads COCO-style annotation jsons. The annotation format we care about:

    {
      "images": [{"id": 1, "file_name": "...", "width": W, "height": H}, ...],
      "annotations": [
          {"id": k, "image_id": 1, "category_id": c, "bbox": [x, y, w, h],
           "area": ..., "iscrowd": 0}, ...
      ],
      "categories": [{"id": c, "name": "..."}, ...]
    }
"""
import json
import os
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset


class CocoDetection(Dataset):
    def __init__(self, root, ann_file, transforms=None, drop_empty=False):
        self.root = root
        self.transforms = transforms

        with open(ann_file, "r") as f:
            coco = json.load(f)

        self.images = {im["id"]: im for im in coco["images"]}
        self.categories = {c["id"]: c["name"] for c in coco["categories"]}

        anns_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            anns_by_image[ann["image_id"]].append(ann)
        self.anns_by_image = anns_by_image

        ids = sorted(self.images.keys())
        if drop_empty:
            ids = [i for i in ids if anns_by_image.get(i)]
        self.image_ids = ids

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        im_meta = self.images[img_id]
        path = os.path.join(self.root, im_meta["file_name"])
        img = Image.open(path).convert("RGB")

        anns = self.anns_by_image.get(img_id, [])
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(a["category_id"])
            areas.append(a.get("area", w * h))
            iscrowd.append(a.get("iscrowd", 0))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
