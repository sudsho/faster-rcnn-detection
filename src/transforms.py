"""Image + bbox transforms.

Wraps albumentations so the same pipeline can apply to PIL images and
their bounding boxes consistently.
"""
import numpy as np
import torch

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _HAS_ALB = True
except ImportError:
    _HAS_ALB = False


def _alb_train(min_size=600, max_size=1000, hflip=0.5, brightness=0.2, contrast=0.2):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=max_size),
            A.PadIfNeeded(min_height=min_size, min_width=min_size, border_mode=0),
            A.HorizontalFlip(p=hflip),
            A.RandomBrightnessContrast(brightness_limit=brightness, contrast_limit=contrast, p=0.5),
            A.MotionBlur(p=0.1),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.1),
    )


def _alb_val(max_size=1000):
    return A.Compose(
        [A.LongestMaxSize(max_size=max_size), ToTensorV2()],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


class AlbumentationsAdapter:
    def __init__(self, train=True):
        if not _HAS_ALB:
            raise ImportError("albumentations is required")
        self.transform = _alb_train() if train else _alb_val()

    def __call__(self, img, target):
        arr = np.array(img)
        out = self.transform(
            image=arr,
            bboxes=target["boxes"].tolist(),
            labels=target["labels"].tolist(),
        )
        img_t = out["image"].float() / 255.0
        target = dict(target)
        target["boxes"] = torch.tensor(out["bboxes"], dtype=torch.float32).reshape(-1, 4)
        target["labels"] = torch.tensor(out["labels"], dtype=torch.int64)
        return img_t, target
