"""Offline tiny-CPU smoke for faster-rcnn-detection.

Proves the training and inference plumbing end to end without a GPU, a
real dataset, or any network download:

  1. build the torchvision Faster R-CNN from RANDOM init (pretrained=False
     -> no COCO weights, no ImageNet backbone download);
  2. synthesize a handful of tiny images by drawing colored rectangles and
     recording their boxes as ground truth (no dataset on disk);
  3. run a train-mode forward pass so the model returns a loss dict, then
     take a few SGD steps and check the total loss goes down;
  4. switch to eval mode and run inference, checking the output has the
     right structure (boxes / labels / scores tensors of matching length).

The real headline (VOC / COCO mAP) needs a GPU and a real dataset; this
only exercises the code paths on synthetic data.
"""
import os
import sys

# Belt-and-suspenders: forbid any accidental torch.hub download during the
# smoke. build_model(pretrained=False) already avoids it; this makes a
# regression fail loudly offline instead of hanging on a socket.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import build_model  # noqa: E402
from src.utils import set_seed, collate_fn  # noqa: E402


# label 0 is background; foreground classes are 1 (red) and 2 (green)
NUM_CLASSES = 3
IMG_SIZE = 128


def make_sample(rng):
    """Draw 1-2 solid rectangles on a noisy background; return image + target."""
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (40, 40, 60))
    draw = ImageDraw.Draw(img)
    colors = {1: (220, 30, 30), 2: (30, 200, 60)}
    boxes, labels = [], []
    n = rng.randint(1, 2)
    for _ in range(n):
        label = rng.randint(1, 2)
        w = rng.randint(24, 48)
        h = rng.randint(24, 48)
        x1 = rng.randint(4, IMG_SIZE - w - 4)
        y1 = rng.randint(4, IMG_SIZE - h - 4)
        x2, y2 = x1 + w, y1 + h
        draw.rectangle([x1, y1, x2, y2], fill=colors[label])
        boxes.append([x1, y1, x2, y2])
        labels.append(label)
    target = {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }
    return TF.to_tensor(img), target


def main():
    set_seed(0)
    import random
    rng = random.Random(0)
    device = torch.device("cpu")

    print(f"torch {torch.__version__}  device={device}")

    # fixed synthetic batch we repeatedly overfit so loss must fall
    images, targets = zip(*[make_sample(rng) for _ in range(4)])
    images = [im.to(device) for im in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    n_boxes = sum(int(t["boxes"].shape[0]) for t in targets)
    print(f"synthetic batch: {len(images)} images {IMG_SIZE}x{IMG_SIZE}, "
          f"{n_boxes} boxes, classes {sorted({int(l) for t in targets for l in t['labels']})}")

    model = build_model(NUM_CLASSES, pretrained=False).to(device)
    # Keep the internal resize at the synthetic image size instead of the
    # default 800px min side, so a CPU step is cheap. Freeze the resnet body
    # so only the FPN + RPN + detection heads train: faster backward and a
    # stable loss from random init.
    model.transform.min_size = (IMG_SIZE,)
    model.transform.max_size = IMG_SIZE
    for p in model.backbone.body.parameters():
        p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.003, momentum=0.9, weight_decay=5e-4)

    model.train()
    n_steps = 30
    first_loss = None
    last_loss = None
    for step in range(n_steps):
        loss_dict = model(list(images), list(targets))
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
        last_loss = float(loss.item())
        if first_loss is None:
            first_loss = last_loss
            print("loss components:", {k: round(float(v), 4) for k, v in loss_dict.items()})
        if step % 5 == 0 or step == n_steps - 1:
            print(f"  step {step:2d}  total_loss {last_loss:.4f}")

    print(f"train-mode loss: {first_loss:.4f} -> {last_loss:.4f}")
    assert first_loss is not None and last_loss < first_loss, (
        f"loss did not decrease ({first_loss:.4f} -> {last_loss:.4f})")

    # eval-mode inference: structure check
    model.eval()
    with torch.no_grad():
        outputs = model([images[0]])
    assert isinstance(outputs, list) and len(outputs) == 1
    out = outputs[0]
    for key in ("boxes", "labels", "scores"):
        assert key in out and torch.is_tensor(out[key]), f"missing {key}"
    nb = out["boxes"].shape[0]
    assert out["boxes"].shape == (nb, 4)
    assert out["labels"].shape == (nb,)
    assert out["scores"].shape == (nb,)
    print(f"eval-mode inference: {nb} raw detections, "
          f"boxes{tuple(out['boxes'].shape)} labels{tuple(out['labels'].shape)} "
          f"scores{tuple(out['scores'].shape)}")

    print("SMOKE OK")


if __name__ == "__main__":
    main()
