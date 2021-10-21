# Results

## Pascal VOC 2007 + 2012 trainval -> VOC 2007 test

12 epochs, SGD lr=0.005, momentum=0.9, weight decay=5e-4, step LR @ epoch 8.
ResNet50-FPN backbone with the COCO-pretrained init, last 3 stages
trainable. Mixed precision off; batch 4 on a single 12GB GPU.

| metric         | value  |
|----------------|--------|
| mAP @ 0.5      | 0.764  |
| mAP @ 0.5:0.95 | 0.451  |
| AR @ 100       | 0.589  |

Per-class AP @ 0.5 (rounded):

| class       | AP    |
|-------------|-------|
| aeroplane   | 0.81  |
| bicycle     | 0.83  |
| bird        | 0.74  |
| boat        | 0.62  |
| bottle      | 0.61  |
| bus         | 0.83  |
| car         | 0.86  |
| cat         | 0.86  |
| chair       | 0.55  |
| cow         | 0.79  |
| diningtable | 0.66  |
| dog         | 0.83  |
| horse       | 0.84  |
| motorbike   | 0.81  |
| person      | 0.80  |
| pottedplant | 0.46  |
| sheep       | 0.72  |
| sofa        | 0.69  |
| train       | 0.83  |
| tvmonitor   | 0.74  |

Bottle, chair and pottedplant lag because they are the smallest and most
cluttered classes in VOC.

## What helped

- Warmup of 1000 iters with `warmup_factor=0.001` removed the spikes at
  iteration 0 caused by the freshly-initialised box head.
- `pin_memory=True` plus `num_workers=4` shaved ~15% off iteration time.
- albumentations' `min_visibility=0.1` saved the boxes that flips/crops
  push partly out of frame instead of dropping them entirely.

## What did not help

- Longer than 12 epochs on VOC. The val mAP plateaus around epoch 10.
- Stronger color jitter (>0.4) -- the model started detecting fewer
  bottles/tv monitors.
- `trainable_backbone_layers=5` (full fine-tune). Slightly better train
  loss, slightly worse val mAP. Might just need a smaller LR.
