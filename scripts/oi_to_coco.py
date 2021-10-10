"""Convert an Open Images CSV class-export to a COCO-format json.

Expects images at <root>/<split>/ and a CSV with columns:
    ImageID, LabelName, XMin, XMax, YMin, YMax
which is what the OIDv4 toolkit produces.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--csv", required=True, help="annotations csv")
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cat_id = {c: i + 1 for i, c in enumerate(args.classes)}
    images = {}
    anns = []
    ann_id = 1

    img_dir = os.path.join(args.root, args.split)
    file_index = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir)}

    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            iid = row["ImageID"]
            if iid not in file_index:
                continue
            label = row["LabelName"]
            if label not in cat_id:
                continue
            if iid not in images:
                fname = file_index[iid]
                with Image.open(os.path.join(img_dir, fname)) as im:
                    w, h = im.size
                images[iid] = {
                    "id": len(images) + 1,
                    "file_name": os.path.join(args.split, fname),
                    "width": w,
                    "height": h,
                }
            im_id = images[iid]["id"]
            xmin = float(row["XMin"]) * images[iid]["width"]
            ymin = float(row["YMin"]) * images[iid]["height"]
            xmax = float(row["XMax"]) * images[iid]["width"]
            ymax = float(row["YMax"]) * images[iid]["height"]
            bw, bh = xmax - xmin, ymax - ymin
            if bw <= 0 or bh <= 0:
                continue
            anns.append({
                "id": ann_id,
                "image_id": im_id,
                "category_id": cat_id[label],
                "bbox": [xmin, ymin, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            })
            ann_id += 1

    out = {
        "images": list(images.values()),
        "annotations": anns,
        "categories": [{"id": i + 1, "name": c} for i, c in enumerate(args.classes)],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print("wrote", args.out, len(images), "images,", len(anns), "anns")


if __name__ == "__main__":
    main()
