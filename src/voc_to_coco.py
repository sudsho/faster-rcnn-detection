"""Convert Pascal VOC XML annotations to a single COCO-format json.

Usage:
    python -m src.voc_to_coco --voc-root data/voc/VOC2007 --split trainval \
        --out data/voc/annotations/voc2007_trainval.json
"""
import argparse
import json
import os
import xml.etree.ElementTree as ET


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc-root", required=True)
    ap.add_argument("--split", default="trainval")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    split_file = os.path.join(args.voc_root, "ImageSets", "Main", args.split + ".txt")
    with open(split_file) as f:
        ids = [line.strip() for line in f if line.strip()]

    cat_id = {name: i + 1 for i, name in enumerate(VOC_CLASSES)}
    images, anns, ann_id = [], [], 1
    for image_id, vid in enumerate(ids, 1):
        xml_path = os.path.join(args.voc_root, "Annotations", vid + ".xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        images.append({
            "id": image_id,
            "file_name": os.path.join("JPEGImages", vid + ".jpg"),
            "width": width,
            "height": height,
        })
        for obj in root.findall("object"):
            name = obj.find("name").text.strip()
            if name not in cat_id:
                continue
            difficult = int(obj.find("difficult").text or 0)
            bb = obj.find("bndbox")
            xmin = float(bb.find("xmin").text) - 1
            ymin = float(bb.find("ymin").text) - 1
            xmax = float(bb.find("xmax").text) - 1
            ymax = float(bb.find("ymax").text) - 1
            w = max(0.0, xmax - xmin)
            h = max(0.0, ymax - ymin)
            anns.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id[name],
                "bbox": [xmin, ymin, w, h],
                "area": w * h,
                "iscrowd": difficult,
            })
            ann_id += 1

    out = {
        "images": images,
        "annotations": anns,
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(VOC_CLASSES)],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print("wrote", args.out, len(images), "images,", len(anns), "anns")


if __name__ == "__main__":
    main()
