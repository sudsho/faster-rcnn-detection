"""Generate train/val splits as separate COCO json files.

Reads the merged VOC json (output of `voc_to_coco`) and writes
`train.json` + `val.json` under the same `annotations/` folder.
Image-level split is taken from VOC's own `ImageSets/Main/<split>.txt`.
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-json", required=True)
    ap.add_argument("--voc-root", required=True)
    ap.add_argument("--train-list", default="ImageSets/Main/train.txt")
    ap.add_argument("--val-list", default="ImageSets/Main/val.txt")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    with open(args.coco_json) as f:
        d = json.load(f)

    def read(p):
        with open(os.path.join(args.voc_root, p)) as f:
            return {line.strip() for line in f if line.strip()}

    train_ids = read(args.train_list)
    val_ids = read(args.val_list)

    def filter_by(name_set):
        keep_ids = []
        for im in d["images"]:
            stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
            if stem in name_set:
                keep_ids.append(im["id"])
        keep_set = set(keep_ids)
        return {
            "images": [im for im in d["images"] if im["id"] in keep_set],
            "annotations": [a for a in d["annotations"] if a["image_id"] in keep_set],
            "categories": d["categories"],
        }

    os.makedirs(args.out_dir, exist_ok=True)
    for name, ids in [("train.json", train_ids), ("val.json", val_ids)]:
        out = filter_by(ids)
        with open(os.path.join(args.out_dir, name), "w") as f:
            json.dump(out, f)
        print("wrote", name, len(out["images"]), "images")


if __name__ == "__main__":
    main()
