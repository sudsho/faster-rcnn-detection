"""Drawing helpers."""
from PIL import Image, ImageDraw, ImageFont


PALETTE = [
    (220, 20, 60), (255, 215, 0), (50, 205, 50), (30, 144, 255),
    (148, 0, 211), (255, 140, 0), (47, 79, 79), (210, 105, 30),
    (255, 105, 180), (128, 128, 0), (70, 130, 180), (139, 69, 19),
    (75, 0, 130), (0, 191, 255), (220, 20, 60), (152, 251, 152),
    (255, 99, 71), (218, 112, 214), (123, 104, 238), (255, 250, 205),
]


def _color_for(label):
    return PALETTE[(label - 1) % len(PALETTE)]


def draw_boxes(image, boxes, labels, scores, classes=None, score_thresh=0.0):
    """Return a copy of the PIL image with boxes drawn."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        color = _color_for(int(label))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        name = classes[int(label) - 1] if classes else str(label)
        text = f"{name} {score:.2f}"
        try:
            tw, th = draw.textsize(text, font=font)
        except AttributeError:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ty = max(0, y1 - th - 2)
        draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 2], fill=color)
        draw.text((x1 + 2, ty), text, fill=(255, 255, 255), font=font)
    return img
