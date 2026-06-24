import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import engine


IMAGE_PATH = PROJECT_DIR / "test_sets" / "green_multimeter_v3_cleaned" / "cropped" / "meter_hold_06p525.jpg"
OUT_DIR = PROJECT_DIR / "report_figures"
OUT_PATH = OUT_DIR / "engine_flow_06p525.png"


def to_bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def resize_height(img, target_h=180):
    h, w = img.shape[:2]
    scale = target_h / float(h)
    return cv2.resize(img, (int(w * scale), target_h), interpolation=cv2.INTER_AREA)


def add_title(img, title):
    pad = 38
    canvas = np.full((img.shape[0] + pad, img.shape[1], 3), 255, dtype=np.uint8)
    canvas[pad:, :] = img
    cv2.putText(canvas, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    return canvas


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    roi = cv2.imread(str(IMAGE_PATH))
    if roi is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

    result = engine.read_number_from_roi(roi, mode="fast")
    text = result.get("text", "")
    value = result.get("value", "")
    conf = result.get("conf", 0)

    params = engine.Params()
    stages = engine.preprocess(roi, params)

    gray = stages["gray"]
    mask = stages["after_filters"]

    panels = [
        add_title(resize_height(roi), "Original ROI"),
        add_title(resize_height(to_bgr(gray)), "Grayscale"),
        add_title(resize_height(to_bgr(mask)), "Processed mask"),
    ]

    max_h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < max_h:
            extra = np.full((max_h - p.shape[0], p.shape[1], 3), 255, dtype=np.uint8)
            p = np.vstack([p, extra])
        padded.append(p)

    spacer = np.full((max_h, 20, 3), 255, dtype=np.uint8)
    figure = padded[0]
    for p in padded[1:]:
        figure = np.hstack([figure, spacer, p])

    footer_h = 55
    footer = np.full((footer_h, figure.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        footer,
        f"OCR result: {text}   value={value}   confidence={conf}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        2,
    )

    final = np.vstack([figure, footer])
    cv2.imwrite(str(OUT_PATH), final)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
