from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import pytesseract

import ocr_engine
from legacy import ocr_pi


def parse_xyxy_roi(roi_arg: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not roi_arg:
        return None
    try:
        values = [int(part.strip()) for part in roi_arg.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--roi must contain integers: x1,y1,x2,y2") from exc
    if len(values) != 4:
        raise argparse.ArgumentTypeError("--roi must contain exactly four values: x1,y1,x2,y2")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError("--roi must satisfy x2>x1 and y2>y1")
    return x1, y1, x2, y2


def load_config_roi(config_path: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not config_path:
        return None
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    roi = data.get("roi_coordinates")
    if roi is None:
        return None
    if not isinstance(roi, list) or len(roi) != 4:
        raise ValueError("config roi_coordinates must be [x1, y1, x2, y2]")
    x1, y1, x2, y2 = (int(v) for v in roi)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("config roi_coordinates must satisfy x2>x1 and y2>y1")
    return x1, y1, x2, y2


def crop_xyxy(image: Any, roi: Optional[Tuple[int, int, int, int]]) -> Any:
    if roi is None:
        return image.copy()
    h, w = image.shape[:2]
    x1, y1, x2, y2 = roi
    x1 = max(0, min(w, int(x1)))
    y1 = max(0, min(h, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2].copy()


def select_roi_xyxy(image: Any, image_name: str) -> Optional[Tuple[int, int, int, int]]:
    print(f"[ROI] Select LCD numeric window for {image_name}. ENTER/SPACE accepts, ESC cancels.")
    window_name = f"Select ROI - {image_name}"
    roi = cv2.selectROI(window_name, image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)
    if roi == (0, 0, 0, 0):
        return None

    x, y, w, h = (int(v) for v in roi)
    if w <= 0 or h <= 0:
        return None
    return x, y, x + w, y + h


def collect_images(image: Optional[str], image_dir: Optional[str]) -> List[Path]:
    paths = ocr_engine.collect_image_paths(image, image_dir)
    return [path for path in paths if path.exists()]


def short_raw(raw: str, limit: int = 72) -> str:
    raw = raw.replace("\n", " ")
    return raw if len(raw) <= limit else raw[: limit - 3] + "..."


def main() -> None:
    ap = argparse.ArgumentParser(description="Raspberry Pi lightweight OCR test on existing image files only.")
    ap.add_argument("--image", default=None, help="Single image path. Overrides --image-dir.")
    ap.add_argument("--image-dir", default=str(ocr_engine.default_test_dir()), help="Directory of images to test.")
    ap.add_argument("--roi", default=None, help="Optional image ROI as x1,y1,x2,y2. Uses full image if omitted.")
    ap.add_argument("--config", default=None, help="Optional config.json containing roi_coordinates.")
    ap.add_argument("--tess", default=None, help="Optional path to Tesseract executable.")
    ap.add_argument("--no-tesseract-fallback", action="store_true", help="Disable final Tesseract fallback.")
    ap.add_argument("--expand-weak-7seg", action="store_true", help="Keep expanding after weak-but-valid 7-seg reads.")
    ap.add_argument(
        "--select-roi-each-image",
        action="store_true",
        help="Interactively select an x1,y1,x2,y2 ROI for each image before running Pi OCR.",
    )
    args = ap.parse_args()

    if args.tess:
        pytesseract.pytesseract.tesseract_cmd = args.tess

    roi = parse_xyxy_roi(args.roi) if args.roi else load_config_roi(args.config)
    image_paths = collect_images(args.image, args.image_dir)
    if not image_paths:
        raise SystemExit(f"No image files found. Checked: {args.image or args.image_dir}")

    total = 0
    comparable = 0
    exact = 0
    params = ocr_engine.Params()
    fallback_enabled = not bool(args.no_tesseract_fallback)

    roi_label = "select-each-image" if args.select_roi_each_image else roi if roi else "full-image"
    print(f"[PI TEST] images={len(image_paths)} roi={roi_label} tesseract_fallback={fallback_enabled}")
    for index, image_path in enumerate(image_paths, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"{index:02d}) {image_path.name}: ERROR failed to load")
            continue

        current_roi = roi
        if args.select_roi_each_image:
            selected_roi = select_roi_xyxy(image, image_path.name)
            if selected_roi is None:
                print(f"{index:02d}) {image_path.name}: SKIP ROI selection cancelled")
                continue
            current_roi = selected_roi
            print(f"{index:02d}) {image_path.name}: selected_roi={current_roi}")

        roi_image = crop_xyxy(image, current_roi)
        if roi_image is None or roi_image.size == 0:
            print(f"{index:02d}) {image_path.name}: ERROR invalid ROI {current_roi} for image shape={image.shape[:2]}")
            continue

        start = time.perf_counter()
        text, conf, raw, debug = ocr_pi.fast_ocr_from_lcd_roi(
            roi_image,
            params,
            allow_tesseract_fallback=fallback_enabled,
            expand_weak_7seg=bool(args.expand_weak_7seg),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        expected = ocr_engine.expected_text_from_filename(image_path)
        status = "UNK"
        if expected is not None:
            comparable += 1
            status = "OK " if text == expected else "BAD"
            exact += int(text == expected)
        total += 1

        winner = debug.get("winner_label", "") if isinstance(debug, dict) else ""
        stage = debug.get("winner_stage", "") if isinstance(debug, dict) else ""
        attempts = int(debug.get("attempt_count", 0)) if isinstance(debug, dict) else 0
        source = debug.get("source", "") if isinstance(debug, dict) else ""
        labels = debug.get("debug_labels", []) if isinstance(debug, dict) else []
        label_text = ",".join(str(item) for item in labels) if isinstance(labels, list) else str(labels)
        primary_ms = float(debug.get("primary_7seg_elapsed_ms", 0.0)) if isinstance(debug, dict) else 0.0
        print(
            f"{index:02d}) [{status}] {image_path.name:<24} expected='{expected or '?'}' "
            f"got='{text}' conf={conf:.1f} elapsed={elapsed_ms:.1f}ms "
            f"source={source} winner={winner} stage={stage} attempts={attempts} "
            f"labels={label_text} primary_ms={primary_ms:.1f} raw='{short_raw(raw)}'"
        )

    if comparable:
        print(f"[PI TEST] exact={exact}/{comparable} accuracy={exact / float(max(1, comparable)):.3f} images={total}")
    else:
        print(f"[PI TEST] images={total}")


if __name__ == "__main__":
    main()
