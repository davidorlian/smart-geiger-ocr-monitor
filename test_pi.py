from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import pytesseract

import ocr_engine
import ocr_pi


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

    print(f"[PI TEST] images={len(image_paths)} roi={roi if roi else 'full-image'} tesseract_fallback={fallback_enabled}")
    for index, image_path in enumerate(image_paths, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"{index:02d}) {image_path.name}: ERROR failed to load")
            continue

        roi_image = crop_xyxy(image, roi)
        if roi_image is None or roi_image.size == 0:
            print(f"{index:02d}) {image_path.name}: ERROR invalid ROI for image shape={image.shape[:2]}")
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
        print(
            f"{index:02d}) [{status}] {image_path.name:<24} expected='{expected or '?'}' "
            f"got='{text}' conf={conf:.1f} elapsed={elapsed_ms:.1f}ms "
            f"source={source} winner={winner} stage={stage} attempts={attempts} raw='{short_raw(raw)}'"
        )

    if comparable:
        print(f"[PI TEST] exact={exact}/{comparable} accuracy={exact / float(max(1, comparable)):.3f} images={total}")
    else:
        print(f"[PI TEST] images={total}")


if __name__ == "__main__":
    main()
