from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import cv2

import test_ocr


ImageRoiMap = Dict[str, Tuple[int, int, int, int]]
ImageParamMap = Dict[str, test_ocr.Params]


def expected_from_filename(image_path: Path) -> str:
    expected = test_ocr.expected_text_from_filename(image_path)
    if expected is None:
        raise ValueError(f"Unsupported benchmark filename format: {image_path.name}")
    return expected


def collect_images(image_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=lambda p: p.name.lower(),
    )


def load_config_roi(config_path: Path) -> Optional[Tuple[int, int, int, int]]:
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    roi = data.get("roi_coordinates")
    if roi is None:
        return None
    if not isinstance(roi, list) or len(roi) != 4:
        raise ValueError("config roi_coordinates must be [x1, y1, x2, y2]")
    x1, y1, x2, y2 = (int(v) for v in roi)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("config roi_coordinates must satisfy x2>x1 and y2>y1")
    return (x1, y1, x2 - x1, y2 - y1)


def resolve_roi(
    image_path: Path,
    image: Any,
    fixed_roi: Optional[Tuple[int, int, int, int]],
    image_rois: ImageRoiMap,
) -> Tuple[int, int, int, int]:
    key = test_ocr.image_roi_key(image_path)
    if key in image_rois:
        return image_rois[key]
    if fixed_roi is not None:
        return fixed_roi
    h, w = image.shape[:2]
    return (0, 0, w, h)


def crop_xywh(image: Any, roi: Tuple[int, int, int, int]) -> Any:
    cropped = test_ocr.crop_roi(image, roi)
    if cropped is None or cropped.size == 0:
        # Cropped-reading datasets should not fail just because config.json
        # still contains a full-photo mounted-camera ROI.
        return image.copy()
    return cropped


def run_robust(
    lcd_roi_bgr: Any,
    params: test_ocr.Params,
) -> Tuple[str, float, str, Dict[str, Any]]:
    return test_ocr.robust_ocr_from_lcd_roi(lcd_roi_bgr, params)


def run_tesseract_only(
    lcd_roi_bgr: Any,
    params: test_ocr.Params,
) -> Tuple[str, float, str, Dict[str, Any]]:
    stages = test_ocr.preprocess(lcd_roi_bgr, params)
    text, conf, raw = test_ocr.ocr_once(stages["ocr_input"], params)
    debug = {
        "winner_label": "tesseract/base",
        "winner_stage": "ocr_input",
        "winner_raw": raw,
        "winner_stages": stages,
    }
    return text, conf, f"[tesseract:{raw}]", debug


def iter_cases(
    image_paths: Iterable[Path],
    fixed_roi: Optional[Tuple[int, int, int, int]],
    image_rois: ImageRoiMap,
    image_params: ImageParamMap,
    default_params: test_ocr.Params,
):
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")
        roi = resolve_roi(image_path, image, fixed_roi, image_rois)
        roi_image = crop_xywh(image, roi)
        params = test_ocr.clone_params(image_params.get(test_ocr.image_roi_key(image_path), default_params))
        yield image_path, image, roi, roi_image, params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", default=str(test_ocr.default_test_dir()), help="Directory of benchmark images.")
    ap.add_argument("--profile", default=None, help="Optional ocr_profile.json from test_ocr.py.")
    ap.add_argument("--config", default="config.json", help="Optional config.json for a single fixed ROI.")
    ap.add_argument(
        "--engine",
        choices=("robust", "tesseract"),
        default="robust",
        help="OCR engine to benchmark.",
    )
    ap.add_argument("--limit", type=int, default=0, help="Optional image limit for quick checks.")
    args = ap.parse_args()

    image_dir = Path(args.image_dir).resolve()
    image_paths = collect_images(image_dir)
    if not image_paths:
        raise SystemExit(f"No benchmark images found in: {image_dir}")
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    params = test_ocr.Params()
    fixed_roi: Optional[Tuple[int, int, int, int]] = None
    image_rois: ImageRoiMap = {}
    image_params: ImageParamMap = {}

    if args.profile:
        profile_path = Path(args.profile)
        if profile_path.exists():
            params, profile_roi, image_rois, image_params = test_ocr.load_profile(str(profile_path))
            fixed_roi = profile_roi
        else:
            raise SystemExit(f"Profile not found: {profile_path}")
    elif args.config:
        fixed_roi = load_config_roi(Path(args.config))

    engine_fn = run_robust if args.engine == "robust" else run_tesseract_only

    total = 0
    exact = 0
    mismatches: list[Dict[str, Any]] = []

    print(f"[BENCH] engine={args.engine} images={len(image_paths)} image_dir={image_dir}")
    if image_rois:
        print(f"[BENCH] using {len(image_rois)} per-image ROI(s) from profile")
    elif fixed_roi is not None:
        print(f"[BENCH] using fixed ROI: {fixed_roi}")
    else:
        print("[BENCH] using full image as LCD ROI")

    for image_path, _image, roi, roi_image, case_params in iter_cases(
        image_paths, fixed_roi, image_rois, image_params, params
    ):
        expected = expected_from_filename(image_path)
        text, conf, raw, debug = engine_fn(roi_image, case_params)
        ok = text == expected
        total += 1
        exact += int(ok)
        label = debug.get("winner_label", "")
        stage = debug.get("winner_stage", "")
        status = "OK " if ok else "BAD"
        print(
            f"[{status}] {image_path.name:<22} expected='{expected}' got='{text}' "
            f"conf={conf:.1f} roi={roi} winner={label} stage={stage}"
        )
        if not ok:
            mismatches.append(
                {
                    "image": image_path.name,
                    "expected": expected,
                    "got": text,
                    "conf": conf,
                    "raw": raw,
                    "roi": roi,
                    "params": asdict(case_params),
                    "winner_label": label,
                    "winner_stage": stage,
                }
            )

    print()
    print(f"[BENCH] exact={exact}/{total} accuracy={exact / float(max(1, total)):.3f}")
    if mismatches:
        print("[BENCH] mismatches:")
        for item in mismatches:
            print(
                f"  - {item['image']}: expected='{item['expected']}' got='{item['got']}' "
                f"raw='{item['raw']}' winner={item['winner_label']} stage={item['winner_stage']}"
            )


if __name__ == "__main__":
    main()
