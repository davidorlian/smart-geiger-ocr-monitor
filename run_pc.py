from __future__ import annotations

from typing import Any, Dict

import cv2

import ocr_pc
import run as _runtime


def _read_number_from_roi(roi_image: cv2.Mat) -> Dict[str, Any]:
    result = ocr_pc.read_number_from_lcd_roi(roi_image, ocr_pc.Params())
    text = str(result.get("text", ""))
    raw = str(result.get("raw", ""))
    debug = result.get("debug", {})

    if text:
        vote_count = int(debug.get("vote_count", 0)) if isinstance(debug, dict) else 0
        print(f"OCR PC robust pick: {text} {raw} via {vote_count} votes")
        return result

    winner_raw = debug.get("winner_raw", raw) if isinstance(debug, dict) else raw
    rejected = debug.get("rejected", "no_read") if isinstance(debug, dict) else "no_read"
    print(f"OCR PC robust pick rejected: {winner_raw} ({rejected})")
    return result


def _extract_number_from_roi(roi_image: cv2.Mat) -> float | None:
    result = _read_number_from_roi(roi_image)
    value = result.get("value")
    return float(value) if value is not None else None


def _extract_number_from_image_with_roi(
    image: cv2.Mat,
    roi_coords: tuple[int, int, int, int],
    pc_test_mode: bool = False,
) -> float | None:
    roi_image, roi_mode = _runtime.crop_configured_roi(
        image,
        roi_coords,
        allow_cropped_test_image_fallback=pc_test_mode,
    )
    if roi_image is None:
        print(f"OCR error: ROI is invalid for image dimensions. roi={roi_coords} shape={image.shape[:2]}")
        return None

    result = _read_number_from_roi(roi_image)
    text = str(result.get("text", ""))
    value = result.get("value")
    if text and value is not None:
        if roi_mode == "full_image_fallback":
            print("PC test mode: input image already looks cropped; using full image instead of configured ROI.")
        return float(value)
    return None


def run_monitoring() -> None:
    """PC monitoring/simulation entry point using the heavy PC OCR strategy."""
    print("run_pc: using ocr_pc heavy OCR strategy.")
    _runtime._read_number_from_roi = _read_number_from_roi
    _runtime.extract_number_from_roi = _extract_number_from_roi
    _runtime.extract_number_from_image_with_roi = _extract_number_from_image_with_roi
    _runtime.run_monitoring()


if __name__ == "__main__":
    run_monitoring()
