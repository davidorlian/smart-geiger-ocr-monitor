from __future__ import annotations

from typing import Any, Dict

import ocr_engine
import ocr_pi


Params = ocr_engine.Params

_FAST_ALLOW_TESSERACT_FALLBACK = False


def _value_from_text(text: str, debug: Any) -> tuple[float | None, Any]:
    if not text:
        return None, debug
    try:
        return float(text), debug
    except ValueError:
        debug_dict = debug if isinstance(debug, dict) else {}
        debug_dict["rejected"] = "float_conversion"
        return None, debug_dict


def _normalize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(result)
    conf = float(normalized.get("conf", normalized.get("confidence", 0.0)) or 0.0)
    debug = normalized.get("debug", {})
    source = debug.get("source", "") if isinstance(debug, dict) else ""

    normalized["conf"] = conf
    normalized["confidence"] = conf
    normalized["source"] = source
    normalized.setdefault("raw", "")
    normalized.setdefault("text", "")
    normalized.setdefault("debug", debug)
    normalized.setdefault("value", None)
    return normalized


def _result_from_ocr_tuple(text: str, conf: float, raw: str, debug: Any) -> Dict[str, Any]:
    value, debug = _value_from_text(text, debug)
    return _normalize_result(
        {
            "value": value,
            "text": text,
            "conf": conf,
            "raw": raw,
            "debug": debug,
        }
    )


def read_number_from_roi(
    roi_image: Any,
    mode: str = "fast",
    params: Params | None = None,
) -> Dict[str, Any]:
    """Read a numeric measurement from an already-cropped display ROI.

    This facade is the active OCR interface for runtime code. For now it
    delegates to the existing implementations without changing their behavior:
    fast mode uses the Raspberry Pi fast path, and full mode uses the robust
    multi-candidate OCR path.
    """
    selected_mode = mode.strip().lower()
    p = params or Params()

    if selected_mode == "fast":
        text, conf, raw, debug = ocr_pi.fast_ocr_from_lcd_roi(
            roi_image,
            p,
            allow_tesseract_fallback=_FAST_ALLOW_TESSERACT_FALLBACK,
        )
        return _result_from_ocr_tuple(text, conf, raw, debug)

    if selected_mode == "full":
        text, conf, raw, debug = ocr_engine.robust_ocr_from_lcd_roi(roi_image, p)
        return _result_from_ocr_tuple(text, conf, raw, debug)

    raise ValueError(f"Unsupported OCR engine mode: {mode!r}. Expected 'fast' or 'full'.")
