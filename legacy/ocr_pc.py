from __future__ import annotations

from typing import Any, Dict, Tuple

import ocr_engine as _engine
from ocr_engine import *  # noqa: F401,F403 - explicit PC facade over existing engine API.


def robust_ocr_from_lcd_roi(lcd_roi_bgr, p: Params) -> Tuple[str, float, str, Dict[str, Any]]:  # noqa: F405
    """PC/debug OCR strategy.

    This intentionally preserves the existing heavy behavior in ocr_engine:
    multiple reading ROI candidates, parameter variants, voting/scoring, and
    optional Tesseract fallbacks.
    """
    return _engine.robust_ocr_from_lcd_roi(lcd_roi_bgr, p)


def read_number_from_lcd_roi(lcd_roi_bgr, p: Params | None = None) -> Dict[str, Any]:  # noqa: F405
    params = p or Params()  # noqa: F405
    text, conf, raw, debug = robust_ocr_from_lcd_roi(lcd_roi_bgr, params)
    if not text:
        return {"value": None, "text": text, "conf": conf, "raw": raw, "debug": debug}

    try:
        value = float(text)
    except ValueError:
        debug = debug if isinstance(debug, dict) else {}
        debug["rejected"] = "float_conversion"
        return {"value": None, "text": text, "conf": conf, "raw": raw, "debug": debug}

    return {"value": value, "text": text, "conf": conf, "raw": raw, "debug": debug}
