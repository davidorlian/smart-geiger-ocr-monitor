from __future__ import annotations

import ocr_pc
import legacy_manual_ocr as _legacy_manual_ocr


def main() -> None:
    """Explicit legacy PC OCR manual entry point.

    The old lab UI and batch runner live in legacy_manual_ocr.py. Keep that
    behavior intact and route the OCR strategy through ocr_pc.py.
    """
    _legacy_manual_ocr.robust_ocr_from_lcd_roi = ocr_pc.robust_ocr_from_lcd_roi
    _legacy_manual_ocr.main()


if __name__ == "__main__":
    main()
