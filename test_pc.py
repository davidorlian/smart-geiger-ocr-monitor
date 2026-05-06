from __future__ import annotations

import ocr_pc
import test_ocr as _legacy_test_ocr


def main() -> None:
    """Explicit PC OCR test entry point.

    The current lab UI and batch runner live in test_ocr.py. For the staged
    split, keep that behavior intact and route the OCR strategy through
    ocr_pc.py.
    """
    _legacy_test_ocr.robust_ocr_from_lcd_roi = ocr_pc.robust_ocr_from_lcd_roi
    _legacy_test_ocr.main()


if __name__ == "__main__":
    main()
