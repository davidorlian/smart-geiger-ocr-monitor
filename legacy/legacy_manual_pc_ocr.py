from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legacy import legacy_manual_ocr as _legacy_manual_ocr
from legacy import ocr_pc


def main() -> None:
    """Explicit legacy PC OCR manual entry point.

    The old lab UI and batch runner live in legacy/legacy_manual_ocr.py. Keep
    that behavior intact and route the OCR strategy through legacy/ocr_pc.py.
    """
    _legacy_manual_ocr.robust_ocr_from_lcd_roi = ocr_pc.robust_ocr_from_lcd_roi
    _legacy_manual_ocr.main()


if __name__ == "__main__":
    main()
