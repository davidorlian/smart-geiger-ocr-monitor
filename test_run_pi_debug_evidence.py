from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path
from typing import Any, Dict

import numpy as np

import run_pi


ROOT = Path(__file__).resolve().parent


class RunPiDebugEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_get_image = run_pi._get_image_from_pi_camera
        self.original_read_number = run_pi._read_number_from_roi
        self.original_save_debug = run_pi._save_suspicious_debug_capture
        self.case_dir = ROOT / "logs" / "run_pi_debug_evidence_tests" / uuid.uuid4().hex
        self.debug_root = self.case_dir / "debug_captures"
        self.case_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        run_pi._get_image_from_pi_camera = self.original_get_image
        run_pi._read_number_from_roi = self.original_read_number
        run_pi._save_suspicious_debug_capture = self.original_save_debug
        shutil.rmtree(self.case_dir, ignore_errors=True)

    def run_measurement(self, result: Dict[str, Any]) -> tuple[str, Path, str]:
        full_image = np.zeros((48, 64, 3), dtype=np.uint8)
        full_image[8:28, 10:40] = (40, 80, 120)
        roi_coords = (10, 8, 40, 28)
        log_path = self.case_dir / "monitor.log"

        run_pi._get_image_from_pi_camera = lambda _resolution: full_image.copy()
        run_pi._read_number_from_roi = lambda _roi_image: result
        run_pi._save_suspicious_debug_capture = (
            lambda full_image_arg, roi_image_arg, timestamp, reasons, result_arg, roi_coords_arg:
            self.original_save_debug(
                full_image_arg,
                roi_image_arg,
                timestamp,
                reasons,
                result_arg,
                roi_coords_arg,
                debug_dir=str(self.debug_root),
            )
        )

        run_pi._run_single_measurement(
            roi_coords=roi_coords,
            warning_threshold=0.6,
            critical_threshold=1.2,
            camera_resolution=(64, 48),
            log_file_path=str(log_path),
            email_settings=None,
            no_alerts=True,
            print_ocr_summary=False,
            save_debug_images=False,
        )

        log_text = log_path.read_text(encoding="utf-8")
        folders = [path for path in self.debug_root.iterdir() if path.is_dir()]
        self.assertEqual(len(folders), 1)
        folder = folders[0]
        debug_text = (folder / "debug.txt").read_text(encoding="utf-8")
        self.assertTrue((folder / "full.jpg").exists())
        self.assertTrue((folder / "roi_crop.jpg").exists())
        self.assertIn(str(folder), log_text)
        self.assertIn("roi_coordinates: [10, 8, 40, 28]", debug_text)
        return log_text, folder, debug_text

    def test_rejected_unreadable_reading_saves_debug_evidence(self) -> None:
        _log_text, folder, debug_text = self.run_measurement(
            {
                "value": None,
                "text": "",
                "conf": 0.0,
                "raw": "",
                "debug": {
                    "rejected": "no_valid_7seg_or_tesseract",
                    "attempt_count": 3,
                    "winner_raw": "[7seg:none]",
                },
            }
        )

        self.assertIn("no_valid_7seg_or_tesseract", folder.name)
        self.assertIn("reason: no_valid_7seg_or_tesseract", debug_text)
        self.assertIn("attempts_count: 3", debug_text)
        self.assertIn("parsed_value: N/A", debug_text)

    def test_missing_decimal_reading_saves_debug_evidence(self) -> None:
        _log_text, folder, debug_text = self.run_measurement(
            {
                "value": 12.0,
                "text": "12",
                "conf": 95.0,
                "raw": "[7seg:1:bc,2:abdeg]",
                "debug": {"attempt_count": 1, "source": "7seg"},
            }
        )

        self.assertIn("missing_decimal_point", folder.name)
        self.assertIn("reason: missing_decimal_point", debug_text)
        self.assertIn("ocr_text: 12", debug_text)
        self.assertIn("parsed_value: 12.0", debug_text)

    def test_attempts_ge_20_saves_debug_evidence(self) -> None:
        _log_text, folder, debug_text = self.run_measurement(
            {
                "value": 0.404,
                "text": "0.404",
                "conf": 85.0,
                "raw": "[7seg:0,.:dot,4,0,4]",
                "debug": {"attempt_count": 20, "source": "7seg"},
            }
        )

        self.assertIn("attempts_ge_20_20", folder.name)
        self.assertIn("reason: attempts_ge_20_20", debug_text)
        self.assertIn("attempts_count: 20", debug_text)

    def test_unusually_high_value_saves_debug_evidence(self) -> None:
        _log_text, folder, debug_text = self.run_measurement(
            {
                "value": 9.0,
                "text": "9.000",
                "conf": 95.0,
                "raw": "[7seg:9,.:dot,0,0,0]",
                "debug": {"attempt_count": 1, "source": "7seg"},
            }
        )

        self.assertIn("unusually_high_reading", folder.name)
        self.assertIn("reason: unusually_high_reading", debug_text)
        self.assertIn("parsed_value: 9.0", debug_text)


if __name__ == "__main__":
    unittest.main()
