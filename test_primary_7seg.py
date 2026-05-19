from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Optional, Tuple

import cv2

import ocr_engine
import ocr_pi


ROOT = Path(__file__).resolve().parent


def load_image(path: str):
    image = cv2.imread(str(ROOT / path))
    if image is None:
        raise AssertionError(f"failed to load fixture: {path}")
    return image


def crop_xyxy(image, roi: Optional[Tuple[int, int, int, int]] = None):
    if roi is None:
        return image
    x1, y1, x2, y2 = roi
    return image[y1:y2, x1:x2].copy()


def configured_roi() -> Tuple[int, int, int, int]:
    with (ROOT / "config.json").open("r", encoding="utf-8") as f:
        values = json.load(f)["roi_coordinates"]
    return tuple(int(value) for value in values)


def run_fixture(path: str, roi: Optional[Tuple[int, int, int, int]] = None):
    image = crop_xyxy(load_image(path), roi)
    text, conf, raw, debug = ocr_pi.fast_ocr_from_lcd_roi(
        image,
        ocr_engine.Params(),
        allow_tesseract_fallback=False,
    )
    return text, conf, raw, debug


class Primary7SegmentTests(unittest.TestCase):
    def assert_has_primary_labels(self, debug) -> None:
        labels = debug.get("debug_labels", [])
        self.assertIn("primary_7seg_elapsed_ms", debug)
        self.assertTrue(
            "primary_7seg_success" in labels
            or (
                "primary_7seg_failed" in labels
                and "fallback_existing_pipeline_used" in labels
            ),
            labels,
        )

    def test_primary_success_label_on_real_lcd_red_band(self) -> None:
        text, _conf, _raw, debug = run_fixture("test_real_lcd/lcd_0p000.jpg")

        self.assertEqual(text, "0.000")
        self.assertIn("primary_7seg_success", debug.get("debug_labels", []))
        self.assertFalse(debug.get("fallback_existing_pipeline_used", True))

    def test_fallback_label_on_v2_cropped_primary_failure(self) -> None:
        text, _conf, _raw, debug = run_fixture("test_v2_cropped/ram_gene_0p03.png")

        self.assertEqual(text, "0.03")
        self.assertIn("primary_7seg_failed", debug.get("debug_labels", []))
        self.assertIn("fallback_existing_pipeline_used", debug.get("debug_labels", []))
        self.assertTrue(debug.get("fallback_existing_pipeline_used", False))

    def test_primary_labels_on_configured_v2_roi(self) -> None:
        text, _conf, _raw, debug = run_fixture("test_v2/ram_gene_0p03.png", configured_roi())

        self.assertEqual(text, "0.03")
        self.assert_has_primary_labels(debug)

    def test_primary_labels_on_another_real_lcd_frame(self) -> None:
        text, _conf, _raw, debug = run_fixture("test_real_lcd/lcd_0p260.jpg")

        self.assertEqual(text, "0.260")
        self.assert_has_primary_labels(debug)

    def test_legacy_integer_cropped_readings_are_not_decimalized(self) -> None:
        for fixture, expected in (
            ("test_v2_cropped/ram_gene_120.png", "120"),
            ("test_v2_cropped/ram_gene_480.png", "480"),
        ):
            with self.subTest(fixture=fixture):
                text, _conf, _raw, debug = run_fixture(fixture)

                self.assertEqual(text, expected)
                for summary in debug.get("candidate_summaries", []):
                    self.assertNotIn("missing_decimal", summary.get("suspicious_tokens", []))

    def test_configured_v2_480_is_not_decimalized(self) -> None:
        text, _conf, _raw, debug = run_fixture("test_v2/ram_gene_480.png", configured_roi())

        self.assertEqual(text, "480")
        for summary in debug.get("candidate_summaries", []):
            self.assertNotIn("missing_decimal", summary.get("suspicious_tokens", []))

    def test_v2_cropped_fractional_seven_alias_beats_false_point_ten(self) -> None:
        text, _conf, _raw, debug = run_fixture("test_v2_cropped/ram_gene_25p70.png")

        self.assertEqual(text, "25.70")
        self.assertIn("fractional_7_from_smeared_top", debug.get("winner_label", ""))


if __name__ == "__main__":
    unittest.main()
