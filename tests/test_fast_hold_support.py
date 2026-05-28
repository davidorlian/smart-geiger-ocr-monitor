from __future__ import annotations

import unittest
from pathlib import Path

import cv2

import engine


ROOT = Path(__file__).resolve().parents[1]


def load_image(path: str):
    image = cv2.imread(str(ROOT / path))
    if image is None:
        raise AssertionError(f"failed to load fixture: {path}")
    return image


def run_fast_fixture(path: str):
    return engine.read_number_from_roi(load_image(path), mode="fast", params=engine.Params())


def make_suspicious_candidate(
    text: str,
    *,
    crop: str,
    variant: str,
    score: float = 2.5,
) -> engine.Candidate:
    raw = "[7seg:1:narrow-smear1,.:dot,2:abeg-lowd,3:bcdg-3like]"
    return {
        "text": text,
        "conf": 95.0,
        "raw": raw,
        "source": "7seg",
        "crop": crop,
        "variant": variant,
        "phase": "test",
        "stage": "after_filters",
        "score": score,
        "structural_quality": 0.75,
        "artifact_penalty": 0.0,
        "penalties": [],
        "suspicious_tokens": engine._candidate_suspicious_tokens(text, raw),
        "reading_roi": None,
        "stages": {},
        "mask": None,
    }


class FastHoldSupportAggregationTests(unittest.TestCase):
    def test_repeated_thresholds_from_same_crop_do_not_support_suspicious_text(self) -> None:
        candidates = [
            make_suspicious_candidate("12.3", crop="window_frame_trim6", variant=variant)
            for variant in ("base", "dt-4", "dt+2", "dt-2")
        ]

        stats = engine._support_stats(candidates[0], candidates)

        self.assertEqual(stats["support_key_count"], 4)
        self.assertEqual(stats["support_family_count"], 1)
        self.assertFalse(engine._is_supported_7seg_candidate(candidates[0], candidates))

    def test_suspicious_text_can_be_supported_by_independent_crop_families(self) -> None:
        candidates = [
            make_suspicious_candidate("12.3", crop="window_frame_trim6", variant="base"),
            make_suspicious_candidate("12.3", crop="window_low2", variant="dt-2"),
        ]

        stats = engine._support_stats(candidates[0], candidates)

        self.assertEqual(stats["support_family_count"], 2)
        self.assertTrue(engine._is_supported_7seg_candidate(candidates[0], candidates))

    def test_hold_05p491_broad_support_beats_isolated_suspicious_crop(self) -> None:
        result = run_fast_fixture("test_sets/green_multimeter_v2/cropped/meter_hold_05p491.jpg")
        debug = result["debug"]

        self.assertEqual(result["text"], "05.491")
        self.assertTrue(debug.get("accepted_with_support"))
        self.assertGreaterEqual(debug.get("support_family_count", 0), 2)
        self.assertIn("135.49", {row.get("clean") for row in debug.get("candidate_summaries", [])})

    def test_hold_17p235_supported_suspicious_candidate_is_rescued(self) -> None:
        result = run_fast_fixture("test_sets/green_multimeter_v2/cropped/meter_hold_17p235.jpg")
        debug = result["debug"]

        self.assertEqual(result["text"], "17.235")
        self.assertIsNone(debug.get("rejected"))
        self.assertTrue(debug.get("accepted_with_support"))
        self.assertGreaterEqual(debug.get("support_family_count", 0), 2)


if __name__ == "__main__":
    unittest.main()
