from __future__ import annotations

import unittest
from unittest import mock

import engine


def make_candidate(text: str, *, raw: str = "[7seg:1:bc,2:abdeg]", score: float = 3.0) -> engine.Candidate:
    suspicious = engine._candidate_suspicious_tokens(text, raw)
    return {
        "text": text,
        "conf": 95.0,
        "raw": raw,
        "source": "7seg",
        "crop": "selected_roi",
        "variant": "base",
        "phase": "fixed_roi",
        "stage": "after_filters",
        "score": score,
        "structural_quality": 0.95,
        "artifact_penalty": 0.0,
        "penalties": [],
        "suspicious_tokens": suspicious,
        "reading_roi": None,
        "stages": {},
        "mask": None,
    }


class DecimalInferenceTests(unittest.TestCase):
    def test_four_decimal_text_is_valid_final_numeric_text(self) -> None:
        self.assertTrue(engine.is_valid_final_numeric_text("1.3445"))
        self.assertTrue(engine.is_valid_final_numeric_text("0.5027"))
        self.assertTrue(engine.is_valid_final_numeric_text("170.62"))

    def test_four_decimal_text_has_no_fraction_length_penalty(self) -> None:
        structural, penalties, _artifact = engine.structural_quality_score(
            "1.3445",
            "[7seg:1:bc,.:dot,3:abcd,4:bcfg,4:bcfg,5:acdfg]",
            None,
            "base",
        )

        self.assertEqual(engine.numeric_structure_penalty("1.3445"), 0.0)
        self.assertGreater(structural, 0.0)
        self.assertFalse(any(str(item).startswith("malformed_numeric") for item in penalties))

    def test_digit_only_aliases_do_not_infer_decimal_blindly(self) -> None:
        aliases = engine.generate_candidate_aliases(
            "12",
            "[7seg:1:bc,2:abdeg]",
            {"red_digit_display"},
        )

        self.assertFalse(any("." in alias_text for alias_text, _label, _quality in aliases))

    def test_four_decimal_aliases_do_not_trim_to_three_places(self) -> None:
        aliases = engine.generate_candidate_aliases(
            "1.3445",
            "[7seg:1:bc,.:dot,3:abcd,4:bcfg,4:bcfg,5:acdfg]",
        )

        self.assertFalse(any(label == "trim_fractional_three" for _text, label, _quality in aliases))
        self.assertFalse(any(alias_text == "1.344" for alias_text, _label, _quality in aliases))

    def test_selection_does_not_prefer_three_decimal_form_over_raw_four_decimal(self) -> None:
        raw_four = engine._make_7seg_candidate(
            "selected_roi",
            "base",
            None,
            {},
            {
                "text": "1.3445",
                "raw": "[7seg:1:bc,.:dot,3:abcd,4:bcfg,4:bcfg,5:acdfg]",
                "conf": 95.0,
                "stage_name": "after_filters",
                "stage_mask": None,
            },
            "fixed_roi",
        )
        trimmed_three = engine._make_7seg_candidate(
            "selected_roi",
            "trim_fractional_three",
            None,
            {},
            {
                "text": "1.344",
                "raw": "[7seg:1:bc,.:dot,3:abcd,4:bcfg,4:bcfg]",
                "conf": 95.0,
                "stage_name": "after_filters",
                "stage_mask": None,
            },
            "fixed_roi",
        )

        self.assertIsNotNone(raw_four)
        self.assertIsNotNone(trimmed_three)
        best = engine._select_acceptable_7seg_candidate([trimmed_three, raw_four])

        self.assertIs(best, raw_four)

    def test_engine_7seg_candidate_accepts_four_decimal_text(self) -> None:
        candidate = engine._make_7seg_candidate(
            "selected_roi",
            "base",
            None,
            {},
            {
                "text": "1.3445",
                "raw": "[7seg:1:bc,.:dot,3:abcd,4:bcfg,4:bcfg,5:acdfg]",
                "conf": 95.0,
                "stage_name": "after_filters",
                "stage_mask": None,
            },
            "fixed_roi",
        )

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["text"], "1.3445")

    def test_engine_full_mode_surfaces_four_decimal_text_from_backend(self) -> None:
        with mock.patch.object(
            engine.core,
            "robust_ocr_from_lcd_roi",
            return_value=("0.5027", 82.0, "[tesseract:0.5027]", {"source": "tesseract"}),
        ):
            result = engine.read_number_from_roi(None, mode="full", params=engine.Params())

        self.assertEqual(result["text"], "0.5027")
        self.assertEqual(result["value"], 0.5027)
        self.assertEqual(result["source"], "tesseract")

    def test_fast_engine_does_not_expose_independent_tesseract_candidate_path(self) -> None:
        self.assertFalse(
            hasattr(engine, "_make_tesseract_candidate"),
            "fast engine should not have an independent Tesseract candidate constructor",
        )

    def test_mask_dot_inferred_decimal_keeps_visual_decimal_evidence(self) -> None:
        decimal_kind = engine.decimal_evidence_kind(
            "1.6979",
            "[tesseract:16979]",
            "noise_safe|inferred_decimal_mask_dot_1",
            True,
        )
        self.assertEqual(decimal_kind, "visual_mask_dot")

        visual_score, visual_structural, visual_penalties, _artifact = engine._score_candidate(
            "1.6979",
            57.0,
            "[tesseract:16979]",
            "noise_safe|inferred_decimal_mask_dot_1",
            None,
        )
        inferred_score, inferred_structural, inferred_penalties, _artifact = engine._score_candidate(
            "1.6979",
            57.0,
            "[tesseract:16979]",
            "noise_safe|inferred_decimal",
            None,
        )

        self.assertIn("inferred_decimal_source:0.22", visual_penalties)
        self.assertIn("inferred_decimal_source:0.22", inferred_penalties)
        self.assertGreater(visual_structural, inferred_structural)
        self.assertGreater(visual_score, inferred_score)

    def test_final_selection_falls_back_after_rejected_top_candidate(self) -> None:
        rejected_top = make_candidate(
            "1.1143",
            raw="[7seg:1:narrow-smear1,.:dot,1:narrow-smear1,1:narrow-smear1,4:bcfg,3:bcdg-3like]",
            score=9.0,
        )
        fallback = make_candidate(
            "1.6979",
            raw="[7seg:1:bc,.:dot,6:acdefg,9:abcdfg,7:abc,9:abcdfg]",
            score=4.0,
        )

        selected = engine._select_acceptable_7seg_candidate([rejected_top, fallback])

        self.assertIs(selected, fallback)

    def test_digit_only_candidates_are_normal_candidates(self) -> None:
        candidate = make_candidate("12")
        support = make_candidate("12", raw="[7seg:1:bc,2:abdeg]", score=2.9)
        support["crop"] = "window_frame_trim4"

        self.assertNotIn("missing_decimal", candidate["suspicious_tokens"])
        self.assertTrue(engine._is_reliable_7seg_candidate(candidate))
        self.assertTrue(engine._is_supported_7seg_candidate(candidate, [candidate, support]))
        self.assertIs(engine._select_acceptable_7seg_candidate([candidate, support]), candidate)

    def test_inferred_decimal_candidate_is_marked_and_lower_confidence(self) -> None:
        source = make_candidate("12")
        inferred = engine._make_inferred_decimal_candidate(
            source,
            "1.2",
            "competing_decimal",
            quality_adjust=-0.35,
        )

        self.assertEqual(inferred["source"], "inferred_decimal")
        self.assertEqual(inferred["reason"], "competing_decimal")
        self.assertEqual(inferred["source_text"], "12")
        self.assertLess(inferred["conf"], source["conf"])
        self.assertNotIn("missing_decimal", inferred["suspicious_tokens"])
        self.assertTrue(engine._is_reliable_7seg_candidate(inferred))

    def test_competing_decimal_adds_inferred_candidate(self) -> None:
        digit_only = make_candidate("12", score=3.0)
        decimal = make_candidate("1.2", raw="[7seg:1:bc,.:dot,2:abdeg]", score=4.0)
        decimal["suspicious_tokens"] = engine._candidate_suspicious_tokens(decimal["text"], decimal["raw"])
        candidates = [digit_only, decimal]

        engine._add_competing_decimal_inferred_candidates(candidates)

        inferred = [
            candidate
            for candidate in candidates
            if candidate.get("source") == "inferred_decimal"
            and candidate.get("reason") == "competing_decimal"
        ]
        self.assertEqual(len(inferred), 1)
        self.assertEqual(inferred[0]["text"], "1.2")
        self.assertEqual(inferred[0]["source_text"], "12")

    def test_no_inferred_decimal_without_supporting_evidence(self) -> None:
        digit_only = make_candidate("12", score=3.0)
        candidates = [digit_only]

        engine._add_competing_decimal_inferred_candidates(candidates)

        self.assertEqual(len(candidates), 1)
        self.assertFalse(any(candidate.get("source") == "inferred_decimal" for candidate in candidates))


if __name__ == "__main__":
    unittest.main()
