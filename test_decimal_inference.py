from __future__ import annotations

import unittest
from unittest import mock

import ocr_engine
import ocr_pi


def make_candidate(text: str, *, raw: str = "[7seg:1:bc,2:abdeg]", score: float = 3.0) -> ocr_pi.Candidate:
    suspicious = ocr_pi._candidate_suspicious_tokens(text, raw)
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
        self.assertTrue(ocr_engine.is_valid_final_numeric_text("1.3445"))
        self.assertTrue(ocr_engine.is_valid_final_numeric_text("0.5027"))
        self.assertTrue(ocr_engine.is_valid_final_numeric_text("170.62"))

    def test_four_decimal_text_has_no_fraction_length_penalty(self) -> None:
        structural, penalties, _artifact = ocr_engine.structural_quality_score(
            "1.3445",
            "[7seg:1:bc,.:dot,3:abcd,4:bcfg,4:bcfg,5:acdfg]",
            None,
            "base",
        )

        self.assertEqual(ocr_engine.numeric_structure_penalty("1.3445"), 0.0)
        self.assertGreater(structural, 0.0)
        self.assertFalse(any(str(item).startswith("malformed_numeric") for item in penalties))

    def test_digit_only_aliases_do_not_infer_decimal_blindly(self) -> None:
        aliases = ocr_engine.generate_candidate_aliases(
            "12",
            "[7seg:1:bc,2:abdeg]",
            {"red_digit_display"},
        )

        self.assertFalse(any("." in alias_text for alias_text, _label, _quality in aliases))

    def test_four_decimal_aliases_do_not_trim_to_three_places(self) -> None:
        aliases = ocr_engine.generate_candidate_aliases(
            "1.3445",
            "[7seg:1:bc,.:dot,3:abcd,4:bcfg,4:bcfg,5:acdfg]",
        )

        self.assertFalse(any(label == "trim_fractional_three" for _text, label, _quality in aliases))
        self.assertFalse(any(alias_text == "1.344" for alias_text, _label, _quality in aliases))

    def test_selection_does_not_prefer_three_decimal_form_over_raw_four_decimal(self) -> None:
        def vote() -> dict:
            return {
                "final_score": 4.0,
                "best_score": 3.0,
                "score_sum": 3.0,
                "count": 1,
                "best_conf": 95.0,
                "family_count": 1,
                "visual_decimal_observed": True,
            }

        best_text, _best_info = ocr_engine.select_best_vote({
            "1.3445": vote(),
            "1.344": vote(),
        })

        self.assertEqual(best_text, "1.3445")

    def test_pi_7seg_candidate_accepts_four_decimal_text(self) -> None:
        candidate = ocr_pi._make_7seg_candidate(
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

    def test_pi_tesseract_candidate_accepts_four_decimal_text(self) -> None:
        with mock.patch.object(ocr_pi.core, "ocr_once", return_value=("0.5027", 82.0, "0.5027")):
            candidate = ocr_pi._make_tesseract_candidate(
                "selected_roi",
                "base",
                None,
                {"ocr_input": None, "ocr_input_mask": None},
                ocr_engine.Params(),
            )

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["text"], "0.5027")

    def test_tesseract_reliability_accepts_four_decimal_text(self) -> None:
        self.assertTrue(
            ocr_engine.is_reliable_tesseract_vote(
                "1.3445",
                {
                    "raw": "[tesseract:1.3445]",
                    "count": 2,
                    "best_conf": 82.0,
                    "family_count": 2,
                    "final_score": 3.0,
                    "best_score": 3.0,
                    "sample_label": "selected_roi/base",
                    "sample_stages": {},
                },
            )
        )

    def test_mask_dot_inferred_decimal_keeps_visual_decimal_evidence(self) -> None:
        def vote(decimal_kind: str) -> dict:
            return {
                "count": 1,
                "score_sum": 1.6,
                "best_score": 1.6,
                "best_conf": 57.0,
                "raw": "[tesseract:16979]",
                "sample_raw": "[tesseract:16979]",
                "families": {"selected"},
                "sources": {"inferred_decimal"},
                "decimal_evidence": {decimal_kind},
                "candidate_details": [
                    {
                        "family": "selected",
                        "source": "inferred_decimal",
                        "quality_score": 1.6,
                        "structural_quality": 0.78,
                        "artifact_penalty": 0.0,
                        "penalties": ["inferred_decimal_source:0.22"],
                        "decimal_evidence": decimal_kind,
                    }
                ],
            }

        decimal_kind = ocr_engine.decimal_evidence_kind(
            "1.6979",
            "[tesseract:16979]",
            "noise_safe|inferred_decimal_mask_dot_1",
            True,
        )
        self.assertEqual(decimal_kind, "visual_mask_dot")

        visual_votes = {"1.6979": vote(decimal_kind)}
        inferred_only_votes = {"1.6979": vote("inferred_decimal")}
        ocr_engine.apply_combined_vote_scores(visual_votes)
        ocr_engine.apply_combined_vote_scores(inferred_only_votes)

        visual_info = visual_votes["1.6979"]
        inferred_only_info = inferred_only_votes["1.6979"]
        self.assertIn("visual_mask_dot", visual_info["decimal_evidence"])
        self.assertTrue(visual_info["visual_decimal_observed"])
        self.assertIn("inferred_decimal_only:0.45", visual_info["penalties_applied"])
        self.assertGreater(visual_info["final_score"], inferred_only_info["final_score"])

    def test_final_selection_falls_back_after_rejected_top_candidate(self) -> None:
        def vote(text: str, final_score: float) -> dict:
            return {
                "count": 5,
                "score_sum": final_score,
                "best_score": final_score,
                "best_conf": 57.0,
                "raw": "[tesseract:16979]",
                "sample_raw": "[tesseract:16979]",
                "sample_label": "selected_roi/noise_safe",
                "sample_stages": {},
                "family_count": 2,
                "families": {"selected", "window"},
                "sources": {"tesseract"} if "." not in text else {"inferred_decimal"},
                "final_score": final_score,
            }

        selected_text, _selected_info, rejected = ocr_engine.select_best_reliable_vote(
            {
                "16979": vote("16979", 5.0),
                "1.6979": vote("1.6979", 4.0),
            }
        )

        self.assertEqual(selected_text, "1.6979")
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["clean"], "16979")
        self.assertEqual(rejected[0]["reason"], "unreliable_tesseract")

    def test_digit_only_candidates_are_normal_candidates(self) -> None:
        candidate = make_candidate("12")
        support = make_candidate("12", raw="[7seg:1:bc,2:abdeg]", score=2.9)
        support["crop"] = "window_frame_trim4"

        self.assertNotIn("missing_decimal", candidate["suspicious_tokens"])
        self.assertTrue(ocr_pi._is_reliable_7seg_candidate(candidate))
        self.assertTrue(ocr_pi._is_supported_7seg_candidate(candidate, [candidate, support]))
        self.assertIs(ocr_pi._select_acceptable_7seg_candidate([candidate, support]), candidate)

    def test_inferred_decimal_candidate_is_marked_and_lower_confidence(self) -> None:
        source = make_candidate("12")
        inferred = ocr_pi._make_inferred_decimal_candidate(
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
        self.assertTrue(ocr_pi._is_reliable_7seg_candidate(inferred))

    def test_competing_decimal_adds_inferred_candidate(self) -> None:
        digit_only = make_candidate("12", score=3.0)
        decimal = make_candidate("1.2", raw="[7seg:1:bc,.:dot,2:abdeg]", score=4.0)
        decimal["suspicious_tokens"] = ocr_pi._candidate_suspicious_tokens(decimal["text"], decimal["raw"])
        candidates = [digit_only, decimal]

        ocr_pi._add_competing_decimal_inferred_candidates(candidates)

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

        ocr_pi._add_competing_decimal_inferred_candidates(candidates)

        self.assertEqual(len(candidates), 1)
        self.assertFalse(any(candidate.get("source") == "inferred_decimal" for candidate in candidates))


if __name__ == "__main__":
    unittest.main()
