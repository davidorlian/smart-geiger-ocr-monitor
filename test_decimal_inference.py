from __future__ import annotations

import unittest

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
    def test_digit_only_aliases_do_not_infer_decimal_blindly(self) -> None:
        aliases = ocr_engine.generate_candidate_aliases(
            "12",
            "[7seg:1:bc,2:abdeg]",
            {"red_digit_display"},
        )

        self.assertFalse(any("." in alias_text for alias_text, _label, _quality in aliases))

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
