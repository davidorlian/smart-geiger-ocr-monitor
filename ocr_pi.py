from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import ocr_engine as core


Candidate = Dict[str, Any]


def _is_numeric_window(image: np.ndarray) -> bool:
    h, w = image.shape[:2]
    return h > 0 and (w / float(h)) >= 1.60


def _candidate_map(lcd_roi_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    candidates: Dict[str, np.ndarray] = {}
    for name, crop in core.build_reading_roi_candidates(lcd_roi_bgr):
        if crop is not None and crop.size > 0 and name not in candidates:
            candidates[name] = crop
    if "selected_roi" not in candidates:
        candidates["selected_roi"] = lcd_roi_bgr.copy()
    return candidates


def _variant_map(p: core.Params) -> Dict[str, core.Params]:
    variants: Dict[str, core.Params] = {}
    for name, params in core.build_param_variants(p):
        variants.setdefault(name, params)
    return variants


def _ordered_existing(names: List[str], available: Dict[str, Any]) -> List[str]:
    return [name for name in names if name in available]


def _crop_phases(lcd_roi_bgr: np.ndarray) -> List[Tuple[str, List[str]]]:
    if _is_numeric_window(lcd_roi_bgr):
        return [
            ("fixed_roi", ["selected_roi"]),
            ("fixed_roi_thresholds", ["selected_roi"]),
            ("trimmed_window", ["window_frame_trim4", "window_frame_trim6"]),
            ("trimmed_window_thresholds", ["window_frame_trim4", "window_frame_trim6"]),
            ("last_chance_window", ["window_full", "window_trim", "window_xtrim", "window_low"]),
        ]

    return [
        ("fixed_roi", ["selected_roi"]),
        ("reading_band", ["base"]),
        ("reading_band_thresholds", ["base", "wide", "tight"]),
        ("last_chance_lcd", ["selected_roi", "loose", "full_mid1", "full_mid2"]),
    ]


def _variant_phases() -> Dict[str, List[str]]:
    return {
        "fixed_roi": ["base"],
        "reading_band": ["base"],
        "trimmed_window": ["base"],
        "fixed_roi_thresholds": ["dt-4", "dt+2", "dt-2"],
        "reading_band_thresholds": ["dt-4", "dt+2", "dt-2"],
        "trimmed_window_thresholds": ["dt-4", "dt+2", "dt-2"],
        "last_chance_window": ["base", "dt-4"],
        "last_chance_lcd": ["base", "dt-4"],
    }


def _digit_count(text: str) -> int:
    return sum(ch.isdigit() for ch in text)


def _score_candidate(
    text: str,
    conf: float,
    raw: str,
    variant_name: str,
    mask: Optional[np.ndarray],
    source_bias: float = 0.0,
) -> Tuple[float, float, List[str], float]:
    structural, penalties, artifact_penalty = core.structural_quality_score(text, raw, mask, variant_name)
    score = structural * 2.0 + min(max(conf, 0.0), 100.0) / 100.0 + min(_digit_count(text), 5) * 0.05
    if "." in text:
        score += 0.12 if ".:dot" in raw else 0.02
    if artifact_penalty:
        score -= min(0.50, artifact_penalty * 0.20)
    score += source_bias
    return score, structural, penalties, artifact_penalty


def _is_reliable_7seg_candidate(candidate: Candidate) -> bool:
    text = str(candidate.get("text", ""))
    raw = str(candidate.get("raw", ""))
    conf = float(candidate.get("conf", 0.0))
    structural = float(candidate.get("structural_quality", 0.0))

    if not core.is_valid_final_numeric_text(text):
        return False
    if conf >= 95.0 and structural >= 0.85 and "right_border_artifact" not in ",".join(candidate.get("penalties", [])):
        return True
    if conf >= 85.0 and structural >= 0.72:
        ambiguous_tokens = ("soft", "loose", "smear", "clipped", "lowd", "narrow")
        return not any(token in raw for token in ambiguous_tokens)
    return False


def _make_7seg_candidate(
    crop_name: str,
    variant_name: str,
    reading_roi: np.ndarray,
    stages: Dict[str, Any],
    result: Dict[str, Any],
    phase_name: str,
) -> Optional[Candidate]:
    text = str(result.get("text", ""))
    raw = str(result.get("raw", ""))
    conf = float(result.get("conf", 0.0))
    if not core.is_valid_final_numeric_text(text):
        return None

    stage_mask = result.get("stage_mask")
    score, structural, penalties, artifact_penalty = _score_candidate(
        text,
        conf,
        raw,
        variant_name,
        stage_mask,
        source_bias=0.15,
    )
    return {
        "text": text,
        "conf": conf,
        "raw": raw,
        "source": "7seg",
        "crop": crop_name,
        "variant": variant_name,
        "phase": phase_name,
        "stage": str(result.get("stage_name", "")),
        "score": score,
        "structural_quality": structural,
        "artifact_penalty": artifact_penalty,
        "penalties": penalties,
        "reading_roi": reading_roi,
        "stages": stages,
        "mask": stage_mask,
    }


def _make_tesseract_candidate(
    crop_name: str,
    variant_name: str,
    reading_roi: np.ndarray,
    stages: Dict[str, Any],
    p_variant: core.Params,
) -> Optional[Candidate]:
    try:
        text, conf, raw_text = core.ocr_once(stages["ocr_input"], p_variant)
    except Exception as exc:
        return {
            "text": "",
            "conf": 0.0,
            "raw": f"[tesseract_error:{exc}]",
            "source": "tesseract_error",
            "crop": crop_name,
            "variant": variant_name,
            "phase": "tesseract_fallback",
            "stage": "ocr_input",
            "score": float("-inf"),
            "structural_quality": 0.0,
            "artifact_penalty": 0.0,
            "penalties": ["tesseract_error"],
            "reading_roi": reading_roi,
            "stages": stages,
            "mask": stages.get("ocr_input_mask"),
        }

    if not text or not core.is_valid_final_numeric_text(text):
        return None

    raw = f"[tesseract:{raw_text}]"
    score, structural, penalties, artifact_penalty = _score_candidate(
        text,
        float(conf),
        raw,
        variant_name,
        stages.get("ocr_input_mask"),
        source_bias=-0.20,
    )
    return {
        "text": text,
        "conf": float(conf),
        "raw": raw,
        "source": "tesseract",
        "crop": crop_name,
        "variant": variant_name,
        "phase": "tesseract_fallback",
        "stage": "ocr_input",
        "score": score,
        "structural_quality": structural,
        "artifact_penalty": artifact_penalty,
        "penalties": penalties,
        "reading_roi": reading_roi,
        "stages": stages,
        "mask": stages.get("ocr_input_mask"),
    }


def _candidate_summaries(candidates: List[Candidate]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rank, candidate in enumerate(sorted(candidates, key=lambda item: item["score"], reverse=True), 1):
        rows.append(
            {
                "rank": rank,
                "clean": candidate.get("text", ""),
                "raw": candidate.get("raw", ""),
                "source": candidate.get("source", ""),
                "crop": candidate.get("crop", ""),
                "best_variant": candidate.get("variant", ""),
                "stage": candidate.get("stage", ""),
                "phase": candidate.get("phase", ""),
                "best_conf": float(candidate.get("conf", 0.0)),
                "structural_quality": float(candidate.get("structural_quality", 0.0)),
                "artifact_penalty": float(candidate.get("artifact_penalty", 0.0)),
                "final_score": float(candidate.get("score", 0.0)),
                "penalties": list(candidate.get("penalties", [])),
                "vote_count": 1,
            }
        )
    return rows


def _debug_from_candidate(
    candidate: Candidate,
    candidates: List[Candidate],
    attempts: int,
    tesseract_attempted: bool,
    rejected: str = "",
) -> Dict[str, Any]:
    debug = {
        "strategy": "pi_fast",
        "winner_label": f"{candidate.get('crop', '')}/{candidate.get('variant', '')}",
        "winner_crop": candidate.get("reading_roi"),
        "winner_stages": candidate.get("stages"),
        "winner_raw": candidate.get("raw", ""),
        "winner_stage": candidate.get("stage", ""),
        "winner_mask": candidate.get("mask"),
        "vote_count": 1,
        "attempt_count": attempts,
        "phase": candidate.get("phase", ""),
        "source": candidate.get("source", ""),
        "final_score": float(candidate.get("score", 0.0)),
        "structural_quality": float(candidate.get("structural_quality", 0.0)),
        "artifact_penalty": float(candidate.get("artifact_penalty", 0.0)),
        "penalties_applied": list(candidate.get("penalties", [])),
        "candidate_summaries": _candidate_summaries(candidates),
        "tesseract_attempted": tesseract_attempted,
    }
    if rejected:
        debug["rejected"] = rejected
    return debug


def fast_ocr_from_lcd_roi(
    lcd_roi_bgr: np.ndarray,
    p: core.Params | None = None,
    allow_tesseract_fallback: bool = True,
    expand_weak_7seg: bool = False,
) -> Tuple[str, float, str, Dict[str, Any]]:
    """Lightweight Raspberry Pi OCR strategy.

    The Pi path is intentionally fast-first:
    fixed ROI and base parameters first, a few 7-segment-only expansions on
    failure, and Tesseract only when no valid 7-segment result exists. Set
    expand_weak_7seg=True for debugging when you want cautious expansion after
    weak-but-valid 7-segment reads.
    """
    params = p or core.Params()
    crops = _candidate_map(lcd_roi_bgr)
    variants = _variant_map(params)
    variant_phase_map = _variant_phases()
    crop_phases = _crop_phases(lcd_roi_bgr)
    candidates: List[Candidate] = []
    attempts = 0
    seen: set[Tuple[str, str]] = set()

    for phase_name, crop_names in crop_phases:
        phase_candidates: List[Candidate] = []
        for crop_name in _ordered_existing(crop_names, crops):
            for variant_name in _ordered_existing(variant_phase_map.get(phase_name, ["base"]), variants):
                key = (crop_name, variant_name)
                if key in seen:
                    continue
                seen.add(key)
                attempts += 1
                reading_roi = crops[crop_name]
                p_variant = variants[variant_name]
                stages = core.preprocess(reading_roi, p_variant)
                result = core.read_7seg_from_stages_debug(stages)
                if result is None:
                    continue
                candidate = _make_7seg_candidate(crop_name, variant_name, reading_roi, stages, result, phase_name)
                if candidate is None:
                    continue
                candidates.append(candidate)
                phase_candidates.append(candidate)

        if phase_candidates:
            best_phase = max(phase_candidates, key=lambda item: item["score"])
            if _is_reliable_7seg_candidate(best_phase):
                debug = _debug_from_candidate(best_phase, candidates, attempts, tesseract_attempted=False)
                return best_phase["text"], float(best_phase["conf"]), best_phase["raw"], debug
            if not expand_weak_7seg:
                debug = _debug_from_candidate(best_phase, candidates, attempts, tesseract_attempted=False)
                debug["accepted_unreliable_7seg"] = True
                return best_phase["text"], float(best_phase["conf"]), best_phase["raw"], debug

    if candidates:
        best = max(candidates, key=lambda item: item["score"])
        debug = _debug_from_candidate(best, candidates, attempts, tesseract_attempted=False)
        debug["accepted_unreliable_7seg"] = not _is_reliable_7seg_candidate(best)
        return best["text"], float(best["conf"]), best["raw"], debug

    tesseract_candidates: List[Candidate] = []
    tesseract_attempted = False
    if allow_tesseract_fallback:
        tesseract_attempted = True
        fallback_crop_names = ["selected_roi"]
        fallback_crop_names.extend(["base", "window_frame_trim4", "window_frame_trim6"])
        for crop_name in _ordered_existing(fallback_crop_names, crops):
            for variant_name in _ordered_existing(["base", "dt-4"], variants):
                reading_roi = crops[crop_name]
                p_variant = variants[variant_name]
                stages = core.preprocess(reading_roi, p_variant)
                attempts += 1
                candidate = _make_tesseract_candidate(crop_name, variant_name, reading_roi, stages, p_variant)
                if candidate is None:
                    continue
                if candidate.get("source") == "tesseract_error":
                    candidates.append(candidate)
                    continue
                candidates.append(candidate)
                tesseract_candidates.append(candidate)

    if tesseract_candidates:
        best = max(tesseract_candidates, key=lambda item: item["score"])
        debug = _debug_from_candidate(best, candidates, attempts, tesseract_attempted=tesseract_attempted)
        return best["text"], float(best["conf"]), best["raw"], debug

    debug: Dict[str, Any] = {
        "strategy": "pi_fast",
        "winner_label": "",
        "winner_stage": "",
        "winner_raw": "",
        "vote_count": 0,
        "attempt_count": attempts,
        "candidate_summaries": _candidate_summaries(candidates),
        "tesseract_attempted": tesseract_attempted,
        "rejected": "no_valid_7seg_or_tesseract",
    }
    return "", 0.0, "", debug


def read_number_from_lcd_roi(
    lcd_roi_bgr: np.ndarray,
    p: core.Params | None = None,
    allow_tesseract_fallback: bool = True,
) -> Dict[str, Any]:
    text, conf, raw, debug = fast_ocr_from_lcd_roi(
        lcd_roi_bgr,
        p=p,
        allow_tesseract_fallback=allow_tesseract_fallback,
    )
    if not text:
        return {"value": None, "text": text, "conf": conf, "raw": raw, "debug": debug}

    try:
        value = float(text)
    except ValueError:
        debug = debug if isinstance(debug, dict) else {}
        debug["rejected"] = "float_conversion"
        return {"value": None, "text": text, "conf": conf, "raw": raw, "debug": debug}

    return {"value": value, "text": text, "conf": conf, "raw": raw, "debug": debug}
