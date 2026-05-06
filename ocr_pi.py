from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import ocr_engine as core


Candidate = Dict[str, Any]
STRONG_SUSPICIOUS_TOKENS = ("narrow-smear1", "trailing_narrow1", "trailing_narrow7")
SOFT_SUSPICIOUS_TOKENS = ("weak", "smear", "clipped", "lowd", "soft", "loose", "abc-narrow")


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
        "raw_band_fallback": ["noise_safe_hi", "noise_safe", "dt-4", "base"],
    }


def _digit_count(text: str) -> int:
    return sum(ch.isdigit() for ch in text)


def _suspicious_tokens(raw: str) -> List[str]:
    tokens: List[str] = []
    for token in STRONG_SUSPICIOUS_TOKENS:
        if token in raw:
            tokens.append(token)
    for token in SOFT_SUSPICIOUS_TOKENS:
        if token == "smear" and any(strong in raw for strong in STRONG_SUSPICIOUS_TOKENS):
            continue
        if token in raw and token not in tokens:
            tokens.append(token)
    return tokens


def _candidate_suspicious_tokens(text: str, raw: str) -> List[str]:
    tokens = _suspicious_tokens(raw)
    if core.should_strip_trailing_narrow_one(text, raw) and "trailing_narrow1" not in tokens:
        tokens.append("trailing_narrow1")
    if core.should_strip_trailing_narrow_seven(text, raw) and "trailing_narrow7" not in tokens:
        tokens.append("trailing_narrow7")
    return tokens


def _has_strong_suspicion(candidate: Candidate) -> bool:
    tokens = candidate.get("suspicious_tokens", [])
    return any(token in tokens for token in STRONG_SUSPICIOUS_TOKENS)


def _same_text_independent(left: Candidate, right: Candidate) -> bool:
    if left is right or left.get("text") != right.get("text"):
        return False
    return (
        left.get("crop") != right.get("crop")
        or left.get("variant") != right.get("variant")
        or left.get("stage") != right.get("stage")
    )


def _same_text_candidates(candidate: Candidate, candidates: List[Candidate]) -> List[Candidate]:
    return [other for other in candidates if other.get("text") == candidate.get("text")]


def _independence_family(candidate: Candidate) -> str:
    # Keep frame-trim crop names separate here. On the Pi path, different trim
    # amounts are useful independent evidence even if the PC scorer groups them.
    return str(candidate.get("crop", ""))


def _support_stats(candidate: Candidate, candidates: List[Candidate]) -> Dict[str, int]:
    same_text = _same_text_candidates(candidate, candidates)
    distinct_keys = {
        (
            str(item.get("crop", "")),
            str(item.get("variant", "")),
            str(item.get("stage", "")),
        )
        for item in same_text
    }
    distinct_families = {_independence_family(item) for item in same_text}
    clean_same_text = [
        item
        for item in same_text
        if not _has_strong_suspicion(item)
    ]
    return {
        "same_text_count": len(same_text),
        "support_key_count": len(distinct_keys),
        "support_family_count": len(distinct_families),
        "clean_same_text_count": len(clean_same_text),
    }


def _support_counts(candidate: Candidate, candidates: List[Candidate]) -> Tuple[int, int]:
    clean_support = 0
    non_smear_support = 0
    for other in candidates:
        if not _same_text_independent(candidate, other):
            continue
        if _has_strong_suspicion(other):
            continue
        non_smear_support += 1
        if not other.get("suspicious_tokens"):
            clean_support += 1
    return clean_support, non_smear_support


def _has_clean_same_text_support(candidate: Candidate, candidates: List[Candidate]) -> bool:
    return _support_stats(candidate, candidates)["clean_same_text_count"] > 0


def _has_dangerous_fractional_narrow_pattern(candidate: Candidate, candidates: List[Candidate]) -> bool:
    text = str(candidate.get("text", ""))
    raw = str(candidate.get("raw", ""))
    if "." not in text:
        return False

    _whole, fraction = text.split(".", 1)
    if fraction not in {"10", "01"}:
        return False

    tokens = candidate.get("suspicious_tokens", [])
    has_narrow_one = (
        "narrow-smear1" in tokens
        or "trailing_narrow1" in tokens
        or "1:narrow" in raw
    )
    has_bad_zero = any(token in raw for token in ("0:left-clipped0", "0:right-clipped0", "0:split0"))
    if not (has_narrow_one and has_bad_zero):
        return False

    return not _has_clean_same_text_support(candidate, candidates)


def _has_clean_fractional_seven_competitor(candidate: Candidate, candidates: List[Candidate]) -> bool:
    text = str(candidate.get("text", ""))
    if "." not in text:
        return False
    whole, fraction = text.split(".", 1)
    if fraction != "1":
        return False

    competitor_text = f"{whole}.7"
    for other in candidates:
        if other.get("text") != competitor_text:
            continue
        if _has_strong_suspicion(other):
            continue
        if core.is_valid_final_numeric_text(str(other.get("text", ""))):
            return True
    return False


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
    suspicion = _candidate_suspicious_tokens(text, raw)
    if any(token in suspicion for token in STRONG_SUSPICIOUS_TOKENS):
        score -= 0.80
    elif suspicion:
        score -= min(0.25, 0.06 * len(suspicion))
    score += source_bias
    return score, structural, penalties, artifact_penalty


def _is_reliable_7seg_candidate(candidate: Candidate) -> bool:
    text = str(candidate.get("text", ""))
    raw = str(candidate.get("raw", ""))
    conf = float(candidate.get("conf", 0.0))
    structural = float(candidate.get("structural_quality", 0.0))

    if not core.is_valid_final_numeric_text(text):
        return False
    if _has_strong_suspicion(candidate):
        return False
    if conf >= 95.0 and structural >= 0.85 and "right_border_artifact" not in ",".join(candidate.get("penalties", [])):
        return True
    if conf >= 85.0 and structural >= 0.72:
        soft_count = len(candidate.get("suspicious_tokens", []))
        return soft_count <= 2 or structural >= 0.90
    return False


def _is_supported_7seg_candidate(candidate: Candidate, candidates: List[Candidate]) -> bool:
    if _is_reliable_7seg_candidate(candidate):
        return True
    if not core.is_valid_final_numeric_text(str(candidate.get("text", ""))):
        return False
    if _has_strong_suspicion(candidate):
        if _has_dangerous_fractional_narrow_pattern(candidate, candidates):
            return False
        if _has_clean_fractional_seven_competitor(candidate, candidates):
            return False
        stats = _support_stats(candidate, candidates)
        return stats["support_family_count"] >= 2 or stats["support_key_count"] >= 3
    return False


def _select_acceptable_7seg_candidate(
    candidates: List[Candidate],
    allow_supported_suspicious: bool = True,
) -> Optional[Candidate]:
    ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
    for candidate in ranked:
        if _is_reliable_7seg_candidate(candidate):
            return candidate
    if not allow_supported_suspicious:
        return None
    for candidate in ranked:
        if _is_supported_7seg_candidate(candidate, candidates):
            return candidate
    return None


def _is_reliable_tesseract_candidate(candidate: Candidate) -> bool:
    if candidate.get("source") != "tesseract":
        return False
    text = str(candidate.get("text", ""))
    conf = float(candidate.get("conf", 0.0))
    if not core.is_valid_final_numeric_text(text):
        return False
    digits = _digit_count(text)
    if conf < 40.0:
        return False
    if digits <= 1 and conf < 80.0:
        return False
    if digits >= 2 and conf >= 55.0:
        return True
    return conf >= 80.0


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
    suspicious_tokens = _candidate_suspicious_tokens(text, raw)
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
        "suspicious_tokens": suspicious_tokens,
        "reading_roi": reading_roi,
        "stages": stages,
        "mask": stage_mask,
    }


def _make_alias_candidate(
    source_candidate: Candidate,
    alias_text: str,
    alias_label: str,
    quality_adjust: float,
) -> Candidate:
    raw = str(source_candidate.get("raw", ""))
    variant_name = f"{source_candidate.get('variant', '')}|{alias_label}"
    suspicious_tokens = _candidate_suspicious_tokens(alias_text, raw)
    score, structural, penalties, artifact_penalty = _score_candidate(
        alias_text,
        float(source_candidate.get("conf", 0.0)),
        raw,
        variant_name,
        source_candidate.get("mask"),
        source_bias=-0.05,
    )
    alias_candidate = dict(source_candidate)
    alias_candidate.update(
        {
            "text": alias_text,
            "source": "alias",
            "variant": variant_name,
            "score": score + quality_adjust,
            "structural_quality": structural,
            "artifact_penalty": artifact_penalty,
            "penalties": penalties,
            "suspicious_tokens": suspicious_tokens,
        }
    )
    return alias_candidate


def _make_7seg_alias_candidates(candidate: Candidate, result: Dict[str, Any]) -> List[Candidate]:
    alias_hints = set(result.get("alias_hints", set()))
    aliases = core.generate_candidate_aliases(
        str(candidate.get("text", "")),
        str(candidate.get("raw", "")),
        alias_hints,
    )
    return [
        _make_alias_candidate(candidate, alias_text, alias_label, quality_adjust)
        for alias_text, alias_label, quality_adjust in aliases
        if core.is_valid_final_numeric_text(alias_text)
    ]


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
            "suspicious_tokens": [],
            "reading_roi": reading_roi,
            "stages": stages,
            "mask": stages.get("ocr_input_mask"),
        }

    if not text or not core.is_valid_final_numeric_text(text):
        return None

    raw = f"[tesseract:{raw_text}]"
    suspicious_tokens = _candidate_suspicious_tokens(text, raw)
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
        "suspicious_tokens": suspicious_tokens,
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
                "suspicious_tokens": list(candidate.get("suspicious_tokens", [])),
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
        "suspicious_tokens": list(candidate.get("suspicious_tokens", [])),
        "candidate_summaries": _candidate_summaries(candidates),
        "tesseract_attempted": tesseract_attempted,
    }
    clean_support, non_smear_support = _support_counts(candidate, candidates)
    support_stats = _support_stats(candidate, candidates)
    debug["clean_support_count"] = clean_support
    debug["non_smear_support_count"] = non_smear_support
    debug["same_text_count"] = support_stats["same_text_count"]
    debug["support_key_count"] = support_stats["support_key_count"]
    debug["support_family_count"] = support_stats["support_family_count"]
    debug["clean_same_text_count"] = support_stats["clean_same_text_count"]
    debug["dangerous_fractional_narrow_pattern"] = _has_dangerous_fractional_narrow_pattern(candidate, candidates)
    debug["clean_fractional_seven_competitor"] = _has_clean_fractional_seven_competitor(candidate, candidates)
    if rejected:
        debug["rejected"] = rejected
    return debug


def _uncertain_from_candidates(
    candidates: List[Candidate],
    attempts: int,
    tesseract_attempted: bool,
    rejected: str,
) -> Tuple[str, float, str, Dict[str, Any]]:
    if candidates:
        best = max(candidates, key=lambda item: item["score"])
        debug = _debug_from_candidate(best, candidates, attempts, tesseract_attempted, rejected=rejected)
        debug["uncertain"] = True
        return "", 0.0, str(best.get("raw", "")), debug

    debug: Dict[str, Any] = {
        "strategy": "pi_fast",
        "winner_label": "",
        "winner_stage": "",
        "winner_raw": "",
        "vote_count": 0,
        "attempt_count": attempts,
        "candidate_summaries": [],
        "tesseract_attempted": tesseract_attempted,
        "rejected": rejected,
        "uncertain": True,
    }
    return "", 0.0, "", debug


def _process_7seg_phase(
    phase_name: str,
    crop_names: List[str],
    variant_names: List[str],
    crops: Dict[str, np.ndarray],
    variants: Dict[str, core.Params],
    candidates: List[Candidate],
    seen: set[Tuple[str, str]],
    attempts: int,
) -> Tuple[List[Candidate], int]:
    phase_candidates: List[Candidate] = []
    for crop_name in _ordered_existing(crop_names, crops):
        for variant_name in _ordered_existing(variant_names, variants):
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
            for alias_candidate in _make_7seg_alias_candidates(candidate, result):
                candidates.append(alias_candidate)
                phase_candidates.append(alias_candidate)
    return phase_candidates, attempts


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
        phase_candidates, attempts = _process_7seg_phase(
            phase_name,
            crop_names,
            variant_phase_map.get(phase_name, ["base"]),
            crops,
            variants,
            candidates,
            seen,
            attempts,
        )

        if phase_candidates:
            accepted = _select_acceptable_7seg_candidate(candidates, allow_supported_suspicious=False)
            if accepted is not None:
                debug = _debug_from_candidate(accepted, candidates, attempts, tesseract_attempted=False)
                if _has_strong_suspicion(accepted):
                    debug["accepted_with_support"] = True
                return accepted["text"], float(accepted["conf"]), accepted["raw"], debug

    if candidates:
        accepted = _select_acceptable_7seg_candidate(candidates, allow_supported_suspicious=True)
        if accepted is not None:
            debug = _debug_from_candidate(accepted, candidates, attempts, tesseract_attempted=False)
            if _has_strong_suspicion(accepted):
                debug["accepted_with_support"] = True
            return accepted["text"], float(accepted["conf"]), accepted["raw"], debug
        if expand_weak_7seg:
            best = max(candidates, key=lambda item: item["score"])
            debug = _debug_from_candidate(best, candidates, attempts, tesseract_attempted=False)
            debug["accepted_unreliable_7seg"] = True
            return best["text"], float(best["conf"]), best["raw"], debug

    raw_band_crops = ["raw_band1", "raw_band2", "raw_band3"]
    if any(name in crops for name in raw_band_crops):
        phase_candidates, attempts = _process_7seg_phase(
            "raw_band_fallback",
            raw_band_crops,
            variant_phase_map["raw_band_fallback"],
            crops,
            variants,
            candidates,
            seen,
            attempts,
        )
        if phase_candidates:
            accepted = _select_acceptable_7seg_candidate(candidates, allow_supported_suspicious=True)
            if accepted is not None:
                debug = _debug_from_candidate(accepted, candidates, attempts, tesseract_attempted=False)
                if _has_strong_suspicion(accepted):
                    debug["accepted_with_support"] = True
                return accepted["text"], float(accepted["conf"]), accepted["raw"], debug

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
        reliable_tesseract = [
            candidate
            for candidate in sorted(tesseract_candidates, key=lambda item: item["score"], reverse=True)
            if _is_reliable_tesseract_candidate(candidate)
        ]
        if reliable_tesseract:
            best = reliable_tesseract[0]
            debug = _debug_from_candidate(best, candidates, attempts, tesseract_attempted=tesseract_attempted)
            return best["text"], float(best["conf"]), best["raw"], debug

    if candidates and any(_has_dangerous_fractional_narrow_pattern(candidate, candidates) for candidate in candidates):
        rejected = "dangerous_fractional_narrow_pattern"
    elif candidates and any(_has_strong_suspicion(candidate) for candidate in candidates):
        rejected = "suspicious_7seg_no_independent_support"
    elif tesseract_candidates:
        rejected = "unreliable_tesseract"
    else:
        rejected = "no_valid_7seg_or_tesseract"
    return _uncertain_from_candidates(candidates, attempts, tesseract_attempted, rejected)


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
