from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import ocr_engine as core


Params = core.Params
Candidate = Dict[str, Any]

FINAL_NUMERIC_RE = re.compile(r"^\d+(?:\.\d+)?$")
ALIAS_VARIANT_MARKERS = (
    "leading_zero_decimal",
    "single_digit_decimal",
    "append_trailing_narrow1",
    "trim_trailing_narrow1",
    "trim_trailing_narrow7",
    "fractional_7_from_smeared_top",
    "mask_dot_",
    "dot_pos_",
)
STRONG_SUSPICIOUS_TOKENS = ("narrow-smear1", "trailing_narrow1", "trailing_narrow7")
SOFT_SUSPICIOUS_TOKENS = ("weak", "smear", "clipped", "lowd", "soft", "loose", "abc-narrow")


def normalize_numeric_text(text: str) -> str:
    cleaned = re.sub(r"[^0-9.]", "", text)
    if not cleaned:
        return ""

    if cleaned.count(".") > 1:
        head, tail = cleaned.split(".", 1)
        cleaned = head + "." + tail.replace(".", "")

    if cleaned == ".":
        return ""

    return cleaned


def is_valid_final_numeric_text(text: str) -> bool:
    return bool(FINAL_NUMERIC_RE.fullmatch(text))


def numeric_structure_penalty(text: str) -> float:
    if not text:
        return -4.0
    if text == ".":
        return -4.0
    if text.startswith(".") or text.endswith("."):
        return -2.25
    if text.count(".") > 1:
        return -2.50
    if "." in text:
        whole, fraction = text.split(".", 1)
        if not whole or not fraction:
            return -2.25
    return 0.0


def candidate_source(raw: str, variant_name: str) -> str:
    if "inferred_decimal" in variant_name:
        return "inferred_decimal"
    if any(marker in variant_name for marker in ALIAS_VARIANT_MARKERS):
        return "alias"
    if raw.startswith("[7seg:"):
        return "7seg"
    if raw.startswith("[tesseract:"):
        return "tesseract"
    return "unknown"


def decimal_evidence_kind(text: str, raw: str, variant_name: str, decimal_observed: bool) -> str:
    if ".:dot" in raw:
        return "visual_dot"
    if "mask_dot_" in variant_name:
        return "visual_mask_dot"
    if "inferred_decimal" in variant_name and "." in text:
        return "inferred_decimal"
    if any(marker in variant_name for marker in ALIAS_VARIANT_MARKERS) and "." in text:
        return "heuristic_alias"
    if raw.startswith("[tesseract:") and "." in raw:
        return "tesseract_dot"
    if decimal_observed:
        return "observed"
    return "none"


def should_strip_trailing_narrow_one(text: str, raw: str) -> bool:
    if not raw.startswith("[7seg:") or not raw.endswith("1:narrow]") or not text.endswith("1"):
        return False

    if "." in text:
        return False

    digit_count = sum(ch.isdigit() for ch in text)
    return digit_count >= 4


def should_strip_trailing_narrow_seven(text: str, raw: str) -> bool:
    if not raw.startswith("[7seg:") or not raw.endswith("7:abc-narrow]") or not text.endswith("7"):
        return False

    if "." not in text:
        return False

    return False


def generate_candidate_aliases(
    text: str,
    raw: str,
    alias_hints: Optional[set[str]] = None,
) -> List[Tuple[str, str, float]]:
    aliases: List[Tuple[str, str, float]] = []
    alias_hints = alias_hints or set()

    if "." in text:
        whole, fraction = text.split(".", 1)
        if "right_unknown_narrow" in alias_hints and whole == "0" and fraction == "0":
            aliases.append((f"{whole}.{fraction}1", "append_trailing_narrow1", 0.55))
        if (
            len(fraction) == 2
            and fraction == "10"
            and "1:narrow-smear1,0:left-clipped0" in raw
            and ".:dot" in raw
        ):
            aliases.append((f"{whole}.70", "fractional_7_from_smeared_top", 0.65))

    if should_strip_trailing_narrow_one(text, raw):
        alias = text[:-1]
        if alias and not alias.endswith("."):
            aliases.append((alias, "trim_trailing_narrow1", 0.75))

    if should_strip_trailing_narrow_seven(text, raw):
        alias = text[:-1]
        if alias and not alias.endswith("."):
            aliases.append((alias, "trim_trailing_narrow7", 0.60))

    return aliases


def infer_decimal_text_from_slot(text: str, dot_slot: Optional[int]) -> Optional[str]:
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits or "." in text:
        return None
    if dot_slot is None or dot_slot <= 0 or dot_slot >= len(digits):
        return None
    inferred = f"{digits[:dot_slot]}.{digits[dot_slot:]}"
    if inferred == text or not is_valid_final_numeric_text(inferred):
        return None
    return inferred


def estimate_digit_run_count(mask: Optional[np.ndarray]) -> int:
    if mask is None or cv2.countNonZero(mask) == 0:
        return 0

    img_h = mask.shape[0]
    count = 0
    for x1, x2 in core.active_column_runs(mask):
        column_slice = mask[:, x1:x2]
        points = cv2.findNonZero(column_slice)
        if points is None:
            continue
        _, _y, _w, h = cv2.boundingRect(points)
        if h < img_h * 0.25:
            continue
        count += 1
    return count


def mask_border_artifact_penalties(mask: Optional[np.ndarray]) -> Tuple[float, List[str]]:
    if mask is None or mask.size == 0:
        return 0.0, []

    working = mask
    if len(working.shape) == 3:
        working = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    _, working = cv2.threshold(working, 0, 255, cv2.THRESH_BINARY)
    if cv2.countNonZero(working) > (working.shape[0] * working.shape[1] / 2):
        working = cv2.bitwise_not(working)

    img_h, img_w = working.shape[:2]
    img_area = max(1, img_h * img_w)
    edge_x = max(2, int(img_w * 0.025))
    edge_y = max(2, int(img_h * 0.025))
    near_right_x = img_w - max(3, int(img_w * 0.045))

    penalty = 0.0
    labels: List[str] = []
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(working, 8)
    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])
        if area <= 0:
            continue

        touches_right = x + w >= img_w - edge_x
        near_right = x + w >= near_right_x
        touches_any = (
            x <= edge_x
            or y <= edge_y
            or touches_right
            or y + h >= img_h - edge_y
        )
        tall = h >= img_h * 0.42
        wide = w >= img_w * 0.55
        large = area >= img_area * 0.018
        strip_like = w <= img_w * 0.12 and h >= img_h * 0.35

        if near_right and (tall or large):
            value = 0.45
            if touches_right and strip_like:
                value += 0.30
            if h >= img_h * 0.65:
                value += 0.20
            penalty += value
            labels.append(f"right_border_artifact:{value:.2f}")
        elif touches_any and large:
            value = 0.22
            if wide and (y <= edge_y or y + h >= img_h - edge_y):
                value += 0.18
            penalty += value
            labels.append(f"border_artifact:{value:.2f}")

    return min(penalty, 1.60), labels


def structural_quality_score(
    text: str,
    raw: str,
    mask: Optional[np.ndarray],
    variant_name: str,
) -> Tuple[float, List[str], float]:
    score = 1.0
    penalties: List[str] = []

    numeric_penalty = numeric_structure_penalty(text)
    if numeric_penalty < 0:
        value = min(1.0, abs(numeric_penalty) / 3.0)
        score -= value
        penalties.append(f"malformed_numeric:{value:.2f}")

    token_penalties = (
        ("loose", 0.22),
        ("soft", 0.08),
        ("smear", 0.16),
        ("lowd", 0.12),
        ("clipped", 0.16),
        ("split0", 0.35),
        ("weak-top", 0.07),
        ("abc-narrow", 0.16),
    )
    for token, value in token_penalties:
        occurrences = raw.count(token)
        if occurrences:
            total = min(0.55, occurrences * value)
            score -= total
            penalties.append(f"{token}:{total:.2f}")

    source = candidate_source(raw, variant_name)
    if source == "alias":
        score -= 0.18
        penalties.append("alias_source:0.18")
    if source == "inferred_decimal":
        score -= 0.22
        penalties.append("inferred_decimal_source:0.22")
    if source == "tesseract":
        score -= 0.10
        penalties.append("tesseract_source:0.10")

    border_penalty, border_labels = mask_border_artifact_penalties(mask)
    if border_penalty and "right-clipped0" in raw:
        border_penalty *= 0.45
        border_labels = [f"discounted_{label}" for label in border_labels]
    if border_penalty:
        score -= border_penalty
        penalties.extend(border_labels)

    digit_count = sum(ch.isdigit() for ch in text)
    run_count = estimate_digit_run_count(mask)
    if run_count and digit_count:
        if run_count >= digit_count + 2:
            value = min(0.55, 0.18 * (run_count - digit_count))
            score -= value
            penalties.append(f"extra_digit_runs:{value:.2f}")
        elif digit_count >= run_count + 2:
            value = min(0.35, 0.12 * (digit_count - run_count))
            score -= value
            penalties.append(f"missing_digit_runs:{value:.2f}")

    if "." in text:
        evidence = decimal_evidence_kind(text, raw, variant_name, ".:dot" in raw or "mask_dot_" in variant_name)
        if evidence == "visual_dot":
            score += 0.12
        elif evidence == "visual_mask_dot":
            score += 0.06
        elif evidence == "heuristic_alias":
            score -= 0.10
            penalties.append("heuristic_decimal:0.10")
    elif ".:dot" in raw:
        score -= 0.35
        penalties.append("dropped_visual_decimal:0.35")

    return max(0.0, min(1.25, score)), penalties, border_penalty


def _value_from_text(text: str, debug: Any) -> tuple[float | None, Any]:
    if not text:
        return None, debug
    try:
        return float(text), debug
    except ValueError:
        debug_dict = debug if isinstance(debug, dict) else {}
        debug_dict["rejected"] = "float_conversion"
        return None, debug_dict


def _normalize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(result)
    conf = float(normalized.get("conf", normalized.get("confidence", 0.0)) or 0.0)
    debug = normalized.get("debug", {})
    source = debug.get("source", "") if isinstance(debug, dict) else ""

    normalized["conf"] = conf
    normalized["confidence"] = conf
    normalized["source"] = source
    normalized.setdefault("raw", "")
    normalized.setdefault("text", "")
    normalized.setdefault("debug", debug)
    normalized.setdefault("value", None)
    return normalized


def _result_from_ocr_tuple(text: str, conf: float, raw: str, debug: Any) -> Dict[str, Any]:
    value, debug = _value_from_text(text, debug)
    return _normalize_result(
        {
            "value": value,
            "text": text,
            "conf": conf,
            "raw": raw,
            "debug": debug,
        }
    )


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


def _digits_only(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit())


def _is_digit_only_reading(candidate: Candidate) -> bool:
    text = str(candidate.get("text", ""))
    return text.isdigit() and _digit_count(text) >= 2


def _should_defer_digit_only_early_exit(candidate: Candidate) -> bool:
    text = str(candidate.get("text", ""))
    if not _is_digit_only_reading(candidate):
        return False
    return (len(text) > 1 and text.startswith("0")) or len(text) >= 4


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
    if should_strip_trailing_narrow_one(text, raw) and "trailing_narrow1" not in tokens:
        tokens.append("trailing_narrow1")
    if should_strip_trailing_narrow_seven(text, raw) and "trailing_narrow7" not in tokens:
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

    whole, fraction = text.split(".", 1)
    if fraction not in {"10", "01"}:
        return False

    tokens = candidate.get("suspicious_tokens", [])
    has_narrow_one = (
        "narrow-smear1" in tokens
        or "trailing_narrow1" in tokens
        or "1:narrow" in raw
    )
    has_bad_zero = any(token in raw for token in ("0:left-clipped0", "0:right-clipped0", "0:split0"))
    has_smeared_seven_competitor = any(
        str(other.get("text", "")) == f"{whole}.70"
        and str(other.get("source", "")) == "alias"
        and "fractional_7_from_smeared_top" in str(other.get("variant", ""))
        for other in candidates
    )
    has_weak_zero_with_smeared_seven_competitor = (
        "0:weak-top0" in raw and has_smeared_seven_competitor
    )
    if not (has_narrow_one and (has_bad_zero or has_weak_zero_with_smeared_seven_competitor)):
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
        if is_valid_final_numeric_text(str(other.get("text", ""))):
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
    structural, penalties, artifact_penalty = structural_quality_score(text, raw, mask, variant_name)
    score = structural * 2.0 + min(max(conf, 0.0), 100.0) / 100.0 + min(_digit_count(text), 5) * 0.05
    if "." in text:
        score += 0.12 if ".:dot" in raw else 0.02
    if "inferred_decimal" in variant_name:
        score -= 0.16
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
    conf = float(candidate.get("conf", 0.0))
    structural = float(candidate.get("structural_quality", 0.0))
    source = str(candidate.get("source", ""))

    if not is_valid_final_numeric_text(text):
        return False
    if _has_strong_suspicion(candidate):
        return False
    if source == "inferred_decimal":
        reason = str(candidate.get("reason", ""))
        return "." in text and reason in {"mask_dot", "competing_decimal"} and conf >= 80.0 and structural >= 0.70
    if conf >= 95.0 and structural >= 0.85 and "right_border_artifact" not in ",".join(candidate.get("penalties", [])):
        return True
    if conf >= 85.0 and structural >= 0.72:
        soft_count = len(candidate.get("suspicious_tokens", []))
        return soft_count <= 2 or structural >= 0.90
    return False


def _is_supported_7seg_candidate(candidate: Candidate, candidates: List[Candidate]) -> bool:
    if _is_reliable_7seg_candidate(candidate):
        return True
    if not is_valid_final_numeric_text(str(candidate.get("text", ""))):
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
    if not is_valid_final_numeric_text(text):
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


def _make_inferred_decimal_candidate(
    source_candidate: Candidate,
    inferred_text: str,
    reason: str,
    quality_adjust: float,
) -> Candidate:
    raw = str(source_candidate.get("raw", ""))
    variant_name = f"{source_candidate.get('variant', '')}|inferred_decimal_{reason}"
    source_conf = float(source_candidate.get("conf", 0.0))
    conf = min(max(source_conf - 8.0, 0.0), 88.0)
    suspicious_tokens = _candidate_suspicious_tokens(inferred_text, raw)
    score, structural, penalties, artifact_penalty = _score_candidate(
        inferred_text,
        conf,
        raw,
        variant_name,
        source_candidate.get("mask"),
        source_bias=-0.18,
    )
    inferred_candidate = dict(source_candidate)
    inferred_candidate.update(
        {
            "text": inferred_text,
            "conf": conf,
            "source": "inferred_decimal",
            "reason": reason,
            "source_text": source_candidate.get("text", ""),
            "variant": variant_name,
            "score": score + quality_adjust,
            "structural_quality": structural,
            "artifact_penalty": artifact_penalty,
            "penalties": penalties,
            "suspicious_tokens": suspicious_tokens,
        }
    )
    return inferred_candidate


def _make_mask_dot_inferred_decimal_candidates(candidate: Candidate) -> List[Candidate]:
    text = str(candidate.get("text", ""))
    if "." in text or not text.isdigit():
        return []
    mask = candidate.get("mask")
    if mask is None:
        return []
    dot_slot = core.infer_dot_slot_from_mask(mask)
    inferred_text = infer_decimal_text_from_slot(text, dot_slot)
    if inferred_text is None:
        return []
    return [_make_inferred_decimal_candidate(candidate, inferred_text, "mask_dot", quality_adjust=-0.22)]


def _make_7seg_alias_candidates(candidate: Candidate, result: Dict[str, Any]) -> List[Candidate]:
    alias_hints = set(result.get("alias_hints", set()))
    aliases = generate_candidate_aliases(
        str(candidate.get("text", "")),
        str(candidate.get("raw", "")),
        alias_hints,
    )
    return [
        _make_alias_candidate(candidate, alias_text, alias_label, quality_adjust)
        for alias_text, alias_label, quality_adjust in aliases
        if is_valid_final_numeric_text(alias_text)
    ]


def _add_competing_decimal_inferred_candidates(candidates: List[Candidate]) -> None:
    decimal_by_digits: Dict[str, Candidate] = {}
    for candidate in sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True):
        text = str(candidate.get("text", ""))
        if "." not in text or not is_valid_final_numeric_text(text):
            continue
        digits = _digits_only(text)
        if digits:
            decimal_by_digits.setdefault(digits, candidate)

    if not decimal_by_digits:
        return

    existing = {
        (
            str(candidate.get("source", "")),
            str(candidate.get("source_text", "")),
            str(candidate.get("text", "")),
            str(candidate.get("reason", "")),
        )
        for candidate in candidates
    }
    additions: List[Candidate] = []
    for candidate in list(candidates):
        text = str(candidate.get("text", ""))
        if "." in text or not text.isdigit():
            continue
        inferred_text = str(decimal_by_digits.get(text, {}).get("text", ""))
        if not inferred_text:
            continue
        key = ("inferred_decimal", text, inferred_text, "competing_decimal")
        if key in existing:
            continue
        additions.append(
            _make_inferred_decimal_candidate(candidate, inferred_text, "competing_decimal", quality_adjust=-0.35)
        )
        existing.add(key)

    candidates.extend(additions)


def _candidate_summaries(candidates: List[Candidate]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rank, candidate in enumerate(sorted(candidates, key=lambda item: item["score"], reverse=True), 1):
        rows.append(
            {
                "rank": rank,
                "clean": candidate.get("text", ""),
                "raw": candidate.get("raw", ""),
                "source": candidate.get("source", ""),
                "reason": candidate.get("reason", ""),
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
        "reason": candidate.get("reason", ""),
        "source_text": candidate.get("source_text", ""),
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


def _primary_7seg_crop(lcd_roi_bgr: np.ndarray) -> Tuple[str, Optional[np.ndarray]]:
    red_candidates = core.build_red_digit_roi_candidates(lcd_roi_bgr)
    if red_candidates:
        return red_candidates[0]
    return "selected_roi", lcd_roi_bgr


def _primary_7seg_params(params: core.Params, crop_name: str) -> core.Params:
    primary_params = core.clone_params(params)
    primary_params.scale = 3 if crop_name == "red_digit_band" else 1
    primary_params.pad = 0
    return primary_params


def _primary_7seg_failure_debug(
    reason: str,
    elapsed_ms: float,
    candidate: Optional[Candidate] = None,
) -> Dict[str, Any]:
    debug: Dict[str, Any] = {
        "debug_labels": ["primary_7seg_failed"],
        "primary_7seg_status": "primary_7seg_failed",
        "primary_7seg_failure_reason": reason,
        "primary_7seg_elapsed_ms": elapsed_ms,
        "primary_7seg_attempted": True,
    }
    if candidate is not None:
        debug["primary_7seg_candidate"] = _candidate_summaries([candidate])[0]
    return debug


def _primary_7seg_rejection_reason(candidate: Candidate) -> str:
    if candidate.get("crop") != "red_digit_band" and _is_digit_only_reading(candidate):
        return "digit_only_primary_deferred"
    if not _is_reliable_7seg_candidate(candidate):
        return "candidate_not_reliable"
    if candidate.get("crop") == "red_digit_band" and _digit_count(str(candidate.get("text", ""))) < 4:
        return "red_band_incomplete"
    return ""


def _primary_7seg_attempt(lcd_roi_bgr: np.ndarray, params: core.Params) -> Tuple[Optional[Candidate], Dict[str, Any]]:
    start = time.perf_counter()
    try:
        crop_name, reading_roi = _primary_7seg_crop(lcd_roi_bgr)
        if reading_roi is None or reading_roi.size == 0:
            return None, _primary_7seg_failure_debug("no_primary_crop", (time.perf_counter() - start) * 1000.0)

        stages = core.preprocess(reading_roi, _primary_7seg_params(params, crop_name))
        result = core.read_7seg_from_stages_debug(stages)
        if result is None:
            return None, _primary_7seg_failure_debug("no_7seg_candidate", (time.perf_counter() - start) * 1000.0)

        candidate = _make_7seg_candidate(
            crop_name,
            "primary_7seg_base",
            reading_roi,
            stages,
            result,
            "primary_7seg",
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if candidate is None:
            return None, _primary_7seg_failure_debug("invalid_7seg_candidate", elapsed_ms)

        rejection_reason = _primary_7seg_rejection_reason(candidate)
        if rejection_reason:
            return None, _primary_7seg_failure_debug(rejection_reason, elapsed_ms, candidate)

        return candidate, {
            "debug_labels": ["primary_7seg_success"],
            "primary_7seg_status": "primary_7seg_success",
            "primary_7seg_elapsed_ms": elapsed_ms,
            "primary_7seg_attempted": True,
        }
    except Exception as exc:
        return None, _primary_7seg_failure_debug(
            f"primary_exception:{type(exc).__name__}",
            (time.perf_counter() - start) * 1000.0,
        )


def _merge_primary_fallback_debug(
    fallback_debug: Dict[str, Any],
    primary_debug: Dict[str, Any],
    fallback_elapsed_ms: float,
) -> Dict[str, Any]:
    debug = fallback_debug if isinstance(fallback_debug, dict) else {}
    labels = list(primary_debug.get("debug_labels", []))
    labels.append("fallback_existing_pipeline_used")
    debug["debug_labels"] = labels
    debug["pipeline_label"] = "fallback_existing_pipeline_used"
    debug["primary_7seg_status"] = primary_debug.get("primary_7seg_status", "primary_7seg_failed")
    debug["primary_7seg_failure_reason"] = primary_debug.get("primary_7seg_failure_reason", "")
    debug["primary_7seg_elapsed_ms"] = float(primary_debug.get("primary_7seg_elapsed_ms", 0.0))
    debug["primary_7seg_attempted"] = True
    if "primary_7seg_candidate" in primary_debug:
        debug["primary_7seg_candidate"] = primary_debug["primary_7seg_candidate"]
    debug["fallback_existing_pipeline_used"] = True
    debug["fallback_existing_pipeline_elapsed_ms"] = fallback_elapsed_ms
    return debug


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
            for inferred_candidate in _make_mask_dot_inferred_decimal_candidates(candidate):
                candidates.append(inferred_candidate)
                phase_candidates.append(inferred_candidate)
    return phase_candidates, attempts


def _fast_ocr_existing_pipeline(
    lcd_roi_bgr: np.ndarray,
    p: core.Params | None = None,
    expand_weak_7seg: bool = False,
) -> Tuple[str, float, str, Dict[str, Any]]:
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
            _add_competing_decimal_inferred_candidates(candidates)
            accepted = _select_acceptable_7seg_candidate(candidates, allow_supported_suspicious=False)
            if accepted is not None:
                if _should_defer_digit_only_early_exit(accepted):
                    continue
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
            _add_competing_decimal_inferred_candidates(candidates)
            accepted = _select_acceptable_7seg_candidate(candidates, allow_supported_suspicious=True)
            if accepted is not None:
                debug = _debug_from_candidate(accepted, candidates, attempts, tesseract_attempted=False)
                if _has_strong_suspicion(accepted):
                    debug["accepted_with_support"] = True
                return accepted["text"], float(accepted["conf"]), accepted["raw"], debug

    tesseract_attempted = False

    if candidates and any(_has_dangerous_fractional_narrow_pattern(candidate, candidates) for candidate in candidates):
        rejected = "dangerous_fractional_narrow_pattern"
    elif candidates and any(_has_strong_suspicion(candidate) for candidate in candidates):
        rejected = "suspicious_7seg_no_independent_support"
    else:
        rejected = "no_valid_7seg_or_tesseract"
    return _uncertain_from_candidates(candidates, attempts, tesseract_attempted, rejected)


def _fast_ocr_from_roi(
    roi_image: Any,
    params: Params,
) -> tuple[str, float, str, Dict[str, Any]]:
    primary_candidate, primary_debug = _primary_7seg_attempt(roi_image, params)
    if primary_candidate is not None:
        debug = _debug_from_candidate(primary_candidate, [primary_candidate], 1, tesseract_attempted=False)
        debug.update(primary_debug)
        debug["pipeline_label"] = "primary_7seg_success"
        debug["fallback_existing_pipeline_used"] = False
        return (
            str(primary_candidate["text"]),
            float(primary_candidate["conf"]),
            str(primary_candidate["raw"]),
            debug,
        )

    fallback_start = time.perf_counter()
    text, conf, raw, debug = _fast_ocr_existing_pipeline(
        roi_image,
        p=params,
        expand_weak_7seg=False,
    )
    fallback_elapsed_ms = (time.perf_counter() - fallback_start) * 1000.0
    return text, conf, raw, _merge_primary_fallback_debug(debug, primary_debug, fallback_elapsed_ms)


def fast_ocr_from_lcd_roi(
    roi_image: Any,
    params: Params | None = None,
) -> tuple[str, float, str, Dict[str, Any]]:
    return _fast_ocr_from_roi(roi_image, params or Params())


def read_number_from_roi(
    roi_image: Any,
    mode: str = "fast",
    params: Params | None = None,
) -> Dict[str, Any]:
    selected_mode = mode.strip().lower()
    p = params or Params()

    if selected_mode == "fast":
        text, conf, raw, debug = _fast_ocr_from_roi(roi_image, p)
        return _result_from_ocr_tuple(text, conf, raw, debug)

    if selected_mode == "full":
        text, conf, raw, debug = core.robust_ocr_from_lcd_roi(roi_image, p)
        return _result_from_ocr_tuple(text, conf, raw, debug)

    raise ValueError(f"Unsupported OCR engine mode: {mode!r}. Expected 'fast' or 'full'.")
