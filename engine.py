from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import ocr_engine as core


@dataclass
class Params:
    # Basic
    method: int = 2              # 0=Otsu, 1=Adaptive, 2=Dark pixels
    scale: int = 3               # 1..6
    blur: int = 0                # 0=None, 1=Gaussian(3), 2=Gaussian(5)
    dark_threshold: int = 57     # 0..255, used if method=2

    # Adaptive (only if method=1)
    adaptive_block: int = 31     # odd >=3
    adaptive_c: int = 12         # 0..50

    # 7-seg bridge (fixed kernel size, minimal controls)
    close_enable: int = 1        # 0/1
    close_iter: int = 1          # 0..3
    close_k: int = 7             # 1..31

    # Optional fine tweaks
    dilate_iter: int = 0         # 0..2
    erode_iter: int = 0          # 0..2
    median: int = 0              # 0 or 3

    # OCR
    psm_mode: int = 0            # 0->7, 1->8, 2->13
    dpi: int = 300
    pad: int = 20                # 0..80


Candidate = Dict[str, Any]

LCD_READING_REGION = (0.30, 0.25, 0.98, 0.92)
READING_REGION_VARIANTS = (
    ("base", (0.30, 0.25, 0.98, 0.92)),
    ("wide", (0.24, 0.20, 0.98, 0.95)),
    ("loose", (0.18, 0.16, 0.98, 0.98)),
    ("tight", (0.36, 0.25, 0.98, 0.90)),
    ("full_mid1", (0.00, 0.12, 1.00, 0.95)),
    ("full_mid2", (0.00, 0.18, 1.00, 0.92)),
)
FULL_ROI_READING_REGION_VARIANTS = (
    ("raw_band1", (0.00, 0.00, 0.90, 0.85)),
    ("raw_band2", (0.00, 0.00, 0.95, 0.90)),
    ("raw_band3", (0.00, 0.05, 0.95, 0.92)),
)
WINDOW_READING_REGION_VARIANTS = (
    ("window_full", (0.00, 0.00, 1.00, 1.00)),
    ("window_frame_trim4", (0.04, 0.04, 0.96, 0.96)),
    ("window_frame_trim6", (0.06, 0.06, 0.94, 0.94)),
    ("window_trim", (0.00, 0.02, 1.00, 0.98)),
    ("window_xtrim", (0.02, 0.00, 0.98, 1.00)),
    ("window_low", (0.00, 0.05, 1.00, 0.92)),
    ("window_low2", (0.00, 0.08, 1.00, 0.88)),
)

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
    for x1, x2 in active_column_runs(mask):
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


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


SEGMENT_ZONES = {
    # label: left, top, right, bottom inside one digit box.
    "a": (0.18, 0.00, 0.82, 0.20),
    "b": (0.66, 0.15, 1.00, 0.45),
    "c": (0.66, 0.55, 1.00, 0.85),
    "d": (0.18, 0.80, 0.82, 1.00),
    "e": (0.00, 0.55, 0.34, 0.85),
    "f": (0.00, 0.15, 0.34, 0.45),
    "g": (0.18, 0.40, 0.82, 0.62),
}

SEGMENT_DIGITS = {
    frozenset("abcdef"): "0",
    frozenset("bc"): "1",
    frozenset("abdeg"): "2",
    frozenset("abcdg"): "3",
    frozenset("bcfg"): "4",
    frozenset("acdfg"): "5",
    frozenset("acdefg"): "6",
    frozenset("abc"): "7",
    frozenset("abcdefg"): "8",
    frozenset("abcdfg"): "9",
}

SEGMENT_ACTIVE_THRESHOLDS = {
    "a": 0.20,
    "b": 0.25,
    "c": 0.25,
    "d": 0.40,
    "e": 0.25,
    "f": 0.25,
    "g": 0.20,
}

DIGIT_TO_SEGMENTS = {
    digit: segments
    for segments, digit in SEGMENT_DIGITS.items()
}
TEMPLATE_DIGIT_SIZE = (120, 72)


def render_digit_template(segments: frozenset[str]) -> np.ndarray:
    height, width = TEMPLATE_DIGIT_SIZE
    mask = np.zeros((height, width), dtype=np.uint8)
    for label in segments:
        left, top, right, bottom = SEGMENT_ZONES[label]
        x1 = clamp(int(width * left), 0, width - 1)
        y1 = clamp(int(height * top), 0, height - 1)
        x2 = clamp(int(width * right), x1 + 1, width)
        y2 = clamp(int(height * bottom), y1 + 1, height)
        cv2.rectangle(mask, (x1, y1), (x2 - 1, y2 - 1), 255, thickness=-1)
    return mask


DIGIT_TEMPLATES = {
    digit: render_digit_template(segments)
    for digit, segments in DIGIT_TO_SEGMENTS.items()
}


def mask_iou_score(left: np.ndarray, right: np.ndarray) -> float:
    intersection = cv2.countNonZero(cv2.bitwise_and(left, right))
    union = cv2.countNonZero(cv2.bitwise_or(left, right))
    return intersection / float(max(1, union))


def digit_template_score(digit_mask: np.ndarray, digit: str, ratios: Dict[str, float], aspect_ratio: float) -> float:
    template = DIGIT_TEMPLATES[digit]
    resized = cv2.resize(
        digit_mask,
        (template.shape[1], template.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    _, resized = cv2.threshold(resized, 1, 255, cv2.THRESH_BINARY)

    segments = DIGIT_TO_SEGMENTS[digit]
    segment_similarity = sum(
        ratios[label] if label in segments else (1.0 - ratios[label])
        for label in SEGMENT_ZONES
    ) / float(len(SEGMENT_ZONES))
    score = mask_iou_score(resized, template) * 0.62 + segment_similarity * 0.38

    if digit == "1":
        if aspect_ratio > 0.48:
            score -= 0.35
    elif aspect_ratio < 0.13:
        score -= 0.40

    if digit == "0" and ratios["g"] > 0.34:
        score -= 0.08
    if digit == "8" and ratios["g"] < 0.18:
        score -= 0.10
    if digit in {"2", "3", "5"} and ratios["g"] < 0.16:
        score -= 0.12
    if digit == "7" and max(ratios["d"], ratios["e"], ratios["f"], ratios["g"]) > 0.30:
        score -= 0.15

    return score


def best_soft_digit_match(
    digit_mask: np.ndarray,
    ratios: Dict[str, float],
    aspect_ratio: float,
) -> Optional[Dict[str, Any]]:
    scored: List[Tuple[float, str]] = []
    for digit in sorted(DIGIT_TO_SEGMENTS):
        scored.append((digit_template_score(digit_mask, digit, ratios, aspect_ratio), digit))

    scored.sort(reverse=True)
    if not scored:
        return None

    best_score, best_digit = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else float("-inf")
    if best_score < 0.58:
        return None
    if best_score - second_score < 0.045:
        return None

    return {
        "digit": best_digit,
        "label": f"soft{best_digit}",
        "score": best_score,
        "margin": best_score - second_score,
    }


def clean_7seg_mask(mask):
    def longest_active_run(line: np.ndarray) -> int:
        best = 0
        current = 0
        for is_active in line:
            if is_active:
                current += 1
                best = max(best, current)
            else:
                current = 0
        return best

    cleaned = mask.copy()
    cleaned[:] = 0
    img_h, img_w = mask.shape[:2]
    min_area = max(20, int(img_h * img_w * 0.001))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    kept = []
    max_tall_area = 0

    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])
        if area < min_area:
            continue
        if w <= 2 or h <= 2:
            continue
        kept.append((i, x, y, w, h, area))
        if h >= img_h * 0.50:
            max_tall_area = max(max_tall_area, area)

    top_cutoff = int(img_h * 0.30)
    thin_edge_width = max(10, int(img_w * 0.08))
    for i, x, y, w, h, area in kept:
        if y + h <= top_cutoff and h <= top_cutoff:
            continue

        touches_right = x + w >= img_w - 1
        tall_right_sliver = (
            touches_right
            and w <= thin_edge_width
            and h >= img_h * 0.55
            and max_tall_area > 0
            and area <= max_tall_area * 0.38
        )
        if tall_right_sliver:
            continue

        cleaned[labels == i] = 255

    top_search = max(1, int(img_h * 0.45))
    wide_run = max(20, int(img_w * 0.65))
    dense_row_pixels = max(20, int(img_w * 0.35))
    for y in range(top_search):
        row = cleaned[y] > 0
        if longest_active_run(row) >= wide_run or int(np.count_nonzero(row)) >= dense_row_pixels:
            cleaned[y, :] = 0

    right_search_start = max(0, int(img_w * 0.80))
    tall_run = max(20, int(img_h * 0.70))
    dense_col_pixels = max(20, int(img_h * 0.45))
    for x in range(right_search_start, img_w):
        col = cleaned[:, x] > 0
        if longest_active_run(col) >= tall_run or int(np.count_nonzero(col)) >= dense_col_pixels:
            cleaned[:, x] = 0

    return cleaned


def active_column_runs(mask) -> List[Tuple[int, int]]:
    img_h, img_w = mask.shape[:2]
    min_col_pixels = max(1, int(img_h * 0.01))
    active_cols = [cv2.countNonZero(mask[:, x]) >= min_col_pixels for x in range(img_w)]

    runs: List[Tuple[int, int]] = []
    start = None
    for x, is_active in enumerate(active_cols):
        if is_active and start is None:
            start = x
        elif not is_active and start is not None:
            runs.append((start, x))
            start = None
    if start is not None:
        runs.append((start, img_w))

    if not runs:
        return []

    merged = [runs[0]]
    max_gap = max(2, int(img_w * 0.015))
    tiny_gap = max(2, int(img_w * 0.004))
    small_pair_gap = max(max_gap + 2, int(img_w * 0.020))
    small_run = max(6, int(img_w * 0.04))
    for x1, x2 in runs[1:]:
        prev_x1, prev_x2 = merged[-1]
        gap = x1 - prev_x2
        prev_w = prev_x2 - prev_x1
        curr_w = x2 - x1
        should_merge = gap <= tiny_gap or (
            gap <= max_gap and (prev_w <= small_run or curr_w <= small_run)
        ) or (
            gap <= small_pair_gap and prev_w <= small_run and curr_w <= small_run
        )
        if should_merge:
            merged[-1] = (prev_x1, x2)
        else:
            merged.append((x1, x2))

    return merged


def segment_is_active(digit_mask, label: str, zone: Tuple[float, float, float, float]) -> bool:
    return segment_fill_ratio(digit_mask, label, zone) >= SEGMENT_ACTIVE_THRESHOLDS.get(label, 0.25)


def segment_fill_ratio(digit_mask, label: str, zone: Tuple[float, float, float, float]) -> float:
    h, w = digit_mask.shape[:2]
    left, top, right, bottom = zone
    x1 = clamp(int(w * left), 0, w - 1)
    y1 = clamp(int(h * top), 0, h - 1)
    x2 = clamp(int(w * right), x1 + 1, w)
    y2 = clamp(int(h * bottom), y1 + 1, h)
    segment = digit_mask[y1:y2, x1:x2]
    return cv2.countNonZero(segment) / float(max(1, segment.size))


def classify_7seg_digit(digit_mask) -> Optional[Tuple[str, str]]:
    h, w = digit_mask.shape[:2]
    if h < 10 or w < 3:
        return None

    aspect_ratio = w / float(h)
    ratios = {
        label: segment_fill_ratio(digit_mask, label, zone)
        for label, zone in SEGMENT_ZONES.items()
    }
    soft_match = best_soft_digit_match(digit_mask, ratios, aspect_ratio)

    active = frozenset(
        label
        for label, ratio in ratios.items()
        if ratio >= SEGMENT_ACTIVE_THRESHOLDS.get(label, 0.25)
    )
    if active == frozenset("abcdefg") and ratios["g"] < 0.30:
        return "0", "abcdef-weakg"

    # A seven-segment "1" is intentionally narrow, but width alone is too permissive.
    if aspect_ratio <= 0.30:
        right_strength = max(ratios["b"], ratios["c"])
        left_strength = max(ratios["e"], ratios["f"])
        if (
            ratios["a"] >= 0.22
            and right_strength >= 0.18
            and left_strength <= 0.18
            and ratios["d"] <= 0.18
            and ratios["g"] <= 0.18
        ):
            return "7", "abc-narrow"
        if right_strength >= 0.18 and left_strength <= 0.22 and ratios["g"] <= 0.18:
            return "1", "narrow"
        if (
            aspect_ratio <= 0.28
            and w >= max(26, int(h * 0.10))
            and max(left_strength, right_strength) >= 0.45
            and ratios["d"] >= 0.35
            and max(ratios["a"], ratios["g"]) >= 0.10
        ):
            return "1", "narrow-smear1"
        if soft_match is not None:
            return soft_match["digit"], soft_match["label"]
        return None

    # When the crop merges most of the display into one blob, only the right-side
    # bars may survive the segment test. That can look like "bc", but the shape
    # is far too wide to be a real 7-segment "1".
    if active == frozenset("bc") and aspect_ratio >= 0.50:
        if soft_match is not None and soft_match["digit"] != "1":
            return soft_match["digit"], soft_match["label"]
        return None
    if active == frozenset("bcdg") and aspect_ratio >= 0.32:
        if ratios["d"] >= 0.35 and ratios["e"] <= 0.25 and ratios["f"] <= 0.15:
            return "3", "bcdg-3like"
        if soft_match is not None and soft_match["digit"] != "4" and soft_match["score"] >= 0.62:
            return soft_match["digit"], soft_match["label"]
        return "4", "bcdg-loose"
    if (
        ratios["b"] >= 0.25
        and ratios["e"] >= 0.28
        and ratios["g"] >= 0.25
        and ratios["c"] <= 0.12
        and ratios["f"] <= 0.18
        and (ratios["a"] >= 0.15 or ratios["d"] >= 0.28)
    ):
        return "2", "abeg-lowd"
    if (
        ratios["c"] >= 0.38
        and ratios["d"] >= 0.40
        and ratios["b"] >= 0.18
        and ratios["e"] >= 0.16
        and ratios["a"] <= 0.12
        and ratios["f"] <= 0.14
        and ratios["g"] <= 0.12
    ):
        return "0", "right-clipped0"
    if (
        ratios["b"] >= 0.36
        and ratios["c"] >= 0.35
        and ratios["d"] >= 0.45
        and ratios["e"] >= 0.35
        and ratios["f"] >= 0.35
        and ratios["g"] <= 0.18
    ):
        return "0", "weak-top0"
    if (
        ratios["b"] >= 0.28
        and ratios["c"] >= 0.35
        and ratios["d"] >= 0.40
        and ratios["e"] >= 0.35
        and ratios["f"] >= 0.25
        and ratios["g"] <= 0.18
    ):
        return "0", "dim-top0"
    if (
        ratios["f"] >= 0.32
        and ratios["g"] >= 0.30
        and ratios["d"] >= 0.45
        and ratios["c"] >= 0.22
        and ratios["b"] <= 0.08
        and ratios["e"] <= 0.22
        and ratios["a"] <= 0.12
    ):
        return "5", "weak-top5"
    if (
        ratios["e"] >= 0.35
        and ratios["f"] >= 0.25
        and ratios["d"] >= 0.22
        and ratios["a"] <= 0.12
        and ratios["b"] <= 0.18
        and ratios["c"] <= 0.12
        and ratios["g"] <= 0.22
    ):
        return "0", "left-clipped0"
    if (
        ratios["a"] >= 0.10
        and ratios["b"] >= 0.18
        and ratios["c"] >= 0.35
        and ratios["d"] >= 0.38
        and ratios["e"] >= 0.40
        and ratios["f"] >= 0.25
        and ratios["g"] >= 0.30
    ):
        return "8", "weak-top8"
    if (
        ratios["b"] >= 0.34
        and ratios["c"] >= 0.25
        and ratios["d"] >= 0.45
        and ratios["f"] >= 0.35
        and ratios["g"] >= 0.38
        and ratios["e"] <= 0.18
    ):
        return "9", "weak-top9"
    digit = SEGMENT_DIGITS.get(active)
    if digit is not None:
        return digit, "".join(sorted(active))
    if soft_match is not None:
        return soft_match["digit"], soft_match["label"]

    return None


def classify_red_7seg_digit(digit_mask) -> Optional[Tuple[str, str]]:
    result = classify_7seg_digit(digit_mask)
    if result is not None:
        return result

    h, w = digit_mask.shape[:2]
    if h < 10 or w < 3:
        return None

    aspect_ratio = w / float(h)
    ratios = {
        label: segment_fill_ratio(digit_mask, label, zone)
        for label, zone in SEGMENT_ZONES.items()
    }
    left_strength = max(ratios["e"], ratios["f"])
    right_strength = max(ratios["b"], ratios["c"])
    if (
        aspect_ratio <= 0.36
        and max(left_strength, right_strength) >= 0.45
        and ratios["a"] >= 0.20
        and ratios["d"] >= 0.30
        and ratios["g"] >= 0.25
    ):
        return "1", "narrow-red1"

    return None


def split_unresolved_run(
    column_slice: np.ndarray,
    x_offset: int,
    y_offset: int,
    run_h: int,
) -> List[Dict[str, Any]]:
    digit_mask = column_slice[y_offset:y_offset + run_h, :]
    h, w = digit_mask.shape[:2]
    if w < max(24, int(h * 0.18)):
        return []

    col_counts = np.array([cv2.countNonZero(digit_mask[:, x:x + 1]) for x in range(w)], dtype=np.float32)
    if col_counts.size == 0 or float(col_counts.max()) <= 0.0:
        return []

    kernel_w = max(3, min(9, w // 10 * 2 + 1))
    kernel = np.ones(kernel_w, dtype=np.float32) / float(kernel_w)
    smooth = np.convolve(col_counts, kernel, mode="same")
    low_thresh = max(2.0, float(smooth.max()) * 0.18)
    margin = max(3, int(w * 0.10))

    cut_points: List[int] = []
    start = None
    for x in range(margin, max(margin, w - margin)):
        if smooth[x] <= low_thresh:
            if start is None:
                start = x
        elif start is not None:
            cut_points.append((start + x) // 2)
            start = None
    if start is not None:
        cut_points.append((start + max(margin, w - margin)) // 2)

    cut_points = sorted({x for x in cut_points if margin <= x <= w - margin})
    if not cut_points:
        return []

    cut_points = cut_points[:4]
    partitions: List[Tuple[int, ...]] = []
    partitions.extend((cut,) for cut in cut_points)
    for i in range(len(cut_points)):
        for j in range(i + 1, len(cut_points)):
            partitions.append((cut_points[i], cut_points[j]))

    best_segments: List[Dict[str, Any]] = []
    for cuts in partitions:
        bounds = [0, *cuts, w]
        segments: List[Dict[str, Any]] = []
        valid = True
        for left, right in zip(bounds, bounds[1:]):
            if right - left < max(6, int(w * 0.10)):
                valid = False
                break
            segment = digit_mask[:, left:right]
            points = cv2.findNonZero(segment)
            if points is None:
                valid = False
                break
            _, seg_y, _, seg_h = cv2.boundingRect(points)
            if seg_h < h * 0.35:
                valid = False
                break
            trimmed = segment[seg_y:seg_y + seg_h, :]
            result = classify_7seg_digit(trimmed)
            if result is None:
                valid = False
                break
            segments.append(
                {
                    "result": result,
                    "x1": x_offset + left,
                    "x2": x_offset + right,
                    "y": y_offset + seg_y,
                    "w": trimmed.shape[1],
                    "h": trimmed.shape[0],
                    "area": cv2.countNonZero(trimmed),
                }
            )

        if not valid:
            continue
        if len(segments) > len(best_segments):
            best_segments = segments
        elif len(segments) == len(best_segments) and segments:
            if sum(item["area"] for item in segments) > sum(item["area"] for item in best_segments):
                best_segments = segments

    return best_segments


def merge_split_zero_or_eight_runs(candidates: List[Dict[str, Any]], cleaned: np.ndarray) -> List[Dict[str, Any]]:
    if len(candidates) < 2:
        return candidates

    def is_narrow_one(item: Dict[str, Any]) -> bool:
        return item["digit"] == "1" and item["segments"] in {"narrow", "narrow-smear1", "narrow-red1", "bc"}

    reference = [item for item in candidates if not is_narrow_one(item)]
    if not reference:
        reference = candidates
    ref_w = max(1, int(np.median([item["w"] for item in reference])))
    ref_h = max(1, int(np.median([item["h"] for item in reference])))

    merged: List[Dict[str, Any]] = []
    index = 0
    while index < len(candidates):
        current = candidates[index]
        if index + 1 >= len(candidates) or not is_narrow_one(current) or not is_narrow_one(candidates[index + 1]):
            merged.append(current)
            index += 1
            continue

        nxt = candidates[index + 1]
        gap = nxt["x1"] - current["x2"]
        combined_x1 = min(current["x1"], nxt["x1"])
        combined_x2 = max(current["x2"], nxt["x2"])
        combined_y1 = min(current["y"], nxt["y"])
        combined_y2 = max(current["y"] + current["h"], nxt["y"] + nxt["h"])
        combined_w = combined_x2 - combined_x1
        combined_h = combined_y2 - combined_y1
        overlap_y = min(current["y"] + current["h"], nxt["y"] + nxt["h"]) - max(current["y"], nxt["y"])

        plausible_width = ref_w * 0.45 <= combined_w <= ref_w * 1.45
        plausible_height = combined_h >= ref_h * 0.55
        close_pair = gap <= max(18, int(ref_w * 0.28))
        aligned = overlap_y >= min(current["h"], nxt["h"]) * 0.55
        if not (plausible_width and plausible_height and close_pair and aligned):
            merged.append(current)
            index += 1
            continue

        combined_mask = cleaned[
            max(0, combined_y1):min(cleaned.shape[0], combined_y2),
            max(0, combined_x1):min(cleaned.shape[1], combined_x2),
        ]
        result = None
        points = cv2.findNonZero(combined_mask)
        if points is not None:
            x, y, w, h = cv2.boundingRect(points)
            result = classify_7seg_digit(combined_mask[y:y + h, x:x + w])

        if result is not None and result[0] in {"0", "8", "9"}:
            digit, segments = result
            segments = f"merged-{segments}"
        else:
            digit, segments = "0", "split0"

        merged.append(
            {
                "digit": digit,
                "segments": segments,
                "x1": combined_x1,
                "x2": combined_x2,
                "y": combined_y1,
                "w": combined_w,
                "h": combined_h,
                "area": current["area"] + nxt["area"],
            }
        )
        index += 2

    return merged


def read_7seg_from_mask_debug(mask, red_display: bool = False) -> Optional[Dict[str, Any]]:
    cleaned = clean_7seg_mask(mask)
    if cv2.countNonZero(cleaned) == 0:
        return None

    img_h = cleaned.shape[0]
    img_w = cleaned.shape[1]
    run_infos: List[Dict[str, Any]] = []

    for x1, x2 in active_column_runs(cleaned):
        column_slice = cleaned[:, x1:x2]
        points = cv2.findNonZero(column_slice)
        if points is None:
            continue

        _, y, _, h = cv2.boundingRect(points)
        if h < img_h * 0.25:
            continue

        digit_mask = column_slice[y:y + h, :]
        result = classify_red_7seg_digit(digit_mask) if red_display else classify_7seg_digit(digit_mask)
        if result is None:
            split_segments = split_unresolved_run(column_slice, x1, y, h)
            if split_segments:
                run_infos.extend(split_segments)
                continue
        run_infos.append({
            "result": result,
            "x1": x1,
            "x2": x2,
            "y": y,
            "w": digit_mask.shape[1],
            "h": digit_mask.shape[0],
            "area": cv2.countNonZero(digit_mask),
        })

    if not run_infos:
        return None

    candidate_runs = [item for item in run_infos if item["result"] is not None]
    if not candidate_runs:
        return None

    max_area_all = max(item["area"] for item in candidate_runs)
    max_h_all = max(item["h"] for item in candidate_runs)
    max_w_all = max(item["w"] for item in candidate_runs)
    alias_hints: set[str] = set()

    candidates: List[Dict[str, Any]] = []
    for item in run_infos:
        result = item["result"]
        if result is None:
            edge_side = item["x1"] <= img_w * 0.05 or item["x2"] >= img_w * 0.95
            slim_unknown = item["w"] <= max(10, int(max_w_all * 0.32))
            tiny_unknown = item["area"] <= max_area_all * 0.22
            if edge_side and (slim_unknown or tiny_unknown):
                if item["x2"] >= img_w * 0.95 and slim_unknown and item["h"] >= max_h_all * 0.75:
                    alias_hints.add("right_unknown_narrow")
                continue
            return None

        digit, segments = result
        candidates.append({
            "digit": digit,
            "segments": segments,
            "x1": item["x1"],
            "x2": item["x2"],
            "y": item["y"],
            "w": item["w"],
            "h": item["h"],
            "area": item["area"],
        })

    if not candidates:
        return None

    if len(candidates) > 1:
        core_candidates = [item for item in candidates if not (item["digit"] == "1" and item["segments"] == "narrow")]
        reference_candidates = core_candidates or candidates
        max_area = max(item["area"] for item in reference_candidates)
        max_h = max(item["h"] for item in reference_candidates)
        max_w = max(item["w"] for item in reference_candidates)
        filtered = []
        for index, item in enumerate(candidates):
            prev = candidates[index - 1] if index > 0 else None
            nxt = candidates[index + 1] if index + 1 < len(candidates) else None
            gap_left = item["x1"] - prev["x2"] if prev is not None else 0
            gap_right = nxt["x1"] - item["x2"] if nxt is not None else 0
            edge_side = item["x2"] <= img_w * 0.30 or item["x1"] >= img_w * 0.70
            sparse_gap = gap_left >= max(8, int(max_w * 0.20)) or gap_right >= max(8, int(max_w * 0.20))
            abnormal_shape = item["h"] > max_h * 1.10 or item["area"] < max_area * 0.45
            tiny_artifact = item["area"] < max_area * 0.18 and item["h"] < max_h * 0.80
            huge_merged_artifact = item["h"] > max_h * 1.45 or item["area"] > max_area * 2.20
            is_edge_narrow_artifact = (
                item["digit"] == "1"
                and item["segments"] == "narrow"
                and item["x1"] >= img_w * 0.82
                and (
                    (
                        item["w"] <= max(12, int(max_w * 0.28))
                        and (tiny_artifact or sparse_gap or abnormal_shape)
                    )
                    or huge_merged_artifact
                    or item["x2"] >= img_w - max(4, int(img_w * 0.015))
                )
            )
            if not is_edge_narrow_artifact:
                filtered.append(item)
        candidates = filtered

    if not candidates:
        return None

    candidates = merge_split_zero_or_eight_runs(candidates, cleaned)
    if not candidates:
        return None

    max_area = max(item["area"] for item in candidates)
    max_h = max(item["h"] for item in candidates)
    max_w = max(item["w"] for item in candidates)
    digit_top = min(item["y"] for item in candidates)
    digit_bottom = max(item["y"] + item["h"] for item in candidates)

    dot_candidates: List[Tuple[int, Dict[str, Any]]] = []
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, 8)
    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])
        cx, cy = centroids[i]
        aspect = w / float(max(1, h))

        if area >= max_area * 0.12:
            continue
        if h >= max_h * 0.28 or w >= max_w * 0.28:
            continue
        if not (0.45 <= aspect <= 2.20):
            continue
        if cy < digit_top + max_h * 0.65:
            continue
        if cy > digit_bottom + max_h * 0.10:
            continue
        slot = None
        overlapping = [
            (index, item)
            for index, item in enumerate(candidates)
            if not (x + w <= item["x1"] or x >= item["x2"])
        ]
        if not overlapping:
            for index, item in enumerate(candidates):
                next_x1 = candidates[index + 1]["x1"] if index + 1 < len(candidates) else cleaned.shape[1] + 1
                if item["x2"] <= cx <= next_x1:
                    slot = index + 1
                    break
        elif len(overlapping) == 1:
            index, item = overlapping[0]
            local_x = (cx - item["x1"]) / float(max(1, item["x2"] - item["x1"]))
            if local_x >= 0.72:
                slot = index + 1
            elif local_x <= 0.28:
                slot = index

        if slot is None:
            continue
        if slot <= 0:
            continue

        dot_candidates.append((slot, {"area": area}))

    dot_slots: Dict[int, Dict[str, Any]] = {}
    if dot_candidates:
        best_slot, best_info = max(dot_candidates, key=lambda item: item[1]["area"])
        dot_slots[best_slot] = best_info

    text_parts: List[str] = []
    details: List[str] = []
    for index, item in enumerate(candidates):
        text_parts.append(item["digit"])
        details.append(f"{item['digit']}:{item['segments']}")
        if (index + 1) in dot_slots:
            text_parts.append(".")
            details.append(".:dot")

    digits = text_parts
    confidence = 95.0 if all(":" in item for item in details) else 85.0
    return {
        "text": "".join(digits),
        "conf": confidence,
        "raw": f"[7seg:{','.join(details)}]",
        "clean_mask": cleaned,
        "alias_hints": alias_hints,
    }


def read_7seg_from_mask(mask) -> Optional[Tuple[str, float, str]]:
    result = read_7seg_from_mask_debug(mask)
    if result is None:
        return None
    return result["text"], result["conf"], result["raw"]


def infer_dot_slot_from_mask(mask: np.ndarray) -> Optional[int]:
    cleaned = clean_7seg_mask(mask)
    if cv2.countNonZero(cleaned) == 0:
        return None

    img_h, img_w = cleaned.shape[:2]
    run_infos: List[Dict[str, Any]] = []
    for x1, x2 in active_column_runs(cleaned):
        column_slice = cleaned[:, x1:x2]
        points = cv2.findNonZero(column_slice)
        if points is None:
            continue
        _, y, _, h = cv2.boundingRect(points)
        if h < img_h * 0.20:
            continue
        digit_mask = column_slice[y:y + h, :]
        run_infos.append({
            "x1": x1,
            "x2": x2,
            "y": y,
            "w": digit_mask.shape[1],
            "h": digit_mask.shape[0],
            "area": cv2.countNonZero(digit_mask),
        })

    if len(run_infos) < 2:
        return None

    max_area = max(item["area"] for item in run_infos)
    max_h = max(item["h"] for item in run_infos)
    max_w = max(item["w"] for item in run_infos)
    candidates = [
        item
        for item in run_infos
        if item["h"] >= max_h * 0.65 and item["area"] >= max_area * 0.18
    ]
    if len(candidates) < 2:
        candidates = [item for item in run_infos if item["h"] >= max_h * 0.65]

    filtered_candidates: List[Dict[str, Any]] = []
    for item in candidates:
        is_edge = item["x1"] <= img_w * 0.05 or item["x2"] >= img_w * 0.95
        is_tiny = item["w"] <= max(10, int(max_w * 0.28)) and item["area"] <= max_area * 0.25
        if is_edge and is_tiny:
            continue
        filtered_candidates.append(item)
    candidates = filtered_candidates
    if len(candidates) < 2:
        return None

    digit_top = min(item["y"] for item in candidates)
    digit_bottom = max(item["y"] + item["h"] for item in candidates)

    dot_candidates: List[Tuple[int, int]] = []
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, 8)
    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])
        cx, cy = centroids[i]
        aspect = w / float(max(1, h))

        if area >= max_area * 0.16:
            continue
        if h >= max_h * 0.30 or w >= max_w * 0.35:
            continue
        if not (0.40 <= aspect <= 2.40):
            continue
        if cy < digit_top + max_h * 0.55:
            continue
        if cy > digit_bottom + max_h * 0.15:
            continue

        slot = None
        overlapping = [
            (index, item)
            for index, item in enumerate(candidates)
            if not (x + w <= item["x1"] or x >= item["x2"])
        ]
        if not overlapping:
            for index, item in enumerate(candidates):
                next_x1 = candidates[index + 1]["x1"] if index + 1 < len(candidates) else cleaned.shape[1] + 1
                if item["x2"] <= cx <= next_x1:
                    slot = index + 1
                    break
        elif len(overlapping) == 1:
            index, item = overlapping[0]
            local_x = (cx - item["x1"]) / float(max(1, item["x2"] - item["x1"]))
            if local_x >= 0.72:
                slot = index + 1
            elif local_x <= 0.28:
                slot = index

        if slot is None or slot <= 0:
            continue

        dot_candidates.append((slot, area))

    if not dot_candidates:
        return None

    return max(dot_candidates, key=lambda item: item[1])[0]


def read_7seg_from_stages_debug(stages: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    stage_priority = [
        "after_filters",
        "after_close",
        "bin_digits_white",
        "ocr_input_mask",
    ]

    votes: Dict[str, Dict[str, Any]] = {}
    for stage_name in stage_priority:
        mask = stages.get(stage_name)
        if mask is None:
            continue

        result = read_7seg_from_mask_debug(mask, red_display=stages.get("mask_source") == "red")
        if result is None:
            continue

        text = result["text"]
        conf = result["conf"]
        raw = result["raw"]
        result_alias_hints = set(result.get("alias_hints", set()))
        if stages.get("mask_source") == "red":
            result_alias_hints.add("red_digit_display")
            raw = raw.replace("[7seg:", "[7seg:red,", 1)
        bucket = votes.setdefault(
            text,
            {
                "count": 0,
                "best_conf": 0.0,
                "raw": raw,
                "stage_rank": 999,
                "stage_name": "",
                "stage_mask": None,
                "alias_hints": set(),
            },
        )
        bucket["count"] += 1
        bucket["best_conf"] = max(bucket["best_conf"], conf)
        bucket["raw"] = raw
        bucket["alias_hints"].update(result_alias_hints)
        stage_rank = stage_priority.index(stage_name)
        if stage_rank < bucket["stage_rank"]:
            bucket["stage_rank"] = stage_rank
            bucket["stage_name"] = stage_name
            bucket["stage_mask"] = result["clean_mask"].copy()

    if not votes:
        return None

    best_text, best_info = max(
        votes.items(),
        key=lambda item: (item[1]["count"], item[1]["best_conf"], -item[1]["stage_rank"]),
    )
    conf = 95.0 if best_info["count"] >= 2 else 85.0
    return {
        "text": best_text,
        "conf": conf,
        "raw": best_info["raw"],
        "stage_name": best_info["stage_name"],
        "stage_mask": best_info["stage_mask"],
        "alias_hints": set(best_info.get("alias_hints", set())),
    }


def make_odd(n: int) -> int:
    return n if (n % 2 == 1) else n + 1


def clone_params(p: Params) -> Params:
    return Params(**asdict(p))


def crop_relative_region(lcd_bgr, region, margin_x_ratio: float = 0.03, margin_y_ratio: float = 0.05):
    """Crop a relative region from the selected whole-LCD ROI."""
    h, w = lcd_bgr.shape[:2]
    if h < 20 or w < 20:
        return lcd_bgr

    if margin_x_ratio > 0.0 or margin_y_ratio > 0.0:
        margin_x = max(2, int(w * margin_x_ratio))
        margin_y = max(2, int(h * margin_y_ratio))
        inner = lcd_bgr[margin_y:h - margin_y, margin_x:w - margin_x]
        if inner.size == 0:
            return lcd_bgr
    else:
        inner = lcd_bgr

    inner_h, inner_w = inner.shape[:2]
    left, top, right, bottom = region
    x1 = clamp(int(inner_w * left), 0, inner_w)
    y1 = clamp(int(inner_h * top), 0, inner_h)
    x2 = clamp(int(inner_w * right), 0, inner_w)
    y2 = clamp(int(inner_h * bottom), 0, inner_h)

    if x2 <= x1 or y2 <= y1:
        return inner

    return inner[y1:y2, x1:x2].copy()


def crop_lcd_reading_area(lcd_bgr):
    """Backward-compatible default reading-area crop."""
    return crop_relative_region(lcd_bgr, LCD_READING_REGION)


def build_red_digit_mask(roi_bgr) -> np.ndarray:
    """Return a binary mask for saturated red display strokes."""
    if roi_bgr is None or roi_bgr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    b, g, r = cv2.split(roi_bgr)
    r16 = r.astype(np.int16)
    red_excess = r16 - np.maximum(g, b).astype(np.int16)
    mask = (
        (red_excess >= 45)
        & (r >= 130)
        & (r16 > g.astype(np.int16) + 18)
        & (r16 > b.astype(np.int16) + 18)
    ).astype(np.uint8) * 255

    return mask


def red_mask_has_digit_signal(mask: np.ndarray) -> bool:
    if mask is None or mask.size == 0:
        return False

    nonzero = cv2.countNonZero(mask)
    area = mask.shape[0] * mask.shape[1]
    if nonzero < max(30, int(area * 0.00025)):
        return False
    return nonzero <= area * 0.35


def build_red_digit_roi_candidates(lcd_bgr) -> List[Tuple[str, Any]]:
    """Find the dominant horizontal red display band inside a larger photo."""
    mask = build_red_digit_mask(lcd_bgr)
    if not red_mask_has_digit_signal(mask):
        return []

    img_h, img_w = mask.shape[:2]
    close_w = max(15, min(90, int(img_w * 0.06)))
    close_h = max(3, min(15, int(img_h * 0.015)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h))
    merged = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(merged, 8)
    boxes: List[Tuple[int, int, int, int, int]] = []
    min_area = max(80, int(img_h * img_w * 0.00008))
    min_w = max(30, int(img_w * 0.04))
    min_h = max(10, int(img_h * 0.025))

    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])
        if area < min_area or w < min_w or h < min_h:
            continue
        if w / float(max(1, h)) < 1.20:
            continue
        boxes.append((area, x, y, w, h))

    if not boxes:
        return []

    boxes.sort(reverse=True)
    area, x, y, w, h = boxes[0]
    _ = area
    pad_x = max(4, int(h * 0.25))
    pad_y = max(4, int(h * 0.25))
    x1 = clamp(x - pad_x, 0, img_w)
    y1 = clamp(y - pad_y, 0, img_h)
    x2 = clamp(x + w + pad_x, 0, img_w)
    y2 = clamp(y + h + pad_y, 0, img_h)
    if x2 <= x1 or y2 <= y1:
        return []

    return [("red_digit_band", lcd_bgr[y1:y2, x1:x2].copy())]


def build_reading_roi_candidates(lcd_bgr):
    candidates = []
    h, w = lcd_bgr.shape[:2]
    is_numeric_window = h > 0 and (w / float(h)) >= 1.60

    candidates.extend(build_red_digit_roi_candidates(lcd_bgr))
    candidates.append(("selected_roi", lcd_bgr.copy()))

    if is_numeric_window:
        region_variants = WINDOW_READING_REGION_VARIANTS
        margin_x_ratio = 0.0
        margin_y_ratio = 0.0
    else:
        region_variants = READING_REGION_VARIANTS
        margin_x_ratio = 0.03
        margin_y_ratio = 0.05

    for name, region in region_variants:
        cropped = crop_relative_region(
            lcd_bgr,
            region,
            margin_x_ratio=margin_x_ratio,
            margin_y_ratio=margin_y_ratio,
        )
        if cropped.size == 0:
            continue
        candidates.append((name, cropped))

    for name, region in FULL_ROI_READING_REGION_VARIANTS:
        cropped = crop_relative_region(lcd_bgr, region, margin_x_ratio=0.0, margin_y_ratio=0.0)
        if cropped.size == 0:
            continue
        candidates.append((name, cropped))

    if not candidates:
        candidates.append(("fallback", lcd_bgr))

    return candidates


def build_reading_roi_candidate_groups(lcd_bgr):
    primary: List[Tuple[str, np.ndarray]] = []
    secondary: List[Tuple[str, np.ndarray]] = []
    for name, crop in build_reading_roi_candidates(lcd_bgr):
        if name.startswith("raw_band"):
            secondary.append((name, crop))
        else:
            primary.append((name, crop))
    return primary, secondary


def remove_edge_components(bin_img):
    cleaned = bin_img.copy()
    img_h, img_w = cleaned.shape[:2]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)

    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])
        touches_edge = x <= 1 or y <= 1 or x + w >= img_w - 1 or y + h >= img_h - 1
        large_edge_blob = w >= img_w * 0.35 or h >= img_h * 0.35
        long_thin_edge_blob = (w >= img_w * 0.25 and h <= img_h * 0.08) or (h >= img_h * 0.25 and w <= img_w * 0.08)

        if touches_edge and (large_edge_blob or long_thin_edge_blob):
            cleaned[labels == i] = 0

    return cleaned


def build_param_variants(p: Params) -> List[Tuple[str, Params]]:
    variants: List[Tuple[str, Params]] = [("base", clone_params(p))]
    seen: set[tuple[int, int, int, int, int, int]] = {
        (p.method, p.scale, p.blur, p.dark_threshold, p.close_enable, p.close_k)
    }

    if p.method == 2:
        for delta in (-4, -3, -2, -1, 1, 2):
            p2 = clone_params(p)
            p2.dark_threshold = clamp(p.dark_threshold + delta, 0, 255)
            key = (p2.method, p2.scale, p2.blur, p2.dark_threshold, p2.close_enable, p2.close_k)
            if key not in seen:
                seen.add(key)
                variants.append((f"dt{delta:+d}", p2))

        for name, dark_threshold in (
            ("noise_safe", clamp(p.dark_threshold, 0, 255)),
            ("noise_safe_hi", clamp(p.dark_threshold + 5, 0, 255)),
        ):
            p2 = clone_params(p)
            p2.scale = 1
            p2.blur = 2
            p2.close_enable = 0
            p2.dark_threshold = dark_threshold
            key = (p2.method, p2.scale, p2.blur, p2.dark_threshold, p2.close_enable, p2.close_k)
            if key not in seen:
                seen.add(key)
                variants.append((name, p2))

    return variants


def preprocess(roi_bgr, p: Params) -> Dict[str, Any]:
    stages: Dict[str, Any] = {}

    # Upscale
    s = clamp(p.scale, 1, 6)
    work_bgr = roi_bgr
    if s != 1:
        work_bgr = cv2.resize(roi_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2GRAY)

    # Blur
    if p.blur == 1:
        gray_b = cv2.GaussianBlur(gray, (3, 3), 0)
    elif p.blur == 2:
        gray_b = cv2.GaussianBlur(gray, (5, 5), 0)
    else:
        gray_b = gray

    stages["gray"] = gray_b

    red_mask = build_red_digit_mask(work_bgr)
    if red_mask_has_digit_signal(red_mask):
        bin_img = red_mask
        stages["mask_source"] = "red"
        stages["red_digits_white"] = red_mask
    else:
        stages["mask_source"] = "gray"

        # Threshold: digits WHITE on BLACK
        if p.method == 0:
            _, bin_img = cv2.threshold(gray_b, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif p.method == 1:
            block = make_odd(clamp(p.adaptive_block, 3, 151))
            c_val = clamp(p.adaptive_c, 0, 50)
            bin_img = cv2.adaptiveThreshold(
                gray_b, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block, c_val
            )
        else:
            dark_threshold = clamp(p.dark_threshold, 0, 255)
            _, bin_img = cv2.threshold(gray_b, dark_threshold, 255, cv2.THRESH_BINARY_INV)

    # Ensure digits are WHITE on BLACK
    if cv2.countNonZero(bin_img) > (bin_img.shape[0] * bin_img.shape[1] / 2):
        bin_img = cv2.bitwise_not(bin_img)

    edge = max(2, int(min(bin_img.shape[:2]) * 0.025))
    bin_img[:edge, :] = 0
    bin_img[-edge:, :] = 0
    bin_img[:, :edge] = 0
    bin_img[:, -edge:] = 0
    bin_img = remove_edge_components(bin_img)

    stages["bin_digits_white"] = bin_img

    processed = bin_img.copy()

    # Directional close to bridge 7-seg gaps
    if p.close_enable == 1 and p.close_iter > 0:
        it = clamp(p.close_iter, 1, 3)
        k = clamp(p.close_k, 1, 31)

        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel_h, iterations=it)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel_v, iterations=it)

    stages["after_close"] = processed

    # Optional dilation/erosion
    if p.dilate_iter > 0:
        kd = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.dilate(processed, kd, iterations=clamp(p.dilate_iter, 1, 2))

    if p.erode_iter > 0:
        ke = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.erode(processed, ke, iterations=clamp(p.erode_iter, 1, 2))

    if p.median == 3:
        processed = cv2.medianBlur(processed, 3)

    stages["after_filters"] = processed

    # OCR input: BLACK text on WHITE background
    ocr_input = cv2.bitwise_not(processed)

    # Padding
    pad = clamp(p.pad, 0, 80)
    if pad > 0:
        ocr_input = cv2.copyMakeBorder(ocr_input, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)

    stages["ocr_input"] = ocr_input
    _, ocr_input_mask = cv2.threshold(ocr_input, 245, 255, cv2.THRESH_BINARY_INV)
    stages["ocr_input_mask"] = ocr_input_mask
    return stages


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
    for name, crop in build_reading_roi_candidates(lcd_roi_bgr):
        if crop is not None and crop.size > 0 and name not in candidates:
            candidates[name] = crop
    if "selected_roi" not in candidates:
        candidates["selected_roi"] = lcd_roi_bgr.copy()
    return candidates


def _variant_map(p: Params) -> Dict[str, Params]:
    variants: Dict[str, Params] = {}
    for name, params in build_param_variants(p):
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
    dot_slot = infer_dot_slot_from_mask(mask)
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
    red_candidates = build_red_digit_roi_candidates(lcd_roi_bgr)
    if red_candidates:
        return red_candidates[0]
    return "selected_roi", lcd_roi_bgr


def _primary_7seg_params(params: Params, crop_name: str) -> Params:
    primary_params = clone_params(params)
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


def _primary_7seg_attempt(lcd_roi_bgr: np.ndarray, params: Params) -> Tuple[Optional[Candidate], Dict[str, Any]]:
    start = time.perf_counter()
    try:
        crop_name, reading_roi = _primary_7seg_crop(lcd_roi_bgr)
        if reading_roi is None or reading_roi.size == 0:
            return None, _primary_7seg_failure_debug("no_primary_crop", (time.perf_counter() - start) * 1000.0)

        stages = preprocess(reading_roi, _primary_7seg_params(params, crop_name))
        result = read_7seg_from_stages_debug(stages)
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
    variants: Dict[str, Params],
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
            stages = preprocess(reading_roi, p_variant)
            result = read_7seg_from_stages_debug(stages)
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
    p: Params | None = None,
    expand_weak_7seg: bool = False,
) -> Tuple[str, float, str, Dict[str, Any]]:
    params = p or Params()
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
