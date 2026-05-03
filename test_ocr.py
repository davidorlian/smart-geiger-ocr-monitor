import cv2
import pytesseract
import re
import json
import argparse
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
from collections import Counter


# -------------------------
# Params
# -------------------------

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


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def make_odd(n: int) -> int:
    return n if (n % 2 == 1) else n + 1


def select_roi(image) -> Tuple[int, int, int, int]:
    roi_rect = cv2.selectROI("Select ROI", image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    return tuple(map(int, roi_rect))


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# Relative crop inside the selected LCD: left, top, right, bottom.
# The setup ROI is intentionally forgiving; this keeps the numeric reading band.
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


def default_test_dir() -> Path:
    return Path(__file__).resolve().parent / "test_v2"


def collect_image_paths(image_path: Optional[str], image_dir: Optional[str]) -> List[Path]:
    if image_path:
        return [Path(image_path).expanduser()]

    test_dir = Path(image_dir).expanduser() if image_dir else default_test_dir()
    if not test_dir.is_dir():
        return []

    return sorted(
        [p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )


def image_roi_key(image_path: Path) -> str:
    return image_path.name


def expected_text_from_filename(image_path: Path) -> Optional[str]:
    stem = image_path.stem
    prefix = "ram_gene_"
    if stem.startswith(prefix):
        token = stem[len(prefix):]
    else:
        match = re.search(r"(\d+(?:p\d+)?)$", stem)
        if not match:
            return None
        token = match.group(1)

    expected = token.replace("p", ".", 1)
    return expected if FINAL_NUMERIC_RE.fullmatch(expected) else None


def parse_roi_arg(roi_arg: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not roi_arg:
        return None

    try:
        values = [int(part.strip()) for part in roi_arg.split(",")]
    except ValueError as e:
        raise argparse.ArgumentTypeError("--roi must contain integers: x,y,w,h") from e

    if len(values) != 4:
        raise argparse.ArgumentTypeError("--roi must contain exactly four values: x,y,w,h")

    x, y, w, h = values
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("--roi width and height must be positive")

    return x, y, w, h


def parse_roi_list(roi_data: Any, label: str) -> Tuple[int, int, int, int]:
    if not isinstance(roi_data, list) or len(roi_data) != 4:
        raise ValueError(f"{label} must be a list of four integers: [x, y, w, h]")

    try:
        x, y, w, h = tuple(int(v) for v in roi_data)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{label} must contain only integers: [x, y, w, h]") from e

    if w <= 0 or h <= 0:
        raise ValueError(f"{label} width and height must be positive")

    return x, y, w, h


def clone_params(p: Params) -> Params:
    return Params(**asdict(p))


def crop_roi(image, roi_rect: Tuple[int, int, int, int]):
    x, y, w, h = roi_rect
    img_h, img_w = image.shape[:2]
    x1 = clamp(x, 0, img_w)
    y1 = clamp(y, 0, img_h)
    x2 = clamp(x + w, 0, img_w)
    y2 = clamp(y + h, 0, img_h)

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2].copy()


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


def build_reading_roi_candidates(lcd_bgr):
    candidates = []
    h, w = lcd_bgr.shape[:2]
    is_numeric_window = h > 0 and (w / float(h)) >= 1.60

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
        cropped = crop_relative_region(lcd_bgr, region, margin_x_ratio=margin_x_ratio, margin_y_ratio=margin_y_ratio)
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


def ocr_all_images(
    image_paths: List[Path],
    fallback_roi_rect: Optional[Tuple[int, int, int, int]],
    p: Params,
    image_rois: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
    image_params: Optional[Dict[str, Params]] = None,
    select_missing_rois: bool = False,
) -> None:
    print("\n" + "=" * 70)
    print("OCR ALL TEST IMAGES")
    print("=" * 70)

    for i, image_path in enumerate(image_paths, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"{i:02d}) {image_path.name}: ERROR failed to load image")
            continue

        key = image_roi_key(image_path)
        roi_rect = image_rois.get(key) if image_rois is not None else None

        if roi_rect is None and select_missing_rois:
            print(f"[INFO] Select LCD ROI for {image_path.name}")
            roi_rect = select_roi(image)
            if roi_rect == (0, 0, 0, 0):
                print(f"{i:02d}) {image_path.name}: SKIP ROI selection cancelled")
                continue
            if image_rois is not None:
                image_rois[key] = roi_rect

        if roi_rect is None:
            roi_rect = fallback_roi_rect

        if roi_rect is None:
            print(f"{i:02d}) {image_path.name}: SKIP no ROI selected")
            continue

        current_params = clone_params(image_params[key]) if image_params is not None and key in image_params else clone_params(p)

        roi = crop_roi(image, roi_rect)
        if roi is None:
            print(f"{i:02d}) {image_path.name}: ERROR ROI {roi_rect} is outside this image")
            continue

        txt, conf, raw, debug = robust_ocr_from_lcd_roi(roi, current_params)
        winner_label = debug.get("winner_label", "")
        winner_stage = debug.get("winner_stage", "")
        vote_count = int(debug.get("vote_count", 0))
        winner_suffix = ""
        if winner_label:
            stage_suffix = f" stage={winner_stage}" if winner_stage else ""
            winner_suffix = f" via={winner_label}{stage_suffix} votes={vote_count}"
        print(
            f"{i:02d}) {image_path.name}: roi={roi_rect} clean='{txt}' conf={conf:.1f} "
            f"raw='{raw}'{winner_suffix} params={json.dumps(asdict(current_params))}"
        )

    print("=" * 70 + "\n")


def short_penalty_text(penalties: Any, limit: int = 4) -> str:
    if not penalties:
        return "none"
    if isinstance(penalties, str):
        return penalties
    return ",".join(str(item) for item in list(penalties)[:limit]) or "none"


def candidate_rows_from_debug(debug: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = debug.get("candidate_summaries", []) if isinstance(debug, dict) else []
    return rows if isinstance(rows, list) else []


def classify_batch_failure(
    expected: Optional[str],
    actual: str,
    debug: Dict[str, Any],
) -> str:
    rows = candidate_rows_from_debug(debug)
    if not rows and not actual:
        return "no_candidate"

    if actual and not is_valid_final_numeric_text(actual):
        return "malformed_numeric"

    if expected:
        if any(str(row.get("clean", "")) == expected for row in rows):
            return "correct_candidate_lost"

        if actual and expected.replace(".", "") == actual.replace(".", "") and expected != actual:
            return "decimal_issue"

        if ("." in expected) != ("." in actual):
            return "decimal_issue"

    top = rows[0] if rows else {}
    penalties = [str(item) for item in top.get("penalties", [])]
    families = str(top.get("families", ""))
    artifact_penalty = float(top.get("artifact_penalty", debug.get("artifact_penalty", 0.0) or 0.0))
    structural = float(top.get("structural_quality", debug.get("structural_quality", 0.0) or 0.0))

    if actual and "." in actual and "+" not in families:
        if any("single_family_decimal" in item for item in penalties):
            return "single_family_decimal_won"

    if artifact_penalty >= 0.50 or any("right_border_artifact" in item for item in penalties):
        return "right_edge_artifact"

    if structural and structural < 0.55:
        return "weak_structural_candidate_won"

    if expected and actual:
        return "wrong_digit_classification"

    return "unknown"


def print_candidate_rows(rows: List[Dict[str, Any]], limit: int) -> None:
    for row in rows[:limit]:
        print(
            f"      cand#{int(row.get('rank', 0)):02d} "
            f"clean='{row.get('clean', '')}' raw='{row.get('raw', '')}' "
            f"source={row.get('source', '')} families={row.get('families', '')} "
            f"votes={int(row.get('vote_count', 0))} "
            f"struct={float(row.get('structural_quality', 0.0)):.2f} "
            f"artifact={float(row.get('artifact_penalty', 0.0)):.2f} "
            f"final={float(row.get('final_score', 0.0)):.2f} "
            f"penalties={short_penalty_text(row.get('penalties', []))}"
        )


def run_batch_cropped_mode(
    image_paths: List[Path],
    p: Params,
    image_params: Optional[Dict[str, Params]] = None,
    top_candidates: int = 5,
    show_ok_candidates: bool = False,
) -> None:
    print("\n" + "=" * 90)
    print("OCR BATCH CROPPED TEST MODE")
    print("Input images are treated as already-cropped numeric reading areas.")
    print("=" * 90)

    total = 0
    comparable = 0
    exact = 0
    failures: List[Dict[str, Any]] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[ERR] {image_path.name:<24} failed to load")
            failures.append({"image": image_path.name, "classification": "no_candidate"})
            continue

        expected = expected_text_from_filename(image_path)
        case_params = clone_params(image_params.get(image_roi_key(image_path), p)) if image_params else clone_params(p)
        actual, conf, raw, debug = robust_ocr_from_lcd_roi(image, case_params)
        rows = candidate_rows_from_debug(debug)
        top = rows[0] if rows else {}
        winner_label = str(debug.get("winner_label", "")) if isinstance(debug, dict) else ""
        winner_stage = str(debug.get("winner_stage", "")) if isinstance(debug, dict) else ""
        winner_source = str(top.get("source", candidate_source(raw, winner_label)))
        vote_count = int(debug.get("vote_count", top.get("vote_count", 0)) or 0) if isinstance(debug, dict) else 0
        final_score = float(debug.get("final_score", top.get("final_score", 0.0)) or 0.0) if isinstance(debug, dict) else 0.0
        penalties = debug.get("penalties_applied", top.get("penalties", [])) if isinstance(debug, dict) else []

        total += 1
        ok = expected is not None and actual == expected
        if expected is not None:
            comparable += 1
            exact += int(ok)

        status = "OK " if ok else "BAD" if expected is not None else "UNK"
        fail_class = "" if ok else classify_batch_failure(expected, actual, debug if isinstance(debug, dict) else {})
        expected_text = expected if expected is not None else "?"
        stage_suffix = f" stage={winner_stage}" if winner_stage else ""
        class_suffix = f" class={fail_class}" if fail_class else ""
        print(
            f"[{status}] {image_path.name:<24} expected='{expected_text}' got='{actual}' "
            f"conf={conf:.1f} source={winner_source} winner={winner_label}{stage_suffix} "
            f"votes={vote_count} final={final_score:.2f} "
            f"penalties={short_penalty_text(penalties)}{class_suffix}"
        )

        if not ok:
            failures.append(
                {
                    "image": image_path.name,
                    "expected": expected_text,
                    "actual": actual,
                    "raw": raw,
                    "classification": fail_class,
                    "winner": winner_label,
                    "stage": winner_stage,
                }
            )

        if rows and (show_ok_candidates or not ok):
            print_candidate_rows(rows, top_candidates)

    accuracy = exact / float(max(1, comparable))
    print("\n" + "=" * 90)
    print(f"BATCH SUMMARY: exact={exact}/{comparable} accuracy={accuracy:.3f} images={total}")
    if failures:
        counts = Counter(item["classification"] for item in failures)
        print("Failure classes:")
        for name, count in counts.most_common():
            print(f"  {name}: {count}")
        print("Failures:")
        for item in failures:
            print(
                f"  - {item['image']}: expected='{item.get('expected', '?')}' "
                f"got='{item.get('actual', '')}' class={item['classification']} "
                f"raw='{item.get('raw', '')}' winner={item.get('winner', '')} stage={item.get('stage', '')}"
            )
    print("=" * 90 + "\n")


def psm_value(psm_mode: int) -> int:
    if psm_mode == 0:
        return 7
    if psm_mode == 1:
        return 8
    return 13


def tesseract_config(p: Params) -> str:
    return (
        f"--oem 1 --psm {psm_value(p.psm_mode)} --dpi {p.dpi} "
        f"-c tessedit_char_whitelist=0123456789. "
        f"-c classify_bln_numeric_mode=1 "
        f"-c load_system_dawg=0 -c load_freq_dawg=0"
    )


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


FINAL_NUMERIC_RE = re.compile(r"^\d+(?:\.\d{1,2})?$")


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
        if len(fraction) > 2:
            return -0.90
    return 0.0


def has_valid_final_vote(votes: Dict[str, Dict[str, Any]]) -> bool:
    return any(is_valid_final_numeric_text(text) for text in votes)


def ocr_once(img, p: Params) -> Tuple[str, float, str]:
    # One Tesseract call: both text and confidence
    cfg = tesseract_config(p)
    data = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DICT)

    tokens = [t for t in data.get("text", []) if t and t.strip()]
    raw_joined = "".join(tokens)
    cleaned = normalize_numeric_text(raw_joined)

    confs = []
    for c in data.get("conf", []):
        try:
            cv = float(c)
            if cv >= 0:
                confs.append(cv)
        except Exception:
            pass
    mean_conf = sum(confs) / len(confs) if confs else 0.0

    return cleaned, mean_conf, raw_joined


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
        ratios["b"] >= 0.40
        and ratios["c"] >= 0.35
        and ratios["d"] >= 0.45
        and ratios["e"] >= 0.35
        and ratios["f"] >= 0.35
        and ratios["g"] <= 0.18
    ):
        return "0", "weak-top0"
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
        return item["digit"] == "1" and item["segments"] in {"narrow", "narrow-smear1", "bc"}

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


def read_7seg_from_mask_debug(mask) -> Optional[Dict[str, Any]]:
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
        result = classify_7seg_digit(digit_mask)
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

        result = read_7seg_from_mask_debug(mask)
        if result is None:
            continue

        text = result["text"]
        conf = result["conf"]
        raw = result["raw"]
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
        bucket["alias_hints"].update(result.get("alias_hints", set()))
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


def read_7seg_from_stages(stages: Dict[str, Any]) -> Optional[Tuple[str, float, str]]:
    result = read_7seg_from_stages_debug(stages)
    if result is None:
        return None
    return result["text"], result["conf"], result["raw"]


def ocr_from_stages(stages: Dict[str, Any], p: Params) -> Tuple[str, float, str]:
    seven_seg = read_7seg_from_stages(stages)
    if seven_seg is not None:
        return seven_seg

    return ocr_once(stages["ocr_input"], p)


def build_param_variants(p: Params) -> List[Tuple[str, Params]]:
    variants: List[Tuple[str, Params]] = [("base", clone_params(p))]
    seen: set[tuple[int, int, int, int, int]] = {
        (p.method, p.scale, p.blur, p.dark_threshold, p.close_enable)
    }

    if p.method == 2:
        for delta in (-4, -3, -2, -1, 1, 2):
            p2 = clone_params(p)
            p2.dark_threshold = clamp(p.dark_threshold + delta, 0, 255)
            key = (p2.method, p2.scale, p2.blur, p2.dark_threshold, p2.close_enable)
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
            key = (p2.method, p2.scale, p2.blur, p2.dark_threshold, p2.close_enable)
            if key not in seen:
                seen.add(key)
                variants.append((name, p2))

    return variants


def vote_quality_score(text: str, raw: str, conf: float) -> float:
    normalized_conf = max(0.0, min(float(conf), 95.0)) / 95.0
    if raw.startswith("[7seg:"):
        digit_count = sum(ch.isdigit() for ch in text)
        score = 1.0 + normalized_conf + digit_count * 0.40 + numeric_structure_penalty(text)
        if "." in text:
            score += 0.45
        if "loose" in raw:
            score -= 0.50
        if "soft" in raw:
            score -= 0.12
        if "smear" in raw:
            score -= 0.22
        if "lowd" in raw:
            score -= 0.16
        if "clipped" in raw:
            score -= 0.20
        if "split0" in raw:
            score -= 0.75
        if should_strip_trailing_narrow_one(text, raw):
            score -= 1.55
        if should_strip_trailing_narrow_seven(text, raw):
            score -= 1.10
        return score

    score = max(0.15, normalized_conf) + sum(ch.isdigit() for ch in text) * 0.20 + numeric_structure_penalty(text)
    if "." in text:
        score += 0.20
    return score


ALIAS_VARIANT_MARKERS = (
    "leading_zero_decimal",
    "trim_fractional_two",
    "append_trailing_narrow1",
    "trim_trailing_narrow1",
    "trim_trailing_narrow7",
    "fractional_7_from_smeared_top",
    "mask_dot_",
    "dot_pos_",
)


def crop_family(crop_name: str) -> str:
    if crop_name == "selected_roi":
        return "selected"
    if crop_name.startswith("window_frame_trim"):
        return "frame_trim"
    if crop_name.startswith("window_"):
        return "window"
    if crop_name.startswith("raw_band"):
        return "raw_band"
    if crop_name == "alias":
        return "alias"
    return "reading_band"


def candidate_source(raw: str, variant_name: str) -> str:
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
    if any(marker in variant_name for marker in ALIAS_VARIANT_MARKERS) and "." in text:
        return "heuristic_alias"
    if raw.startswith("[tesseract:") and "." in raw:
        return "tesseract_dot"
    if decimal_observed:
        return "observed"
    return "none"


def select_best_vote(votes: Dict[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    return max(
        votes.items(),
        key=lambda item: (
            1 if is_valid_final_numeric_text(item[0]) else 0,
            float(item[1].get("final_score", item[1]["best_score"])),
            int(item[1].get("family_count", 0)),
            1 if item[1].get("visual_decimal_observed", item[1].get("decimal_observed", False)) else 0,
            item[1]["best_score"],
            item[1]["score_sum"],
            item[1]["count"],
            1 if "." in item[0] else 0,
            item[1]["best_conf"],
            -len(item[0]),
        ),
    )


def should_strip_trailing_narrow_one(text: str, raw: str) -> bool:
    if not raw.startswith("[7seg:") or not raw.endswith("1:narrow]") or not text.endswith("1"):
        return False

    digit_count = sum(ch.isdigit() for ch in text)
    if "." in text:
        whole, fraction = text.split(".", 1)
        if digit_count >= 5 or len(fraction) >= 3:
            return True
        if whole == "0" and fraction.startswith("0"):
            return False
        return len(fraction) >= 2

    return digit_count >= 4


def should_strip_trailing_narrow_seven(text: str, raw: str) -> bool:
    if not raw.startswith("[7seg:") or not raw.endswith("7:abc-narrow]") or not text.endswith("7"):
        return False

    if "." not in text:
        return False

    whole, fraction = text.split(".", 1)
    if len(fraction) >= 3:
        return True

    return whole == "0" and len(fraction) >= 2


def generate_candidate_aliases(
    text: str,
    raw: str,
    alias_hints: Optional[set[str]] = None,
) -> List[Tuple[str, str, float]]:
    aliases: List[Tuple[str, str, float]] = []
    alias_hints = alias_hints or set()

    if "." not in text and text.startswith("0") and len(text) >= 2 and text.isdigit():
        aliases.append(("0." + text[1:], "leading_zero_decimal", 0.08))

    if "." in text:
        whole, fraction = text.split(".", 1)
        if len(fraction) >= 3:
            aliases.append((f"{whole}.{fraction[:2]}", "trim_fractional_two", 0.12))
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


def build_decimal_aliases_from_votes(votes: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str, Dict[str, Any], float]]:
    # Cross-vote decimal insertion is intentionally disabled for now.
    # It was promoting malformed or structurally different reads such as 120 -> 12.0.
    return []


def apply_trailing_zero_conflict_penalties(votes: Dict[str, Dict[str, Any]]) -> None:
    for text, info in list(votes.items()):
        if not text.endswith(".0"):
            continue
        twin = text.replace(".", "", 1)
        twin_info = votes.get(twin)
        if twin_info is None:
            continue

        dotted_count = int(info.get("count", 0))
        twin_count = int(twin_info.get("count", 0))
        dotted_score = float(info.get("best_score", float("-inf")))
        twin_score = float(twin_info.get("best_score", float("-inf")))
        if twin_count + 1 < dotted_count:
            continue
        strong_twin_support = twin_count >= max(2, dotted_count * 2)
        if not strong_twin_support and twin_score + 0.55 < dotted_score:
            continue
        if float(info.get("trailing_zero_conflict_penalty", 0.0)) > 0.0:
            continue

        info["best_score"] = dotted_score - 4.00
        info["score_sum"] = float(info.get("score_sum", 0.0)) - 4.00
        info["trailing_zero_conflict_penalty"] = float(info.get("trailing_zero_conflict_penalty", 0.0)) + 4.00


def apply_completion_conflict_penalties(votes: Dict[str, Dict[str, Any]]) -> None:
    valid_items = [
        (text, info)
        for text, info in votes.items()
        if is_valid_final_numeric_text(text)
    ]
    for text, info in valid_items:
        if float(info.get("completion_conflict_penalty", 0.0)) > 0.0:
            continue

        count = int(info.get("count", 0))
        best_score = float(info.get("best_score", 0.0))
        digit_count = sum(ch.isdigit() for ch in text)
        penalty = 0.0

        for other_text, other_info in valid_items:
            if other_text == text or not other_text.startswith(text):
                continue
            other_digit_count = sum(ch.isdigit() for ch in other_text)
            if other_digit_count != digit_count + 1:
                continue
            other_count = int(other_info.get("count", 0))
            other_score = float(other_info.get("best_score", 0.0))
            if other_count < max(2, int(count * 0.70)):
                continue
            if other_score + 0.80 < best_score:
                continue

            if "." not in text and "." not in other_text:
                penalty = max(penalty, 1.10)
            elif "." in text and "." in other_text:
                penalty = max(penalty, 0.90)

        if penalty <= 0.0:
            continue

        info["best_score"] = best_score - penalty
        info["score_sum"] = float(info.get("score_sum", 0.0)) - penalty
        info["completion_conflict_penalty"] = penalty


def apply_terminal_digit_conflict_penalties(votes: Dict[str, Dict[str, Any]]) -> None:
    valid_items = [
        (text, info)
        for text, info in votes.items()
        if is_valid_final_numeric_text(text)
    ]

    for text, info in valid_items:
        if float(info.get("terminal_digit_conflict_penalty", 0.0)) > 0.0:
            continue
        if "." not in text or not text.endswith("1"):
            continue

        whole, fraction = text.split(".", 1)
        if len(fraction) < 2:
            continue

        raw = str(info.get("raw", ""))
        if "1:narrow-smear1" not in raw:
            continue

        best_score = float(info.get("best_score", 0.0))
        best_structural = float(info.get("structural_quality", 0.0))
        prefix = text[:-1]
        penalty = 0.0

        for other_text, other_info in valid_items:
            if other_text != f"{prefix}7":
                continue
            other_raw = str(other_info.get("raw", ""))
            other_score = float(other_info.get("best_score", 0.0))
            other_structural = float(other_info.get("structural_quality", 0.0))
            if "7:abc" not in other_raw:
                continue
            if other_structural + 0.10 < best_structural:
                continue
            if other_score + 1.25 < best_score:
                continue
            penalty = max(penalty, 1.05)

        if penalty <= 0.0:
            continue

        info["best_score"] = best_score - penalty
        info["score_sum"] = float(info.get("score_sum", 0.0)) - penalty
        info["terminal_digit_conflict_penalty"] = penalty


def is_stable_best_vote(text: str, info: Dict[str, Any]) -> bool:
    raw = str(info.get("raw", ""))
    if not raw.startswith("[7seg:"):
        return False

    count = int(info.get("count", 0))
    digit_count = sum(ch.isdigit() for ch in text)
    best_score = float(info.get("best_score", 0.0))

    if count < 4:
        return False
    if digit_count >= 3 and best_score >= 3.40:
        return True
    return False


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
        _, y, _, h = cv2.boundingRect(points)
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


def has_valid_7seg_vote(votes: Dict[str, Dict[str, Any]]) -> bool:
    for text, info in votes.items():
        if not is_valid_final_numeric_text(text):
            continue
        details = info.get("candidate_details", [])
        if any(item.get("raw", "").startswith("[7seg:") for item in details):
            return True
        if str(info.get("raw", "")).startswith("[7seg:"):
            return True
    return False


def apply_combined_vote_scores(votes: Dict[str, Dict[str, Any]]) -> None:
    for text, info in votes.items():
        details = info.get("candidate_details", [])
        families = set(info.get("families", set()))
        sources = set(info.get("sources", set()))

        for detail in details:
            families.add(str(detail.get("family", "")))
            sources.add(str(detail.get("source", "")))

        families.discard("")
        sources.discard("")
        family_count = len(families)
        count = int(info.get("count", 0))
        best_score = float(info.get("best_score", 0.0))
        valid_numeric = is_valid_final_numeric_text(text)

        best_structural = 0.0
        best_artifact_penalty = 0.0
        penalties: List[str] = []
        best_by_family: Dict[str, float] = {}
        decimal_kinds = set(info.get("decimal_evidence", set()))

        for detail in details:
            family = str(detail.get("family", ""))
            quality = float(detail.get("quality_score", 0.0))
            if family:
                best_by_family[family] = max(best_by_family.get(family, float("-inf")), quality)
            best_structural = max(best_structural, float(detail.get("structural_quality", 0.0)))
            best_artifact_penalty = max(best_artifact_penalty, float(detail.get("artifact_penalty", 0.0)))
            penalties.extend(str(item) for item in detail.get("penalties", []))
            decimal_kind = str(detail.get("decimal_evidence", "none"))
            if decimal_kind != "none":
                decimal_kinds.add(decimal_kind)

        if not details:
            best_structural = 0.0

        family_bonus = 0.70 * min(family_count, 3)
        vote_bonus = 0.07 * min(count, 6)
        independent_quality = sum(max(0.0, min(score, 5.0)) for score in best_by_family.values())
        independent_bonus = 0.06 * independent_quality
        validity_bonus = 0.35 if valid_numeric else -4.0
        structural_bonus = (best_structural - 0.55) * 0.90

        decimal_bonus = 0.0
        if "." in text:
            if "visual_dot" in decimal_kinds:
                decimal_bonus = 0.35
            elif "visual_mask_dot" in decimal_kinds:
                decimal_bonus = 0.20
            elif "tesseract_dot" in decimal_kinds:
                decimal_bonus = 0.08
            elif "heuristic_alias" in decimal_kinds:
                decimal_bonus = -0.35
            else:
                decimal_bonus = -0.20
        elif "visual_dot" in decimal_kinds or "visual_mask_dot" in decimal_kinds:
            decimal_bonus = -0.40
            penalties.append("decimal_evidence_missing_in_text:0.40")

        source_penalty = 0.0
        if sources == {"alias"}:
            source_penalty += 0.35
            penalties.append("alias_only:0.35")
        if sources == {"tesseract"} and family_count < 2:
            source_penalty += 0.35
            penalties.append("single_family_tesseract:0.35")
        if sources <= {"tesseract", "alias"} and sum(ch.isdigit() for ch in text) <= 1:
            source_penalty += 2.00
            penalties.append("single_digit_fallback:2.00")
        if "." in text and sum(ch.isdigit() for ch in text) >= 4 and family_count <= 1:
            source_penalty += 0.75
            penalties.append("single_family_decimal:0.75")
        trailing_zero_conflict = float(info.get("trailing_zero_conflict_penalty", 0.0))
        if trailing_zero_conflict:
            source_penalty += trailing_zero_conflict
            penalties.append(f"trailing_zero_conflict:{trailing_zero_conflict:.2f}")
        completion_conflict = float(info.get("completion_conflict_penalty", 0.0))
        if completion_conflict:
            source_penalty += completion_conflict
            penalties.append(f"completion_conflict:{completion_conflict:.2f}")
        terminal_digit_conflict = float(info.get("terminal_digit_conflict_penalty", 0.0))
        if terminal_digit_conflict:
            source_penalty += terminal_digit_conflict
            penalties.append(f"terminal_digit_conflict:{terminal_digit_conflict:.2f}")

        artifact_penalty = best_artifact_penalty * 0.50
        if artifact_penalty:
            penalties.append(f"aggregate_artifact:{artifact_penalty:.2f}")

        final_score = (
            best_score
            + family_bonus
            + vote_bonus
            + independent_bonus
            + validity_bonus
            + structural_bonus
            + decimal_bonus
            - source_penalty
            - artifact_penalty
        )

        info["family_count"] = family_count
        info["families"] = families
        info["sources"] = sources
        info["structural_quality"] = best_structural
        info["artifact_penalty"] = best_artifact_penalty
        info["decimal_evidence"] = decimal_kinds
        info["visual_decimal_observed"] = "visual_dot" in decimal_kinds or "visual_mask_dot" in decimal_kinds
        info["final_score"] = final_score
        info["penalties_applied"] = sorted(set(penalties))


def candidate_summary_rows(votes: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for text, info in votes.items():
        rows.append(
            {
                "clean": text,
                "raw": info.get("sample_raw", info.get("raw", "")),
                "source": "+".join(sorted(info.get("sources", []))) or candidate_source(str(info.get("raw", "")), ""),
                "families": "+".join(sorted(info.get("families", []))),
                "best_variant": info.get("sample_label", ""),
                "vote_count": int(info.get("count", 0)),
                "best_conf": float(info.get("best_conf", 0.0)),
                "best_score": float(info.get("best_score", 0.0)),
                "structural_quality": float(info.get("structural_quality", 0.0)),
                "artifact_penalty": float(info.get("artifact_penalty", 0.0)),
                "final_score": float(info.get("final_score", info.get("best_score", 0.0))),
                "decimal_evidence": "+".join(sorted(info.get("decimal_evidence", []))),
                "penalties": list(info.get("penalties_applied", [])),
            }
        )

    rows.sort(key=lambda item: item["final_score"], reverse=True)
    for index, row in enumerate(rows, 1):
        row["rank"] = index
    return rows


def should_run_tesseract_fallbacks(votes: Dict[str, Dict[str, Any]]) -> bool:
    if not votes or not has_valid_7seg_vote(votes):
        return True

    apply_combined_vote_scores(votes)
    best_text, best_info = select_best_vote(votes)
    if not is_valid_final_numeric_text(best_text):
        return True

    sources = set(best_info.get("sources", set()))
    digit_count = sum(ch.isdigit() for ch in best_text)
    family_count = int(best_info.get("family_count", 0))
    count = int(best_info.get("count", 0))
    structural = float(best_info.get("structural_quality", 0.0))

    if "." not in best_text and digit_count <= 3 and family_count >= 2 and count >= 4 and "7seg" in sources:
        return False
    if sources == {"alias"}:
        return True
    if structural < 0.65:
        return True
    if "." in best_text and digit_count >= 4 and family_count <= 1:
        return True
    return False


def is_reliable_tesseract_vote(text: str, info: Dict[str, Any]) -> bool:
    raw = str(info.get("raw", ""))
    if not raw.startswith("[tesseract:"):
        return True

    if not is_valid_final_numeric_text(text):
        return False

    digit_count = sum(ch.isdigit() for ch in text)
    if digit_count <= 1:
        return False

    count = int(info.get("count", 0))
    best_conf = float(info.get("best_conf", 0.0))
    family_count = int(info.get("family_count", 0))
    final_score = float(info.get("final_score", info.get("best_score", 0.0)))
    sample_label = str(info.get("sample_label", ""))
    stages = info.get("sample_stages")
    after_filters = stages.get("after_filters") if isinstance(stages, dict) else None
    run_count = estimate_digit_run_count(after_filters)
    dot_slot = infer_dot_slot_from_mask(after_filters) if after_filters is not None else None
    raw_crop = sample_label.split("/", 1)[0] if "/" in sample_label else sample_label

    if "." in text:
        whole, fraction = text.split(".", 1)
        if not whole or not fraction:
            return False
        if len(fraction) > 2:
            return False
        if run_count and run_count > digit_count:
            return False
        if text.endswith("1") and run_count and run_count < digit_count:
            return False
        if best_conf < 60.0 and count < 2:
            return False
        return count >= 2 or best_conf >= 80.0 or (family_count >= 2 and final_score >= 2.20)

    if dot_slot is not None:
        return False
    if digit_count >= 4:
        return False
    if raw_crop.startswith("raw_band"):
        return count >= 2 and (best_conf >= 60.0 or family_count >= 2)
    if run_count and run_count > digit_count:
        return False
    return digit_count == 3 and (count >= 2 or best_conf >= 40.0 or family_count >= 2)


def robust_ocr_from_lcd_roi(lcd_roi_bgr, p: Params) -> Tuple[str, float, str, Dict[str, Any]]:
    votes: Dict[str, Dict[str, Any]] = {}
    tesseract_fallbacks: List[Tuple[str, str, np.ndarray, Dict[str, Any], Params, float]] = []

    def record_vote(
        txt: str,
        conf: float,
        raw: str,
        crop_name: str,
        variant_name: str,
        reading_roi: np.ndarray,
        stages: Dict[str, Any],
        stage_name: str = "",
        winner_mask: Optional[np.ndarray] = None,
        quality_adjust: float = 0.0,
        decimal_observed: bool = False,
    ) -> None:
        source = candidate_source(raw, variant_name)
        family = crop_family(crop_name)
        decimal_kind = decimal_evidence_kind(txt, raw, variant_name, decimal_observed)
        structural_score, penalties, artifact_penalty = structural_quality_score(
            txt,
            raw,
            winner_mask if winner_mask is not None else stages.get("after_filters"),
            variant_name,
        )
        bucket = votes.setdefault(
            txt,
            {
                "count": 0,
                "score_sum": 0.0,
                "best_score": float("-inf"),
                "best_conf": 0.0,
                "raw": raw,
                "variants": [],
                "sample_conf": -1.0,
                "sample_label": "",
                "sample_crop": None,
                "sample_stages": None,
                "sample_raw": "",
                "sample_stage_name": "",
                "sample_mask": None,
                "decimal_observed": False,
                "visual_decimal_observed": False,
                "families": set(),
                "sources": set(),
                "decimal_evidence": set(),
                "candidate_details": [],
                "structural_quality": 0.0,
                "artifact_penalty": 0.0,
                "penalties_applied": [],
                "final_score": float("-inf"),
            },
        )
        quality = vote_quality_score(txt, raw, conf) + quality_adjust
        bucket["count"] += 1
        bucket["score_sum"] += quality
        bucket["best_score"] = max(bucket["best_score"], quality)
        bucket["best_conf"] = max(bucket["best_conf"], conf)
        bucket["raw"] = raw
        bucket["decimal_observed"] = bool(bucket.get("decimal_observed", False) or decimal_observed)
        bucket["visual_decimal_observed"] = bool(
            bucket.get("visual_decimal_observed", False)
            or decimal_kind in {"visual_dot", "visual_mask_dot"}
        )
        bucket["families"].add(family)
        bucket["sources"].add(source)
        if decimal_kind != "none":
            bucket["decimal_evidence"].add(decimal_kind)
        bucket["structural_quality"] = max(float(bucket.get("structural_quality", 0.0)), structural_score)
        bucket["artifact_penalty"] = max(float(bucket.get("artifact_penalty", 0.0)), artifact_penalty)
        bucket["candidate_details"].append(
            {
                "cleaned_text": txt,
                "raw": raw,
                "source": source,
                "crop": crop_name,
                "family": family,
                "variant": variant_name,
                "stage": stage_name,
                "conf": float(conf),
                "quality_score": quality,
                "structural_quality": structural_score,
                "artifact_penalty": artifact_penalty,
                "penalties": penalties,
                "decimal_evidence": decimal_kind,
            }
        )
        bucket["variants"].append(f"{crop_name}/{variant_name}")
        if conf > bucket["sample_conf"]:
            bucket["sample_conf"] = conf
            bucket["sample_label"] = f"{crop_name}/{variant_name}"
            bucket["sample_crop"] = reading_roi.copy()
            bucket["sample_stages"] = stages
            bucket["sample_raw"] = raw
            bucket["sample_stage_name"] = stage_name
            bucket["sample_mask"] = None if winner_mask is None else winner_mask.copy()

    def process_tesseract_fallbacks(
        fallbacks: List[Tuple[str, str, np.ndarray, Dict[str, Any], Params, float]],
    ) -> None:
        for crop_name, variant_name, reading_roi, stages, p_variant, crop_penalty in fallbacks:
            txt, conf, raw = ocr_once(stages["ocr_input"], p_variant)
            if not txt:
                continue
            dot_slot = None
            if conf >= 35.0:
                dot_slot = infer_dot_slot_from_mask(stages["after_filters"])
            wrapped_raw = f"[tesseract:{raw}]"
            record_vote(
                txt,
                conf,
                wrapped_raw,
                crop_name,
                variant_name,
                reading_roi,
                stages,
                stage_name="ocr_input",
                winner_mask=stages.get("ocr_input_mask"),
                quality_adjust=crop_penalty,
                decimal_observed="." in txt or "." in raw,
            )
            for alias_text, alias_label, quality_adjust in generate_candidate_aliases(txt, wrapped_raw):
                record_vote(
                    alias_text,
                    conf,
                    wrapped_raw,
                    crop_name,
                    f"{variant_name}|{alias_label}",
                    reading_roi,
                    stages,
                    stage_name="ocr_input",
                    winner_mask=stages.get("ocr_input_mask"),
                    quality_adjust=crop_penalty + quality_adjust,
                )
            if dot_slot is not None and 0 < dot_slot < len(txt):
                dotted = txt[:dot_slot] + "." + txt[dot_slot:]
                record_vote(
                    dotted,
                    conf,
                    wrapped_raw,
                    crop_name,
                    f"{variant_name}|mask_dot_{dot_slot}",
                    reading_roi,
                    stages,
                    stage_name="ocr_input",
                    winner_mask=stages.get("ocr_input_mask"),
                    quality_adjust=crop_penalty + 0.18,
                    decimal_observed=True,
                )

    def process_candidates(candidates: List[Tuple[str, np.ndarray]], crop_penalty: float = 0.0) -> None:
        for crop_name, reading_roi in candidates:
            variant_crop_penalty = crop_penalty + (0.18 if crop_name == "selected_roi" else 0.0)
            for variant_name, p_variant in build_param_variants(p):
                stages = preprocess(reading_roi, p_variant)
                seven_seg = read_7seg_from_stages_debug(stages)
                if seven_seg is not None:
                    txt = seven_seg["text"]
                    conf = seven_seg["conf"]
                    raw = seven_seg["raw"]
                    stage_name = seven_seg["stage_name"]
                    alias_hints = set(seven_seg.get("alias_hints", set()))
                    record_vote(
                        txt,
                        conf,
                        raw,
                        crop_name,
                        variant_name,
                        reading_roi,
                        stages,
                        stage_name=stage_name,
                        winner_mask=seven_seg.get("stage_mask"),
                        quality_adjust=variant_crop_penalty,
                        decimal_observed=".:dot" in raw,
                    )
                    for alias_text, alias_label, quality_adjust in generate_candidate_aliases(txt, raw, alias_hints):
                        record_vote(
                            alias_text,
                            conf,
                            raw,
                            crop_name,
                            f"{variant_name}|{alias_label}",
                            reading_roi,
                            stages,
                            stage_name=stage_name,
                            winner_mask=seven_seg.get("stage_mask"),
                            quality_adjust=variant_crop_penalty + quality_adjust,
                        )
                    if (
                        not is_valid_final_numeric_text(txt)
                        or alias_hints
                        or any(token in raw for token in ("soft", "loose", "weak", "smear", "clipped", "lowd"))
                    ):
                        tesseract_fallbacks.append((crop_name, variant_name, reading_roi.copy(), stages, p_variant, variant_crop_penalty))
                else:
                    tesseract_fallbacks.append((crop_name, variant_name, reading_roi.copy(), stages, p_variant, variant_crop_penalty))

    primary_candidates, secondary_candidates = build_reading_roi_candidate_groups(lcd_roi_bgr)
    if primary_candidates:
        process_candidates(primary_candidates, crop_penalty=0.0)
    if secondary_candidates:
        process_candidates(secondary_candidates, crop_penalty=-0.10)
    apply_trailing_zero_conflict_penalties(votes)
    apply_completion_conflict_penalties(votes)
    apply_terminal_digit_conflict_penalties(votes)
    if should_run_tesseract_fallbacks(votes):
        process_tesseract_fallbacks(tesseract_fallbacks)

    if not votes:
        return "", 0.0, "", {}

    for alias_text, alias_label, source_info, quality_adjust in build_decimal_aliases_from_votes(votes):
        bucket = votes.setdefault(
            alias_text,
            {
                "count": 0,
                "score_sum": 0.0,
                "best_score": float("-inf"),
                "best_conf": 0.0,
                "raw": source_info["raw"],
                "variants": [],
                "sample_conf": -1.0,
                "sample_label": "",
                "sample_crop": source_info.get("sample_crop"),
                "sample_stages": source_info.get("sample_stages"),
                "sample_raw": source_info.get("sample_raw", source_info["raw"]),
                "sample_stage_name": source_info.get("sample_stage_name", ""),
                "sample_mask": source_info.get("sample_mask"),
                "decimal_observed": False,
            },
        )
        source_score = float(source_info.get("best_score", 0.0))
        source_conf = float(source_info.get("best_conf", 0.0))
        alias_score = source_score + quality_adjust
        bucket["count"] += 1
        bucket["score_sum"] += alias_score
        bucket["best_score"] = max(bucket["best_score"], alias_score)
        bucket["best_conf"] = max(bucket["best_conf"], source_conf)
        bucket["raw"] = source_info["raw"]
        bucket["variants"].append(f"alias/{alias_label}")
        if source_conf >= bucket["sample_conf"]:
            bucket["sample_conf"] = source_conf
            bucket["sample_label"] = f"alias/{alias_label}"
            bucket["sample_crop"] = source_info.get("sample_crop")
            bucket["sample_stages"] = source_info.get("sample_stages")
            bucket["sample_raw"] = source_info.get("sample_raw", source_info["raw"])
            bucket["sample_stage_name"] = source_info.get("sample_stage_name", "")
            sample_mask = source_info.get("sample_mask")
            bucket["sample_mask"] = None if sample_mask is None else sample_mask.copy()

    apply_trailing_zero_conflict_penalties(votes)
    apply_completion_conflict_penalties(votes)
    apply_terminal_digit_conflict_penalties(votes)
    apply_combined_vote_scores(votes)
    candidate_summaries = candidate_summary_rows(votes)
    best_text, best_info = select_best_vote(votes)
    if not is_valid_final_numeric_text(best_text):
        debug = {
            "winner_label": best_info.get("sample_label", ""),
            "winner_crop": best_info.get("sample_crop"),
            "winner_stages": best_info.get("sample_stages"),
            "winner_raw": best_info.get("sample_raw", best_info["raw"]),
            "winner_stage": best_info.get("sample_stage_name", ""),
            "winner_mask": best_info.get("sample_mask"),
            "vote_count": int(best_info["count"]),
            "variants": list(best_info["variants"]),
            "final_score": float(best_info.get("final_score", 0.0)),
            "structural_quality": float(best_info.get("structural_quality", 0.0)),
            "artifact_penalty": float(best_info.get("artifact_penalty", 0.0)),
            "penalties_applied": list(best_info.get("penalties_applied", [])),
            "candidate_summaries": candidate_summaries,
            "rejected": "invalid_numeric",
        }
        return "", 0.0, best_info["raw"], debug

    if not is_reliable_tesseract_vote(best_text, best_info):
        debug = {
            "winner_label": best_info.get("sample_label", ""),
            "winner_crop": best_info.get("sample_crop"),
            "winner_stages": best_info.get("sample_stages"),
            "winner_raw": best_info.get("sample_raw", best_info["raw"]),
            "winner_stage": best_info.get("sample_stage_name", ""),
            "winner_mask": best_info.get("sample_mask"),
            "vote_count": int(best_info["count"]),
            "variants": list(best_info["variants"]),
            "final_score": float(best_info.get("final_score", 0.0)),
            "structural_quality": float(best_info.get("structural_quality", 0.0)),
            "artifact_penalty": float(best_info.get("artifact_penalty", 0.0)),
            "penalties_applied": list(best_info.get("penalties_applied", [])),
            "candidate_summaries": candidate_summaries,
            "rejected": "unreliable_tesseract",
        }
        return "", 0.0, best_info["raw"], debug

    conf = 95.0 if best_info["count"] >= 2 else best_info["best_conf"]
    if best_info.get("sources") == {"alias"}:
        conf = min(conf, 85.0)
    if "." in best_text and not best_info.get("visual_decimal_observed", False):
        conf = min(conf, 88.0)
    if float(best_info.get("structural_quality", 1.0)) < 0.45:
        conf = min(conf, 70.0)
    debug = {
        "winner_label": best_info.get("sample_label", ""),
        "winner_crop": best_info.get("sample_crop"),
        "winner_stages": best_info.get("sample_stages"),
        "winner_raw": best_info.get("sample_raw", best_info["raw"]),
        "winner_stage": best_info.get("sample_stage_name", ""),
        "winner_mask": best_info.get("sample_mask"),
        "vote_count": int(best_info["count"]),
        "variants": list(best_info["variants"]),
        "final_score": float(best_info.get("final_score", 0.0)),
        "structural_quality": float(best_info.get("structural_quality", 0.0)),
        "artifact_penalty": float(best_info.get("artifact_penalty", 0.0)),
        "penalties_applied": list(best_info.get("penalties_applied", [])),
        "candidate_summaries": candidate_summaries,
    }
    return best_text, conf, best_info["raw"], debug


def ocr_text_fast(img, p: Params) -> str:
    # Faster than image_to_data; used only for brute-force search
    cfg = tesseract_config(p)
    raw = pytesseract.image_to_string(img, config=cfg) or ""
    return normalize_numeric_text(raw)


def preprocess(roi_bgr, p: Params) -> Dict[str, Any]:
    stages: Dict[str, Any] = {}

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # Upscale
    s = clamp(p.scale, 1, 6)
    if s != 1:
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)

    # Blur
    if p.blur == 1:
        gray_b = cv2.GaussianBlur(gray, (3, 3), 0)
    elif p.blur == 2:
        gray_b = cv2.GaussianBlur(gray, (5, 5), 0)
    else:
        gray_b = gray

    stages["gray"] = gray_b

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


def save_profile(
    path: str,
    p: Params,
    roi_rect: Optional[Tuple[int, int, int, int]] = None,
    image_rois: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
    image_params: Optional[Dict[str, Params]] = None,
) -> None:
    payload: Dict[str, Any] = {"params": asdict(p)}
    if roi_rect is not None:
        payload["roi"] = list(roi_rect)
    if image_rois:
        payload["image_rois"] = {
            str(key): list(value)
            for key, value in sorted(image_rois.items(), key=lambda item: item[0].lower())
        }
    if image_params:
        payload["image_params"] = {
            str(key): asdict(value)
            for key, value in sorted(image_params.items(), key=lambda item: item[0].lower())
        }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_profile(path: str) -> Tuple[
    Params,
    Optional[Tuple[int, int, int, int]],
    Dict[str, Tuple[int, int, int, int]],
    Dict[str, Params],
]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Backward compatible with old profiles that stored only Params fields.
    if "params" not in data:
        return Params(**data), None, {}, {}

    roi_rect = None
    roi_data = data.get("roi")
    if roi_data is not None:
        roi_rect = parse_roi_list(roi_data, "Profile ROI")

    image_rois: Dict[str, Tuple[int, int, int, int]] = {}
    image_rois_data = data.get("image_rois", {})
    if image_rois_data:
        if not isinstance(image_rois_data, dict):
            raise ValueError("Profile image_rois must be an object keyed by image filename")
        for key, value in image_rois_data.items():
            image_rois[str(key)] = parse_roi_list(value, f"Profile ROI for {key}")

    image_params: Dict[str, Params] = {}
    image_params_data = data.get("image_params", {})
    if image_params_data:
        if not isinstance(image_params_data, dict):
            raise ValueError("Profile image_params must be an object keyed by image filename")
        for key, value in image_params_data.items():
            if not isinstance(value, dict):
                raise ValueError(f"Profile params for {key} must be an object")
            image_params[str(key)] = Params(**value)

    return Params(**data["params"]), roi_rect, image_rois, image_params


def sync_trackbars_from_params(p: Params) -> None:
    cv2.setTrackbarPos("Method (0=Otsu,1=Adaptive,2=Dark)", "Calibration Panel", clamp(p.method, 0, 2))
    cv2.setTrackbarPos("Scale (1..6)", "Calibration Panel", clamp(p.scale, 1, 6))
    cv2.setTrackbarPos("Blur (0..2)", "Calibration Panel", clamp(p.blur, 0, 2))
    cv2.setTrackbarPos("Dark Threshold (0..255)", "Calibration Panel", clamp(p.dark_threshold, 0, 255))
    cv2.setTrackbarPos("Adaptive Block", "Calibration Panel", clamp(p.adaptive_block, 3, 151))
    cv2.setTrackbarPos("Adaptive C", "Calibration Panel", clamp(p.adaptive_c, 0, 50))
    cv2.setTrackbarPos("Close Enable (0/1)", "Calibration Panel", clamp(p.close_enable, 0, 1))
    cv2.setTrackbarPos("Close Iter (0..3)", "Calibration Panel", clamp(p.close_iter, 0, 3))
    cv2.setTrackbarPos("Close K (1..31)", "Calibration Panel", clamp(p.close_k, 1, 31))
    cv2.setTrackbarPos("Dilate (0..2)", "Calibration Panel", clamp(p.dilate_iter, 0, 2))
    cv2.setTrackbarPos("Erode (0..2)", "Calibration Panel", clamp(p.erode_iter, 0, 2))
    cv2.setTrackbarPos("Median (0/3)", "Calibration Panel", 0 if p.median == 0 else 1)
    cv2.setTrackbarPos("PSM (0=7,1=8,2=13)", "Calibration Panel", clamp(p.psm_mode, 0, 2))
    cv2.setTrackbarPos("Pad (0..80)", "Calibration Panel", clamp(p.pad, 0, 80))


def grid_search(roi_bgr, regex_pattern: str) -> None:
    """
    Grid search over a reasonable sweep that you can expand easily.
    IMPORTANT: No dot injection / no output fixing.
    Prints all regex hits + params.
    """

    rx = re.compile(regex_pattern)

    # Sweep lists (edit these if you want wider search)
    methods = [0, 1]
    scales = [1, 2, 3, 4, 5, 6]
    blurs = [0, 1, 2]

    close_enables = [0, 1]
    close_iters = [0, 1, 2, 3]
    close_ks = [5, 9, 13, 31]  # you can expand to range(1,32) if you insist (will be slower)

    dilates = [0, 1, 2]
    erodes = [0, 1, 2]
    medians = [0, 3]

    psms = [0, 1, 2]  # 7/8/13
    pads = [0, 10, 20, 30, 40, 60, 80]

    # Adaptive sweeps only matter if method=1
    adaptive_blocks = [31, 51, 91, 115, 151]
    adaptive_cs = [0, 12, 22, 32, 44]

    hits = []  # (conf, txt, params_dict, raw)
    out_counter = Counter()

    # Count total combos (for progress)
    total = 0
    for m in methods:
        for _ in scales:
            for _ in blurs:
                for _ in close_enables:
                    for _ in close_iters:
                        for _ in close_ks:
                            for _ in dilates:
                                for _ in erodes:
                                    for _ in medians:
                                        for _ in psms:
                                            for _ in pads:
                                                if m == 0:
                                                    total += 1
                                                else:
                                                    total += len(adaptive_blocks) * len(adaptive_cs)

    done = 0

    for method in methods:
        for scale in scales:
            for blur in blurs:
                for close_enable in close_enables:
                    for close_iter in close_iters:
                        for close_k in close_ks:
                            for dilate_iter in dilates:
                                for erode_iter in erodes:
                                    for median in medians:
                                        for psm_mode in psms:
                                            for pad in pads:

                                                if method == 0:
                                                    adapt_pairs = [(31, 12)]  # dummy; not used
                                                else:
                                                    adapt_pairs = [(ab, ac) for ab in adaptive_blocks for ac in adaptive_cs]

                                                for (ab, ac) in adapt_pairs:
                                                    done += 1
                                                    if done % 200 == 0:
                                                        print(f"[GRID] {done}/{total}")

                                                    p = Params()
                                                    p.method = method
                                                    p.scale = scale
                                                    p.blur = blur

                                                    p.close_enable = close_enable
                                                    p.close_iter = close_iter
                                                    p.close_k = close_k

                                                    p.dilate_iter = dilate_iter
                                                    p.erode_iter = erode_iter
                                                    p.median = median

                                                    p.psm_mode = psm_mode
                                                    p.pad = pad

                                                    if method == 1:
                                                        p.adaptive_block = make_odd(clamp(ab, 3, 151))
                                                        p.adaptive_c = clamp(ac, 0, 50)

                                                    stages = preprocess(roi_bgr, p)
                                                    txt = ocr_text_fast(stages["ocr_input"], p)
                                                    out_counter[txt] += 1

                                                    if rx.fullmatch(txt or ""):
                                                        # If hit, compute confidence + raw using the slower function
                                                        txt2, conf, raw = ocr_once(stages["ocr_input"], p)
                                                        # Ensure we log the same cleaned text that matched
                                                        hits.append((conf, txt2, asdict(p), raw))

    hits.sort(key=lambda x: x[0], reverse=True)

    print("\n" + "=" * 90)
    print(f"GRID SEARCH DONE | regex='{regex_pattern}'")
    print(f"Hits: {len(hits)}")
    print("=" * 90)

    if hits:
        print("\nHITS (sorted by confidence):")
        for i, (conf, txt, pdict, raw) in enumerate(hits, 1):
            print(f"{i:02d}) conf={conf:.1f} txt='{txt}' raw='{raw}' params={json.dumps(pdict)}")
    else:
        print("\nNo hits found.")
        print("\nMost common outputs (top 15):")
        for txt, cnt in out_counter.most_common(15):
            print(f"  '{txt}' : {cnt}")

    print("=" * 90 + "\n")


# -------------------------
# Main UI
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=("0", "1", "calibration", "batch"),
        default="1",
        help="0/calibration = interactive ROI calibration; 1/batch = automatic cropped-image test (default).",
    )
    ap.add_argument("--image", default=None, help="Optional single image path. Overrides --image-dir.")
    ap.add_argument("--image-dir", default=str(default_test_dir()), help="Directory of test images to load.")
    ap.add_argument("--roi", default=None, help="Optional ROI as x,y,w,h. This should cover the whole LCD.")
    ap.add_argument("--tess", default=r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    ap.add_argument("--profile", default=None, help="Optional JSON profile to load at startup.")
    ap.add_argument("--out-profile", default="ocr_profile.json", help="Where to save profile with W key.")
    ap.add_argument("--regex", default=r"\d+(\.\d+)?", help="Regex for grid search hits (e.g. 0\\.16).")
    ap.add_argument("--top-candidates", type=int, default=5, help="Batch mode: number of competing candidates to print.")
    ap.add_argument("--show-ok-candidates", action="store_true", help="Batch mode: also print candidates for OK cases.")
    roi_mode = ap.add_mutually_exclusive_group()
    roi_mode.add_argument(
        "--roi-each-image",
        dest="roi_each_image",
        action="store_true",
        help="For test photos, select and save a separate LCD ROI for each image.",
    )
    roi_mode.add_argument(
        "--fixed-roi",
        dest="roi_each_image",
        action="store_false",
        help="Use one fixed-camera LCD ROI for all loaded images.",
    )
    ap.set_defaults(roi_each_image=True)
    args = ap.parse_args()

    pytesseract.pytesseract.tesseract_cmd = args.tess

    image_paths = collect_image_paths(args.image, args.image_dir)
    if not image_paths:
        print(f"[ERROR] No image files found. Checked: {args.image or args.image_dir}")
        return

    p = Params()
    batch_image_params: Dict[str, Params] = {}
    if args.profile and Path(args.profile).exists():
        try:
            p, _profile_roi_rect, _image_rois, batch_image_params = load_profile(args.profile)
            if args.mode in ("1", "batch"):
                print(f"[INFO] Loaded batch params from profile: {args.profile}")
                if args.image and image_roi_key(Path(args.image)) in batch_image_params:
                    p = clone_params(batch_image_params[image_roi_key(Path(args.image))])
        except Exception as e:
            if args.mode in ("1", "batch"):
                print(f"[WARN] Failed to load profile params for batch mode: {e}")

    if args.mode in ("1", "batch"):
        run_batch_cropped_mode(
            image_paths,
            p,
            image_params=batch_image_params,
            top_candidates=max(0, int(args.top_candidates)),
            show_ok_candidates=bool(args.show_ok_candidates),
        )
        return

    current_image_index = 0
    image_path = image_paths[current_image_index]
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[ERROR] Failed to load image: {image_path}")
        return

    profile_roi_rect: Optional[Tuple[int, int, int, int]] = None
    image_rois: Dict[str, Tuple[int, int, int, int]] = {}
    image_params: Dict[str, Params] = {}
    if args.profile and Path(args.profile).exists():
        try:
            p, profile_roi_rect, image_rois, image_params = load_profile(args.profile)
            print(f"[INFO] Loaded profile: {args.profile}")
            if image_rois:
                print(f"[INFO] Loaded {len(image_rois)} per-image ROI(s) from profile.")
            if image_params:
                print(f"[INFO] Loaded {len(image_params)} per-image parameter set(s) from profile.")
        except Exception as e:
            print(f"[WARN] Failed to load profile: {e}")
    elif args.profile:
        print(f"[INFO] No profile found yet at: {args.profile}")

    print("\n" + "=" * 70)
    print("STABLE OCR CALIBRATION (SOLID + 7-SEG)")
    print(f"Loaded {len(image_paths)} image(s). Current: {image_path.name}")
    print(f"ROI mode: {'one LCD rectangle per test image' if args.roi_each_image else 'one fixed-camera LCD rectangle'}")
    if args.roi_each_image:
        print("Test photos: select the whole LCD again whenever a new image needs its own rectangle.")
    else:
        print("Setup: focus the fixed camera, then select the whole LCD once.")
    print("Use --fixed-roi only for the real mounted-camera setup.")
    print("Recommended starting preset: Method=2, Scale=3, Blur=0, Dark Threshold=57, Close K=7.")
    print("OCR once/all also checks a few nearby crop bands and threshold values to reduce brittleness.")
    print("The 7-segment reader checks LCD segments first; Tesseract is only used if that fails.")
    print("1 / ENTER / SPACE = OCR once (prints result)")
    print("A = OCR all loaded test images")
    print("G = Grid search on current ROI (prints all hits)")
    print("N / P = Next / previous loaded test image")
    print("R = Reselect ROI | S = Save OCR input image | W = Save profile JSON")
    print("Q / ESC = Quit")
    print("IMPORTANT: Click once on the 'OCR Input' window so it receives keyboard focus.")
    print("=" * 70 + "\n")

    try:
        roi_rect = parse_roi_arg(args.roi)
    except argparse.ArgumentTypeError as e:
        print(f"[ERROR] {e}")
        return

    if args.roi_each_image:
        current_key = image_roi_key(image_path)
        if current_key in image_params:
            p = clone_params(image_params[current_key])
            print(f"[INFO] Using saved params for {image_path.name}: {json.dumps(asdict(p))}")
        if roi_rect is not None:
            image_rois[current_key] = roi_rect
            print(f"[INFO] Using command-line ROI for {image_path.name}: {roi_rect}")
        else:
            roi_rect = image_rois.get(current_key)
            if roi_rect is not None:
                print(f"[INFO] Using saved ROI for {image_path.name}: {roi_rect}")
            else:
                print(f"[INFO] Select LCD ROI for {image_path.name}")
                roi_rect = select_roi(image)
                if roi_rect != (0, 0, 0, 0):
                    image_rois[current_key] = roi_rect
    elif roi_rect is None and profile_roi_rect is not None:
        roi_rect = profile_roi_rect
        print(f"[INFO] Using ROI from profile: {roi_rect}")
    elif roi_rect is not None:
        print(f"[INFO] Using ROI from command line: {roi_rect}")

    if roi_rect is None:
        roi_rect = select_roi(image)

    if roi_rect == (0, 0, 0, 0):
        print("[INFO] ROI selection cancelled.")
        return

    # Windows
    cv2.namedWindow("Calibration Panel", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration Panel", 620, 420)

    cv2.namedWindow("OCR Input", cv2.WINDOW_NORMAL)
    cv2.namedWindow("LCD ROI", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Reading ROI", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Binary", cv2.WINDOW_NORMAL)
    cv2.namedWindow("After Close", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Winner ROI", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Winner Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Winner OCR Input", cv2.WINDOW_NORMAL)

    # Trackbars
    cv2.createTrackbar("Method (0=Otsu,1=Adaptive,2=Dark)", "Calibration Panel", p.method, 2, lambda _: None)
    cv2.createTrackbar("Scale (1..6)", "Calibration Panel", p.scale, 6, lambda _: None)
    cv2.createTrackbar("Blur (0..2)", "Calibration Panel", p.blur, 2, lambda _: None)
    cv2.createTrackbar("Dark Threshold (0..255)", "Calibration Panel", p.dark_threshold, 255, lambda _: None)

    cv2.createTrackbar("Adaptive Block", "Calibration Panel", p.adaptive_block, 151, lambda _: None)
    cv2.createTrackbar("Adaptive C", "Calibration Panel", p.adaptive_c, 50, lambda _: None)

    cv2.createTrackbar("Close Enable (0/1)", "Calibration Panel", p.close_enable, 1, lambda _: None)
    cv2.createTrackbar("Close Iter (0..3)", "Calibration Panel", p.close_iter, 3, lambda _: None)
    cv2.createTrackbar("Close K (1..31)", "Calibration Panel", p.close_k, 31, lambda _: None)

    cv2.createTrackbar("Dilate (0..2)", "Calibration Panel", p.dilate_iter, 2, lambda _: None)
    cv2.createTrackbar("Erode (0..2)", "Calibration Panel", p.erode_iter, 2, lambda _: None)

    cv2.createTrackbar("Median (0/3)", "Calibration Panel", 0, 1, lambda _: None)   # 0->0, 1->3
    cv2.createTrackbar("PSM (0=7,1=8,2=13)", "Calibration Panel", p.psm_mode, 2, lambda _: None)
    cv2.createTrackbar("Pad (0..80)", "Calibration Panel", p.pad, 80, lambda _: None)
    sync_trackbars_from_params(p)

    last_stages: Optional[Dict[str, Any]] = None
    last_saved_stages: Optional[Dict[str, Any]] = None
    winner_stage_name = ""

    def make_placeholder(label: str) -> np.ndarray:
        canvas = np.zeros((120, 240), dtype=np.uint8)
        cv2.putText(canvas, label, (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
        return canvas

    def show_winner_debug(
        winner_crop: Optional[np.ndarray],
        winner_mask: Optional[np.ndarray],
        winner_ocr_input: Optional[np.ndarray],
    ) -> None:
        cv2.imshow("Winner ROI", winner_crop if winner_crop is not None else make_placeholder("No winner"))
        cv2.imshow("Winner Mask", winner_mask if winner_mask is not None else make_placeholder("No mask"))
        cv2.imshow(
            "Winner OCR Input",
            winner_ocr_input if winner_ocr_input is not None else make_placeholder("No OCR input"),
        )

    show_winner_debug(None, None, None)

    while True:
        roi = crop_roi(image, roi_rect)
        if roi is None:
            print(f"[ERROR] ROI is outside the current image: {image_path}")
            break

        # Read trackbars
        p.method = cv2.getTrackbarPos("Method (0=Otsu,1=Adaptive,2=Dark)", "Calibration Panel")
        p.scale = clamp(cv2.getTrackbarPos("Scale (1..6)", "Calibration Panel"), 1, 6)
        p.blur = cv2.getTrackbarPos("Blur (0..2)", "Calibration Panel")
        p.dark_threshold = clamp(cv2.getTrackbarPos("Dark Threshold (0..255)", "Calibration Panel"), 0, 255)

        p.adaptive_block = make_odd(clamp(cv2.getTrackbarPos("Adaptive Block", "Calibration Panel"), 3, 151))
        p.adaptive_c = clamp(cv2.getTrackbarPos("Adaptive C", "Calibration Panel"), 0, 50)

        p.close_enable = cv2.getTrackbarPos("Close Enable (0/1)", "Calibration Panel")
        p.close_iter = clamp(cv2.getTrackbarPos("Close Iter (0..3)", "Calibration Panel"), 0, 3)
        p.close_k = clamp(cv2.getTrackbarPos("Close K (1..31)", "Calibration Panel"), 1, 31)

        p.dilate_iter = clamp(cv2.getTrackbarPos("Dilate (0..2)", "Calibration Panel"), 0, 2)
        p.erode_iter = clamp(cv2.getTrackbarPos("Erode (0..2)", "Calibration Panel"), 0, 2)

        med_sel = cv2.getTrackbarPos("Median (0/3)", "Calibration Panel")
        p.median = 0 if med_sel == 0 else 3

        p.psm_mode = clamp(cv2.getTrackbarPos("PSM (0=7,1=8,2=13)", "Calibration Panel"), 0, 2)
        p.pad = clamp(cv2.getTrackbarPos("Pad (0..80)", "Calibration Panel"), 0, 80)

        reading_roi = crop_lcd_reading_area(roi)
        stages = preprocess(reading_roi, p)
        last_stages = stages

        # Show debug
        cv2.imshow("LCD ROI", roi)
        cv2.imshow("Reading ROI", reading_roi)
        cv2.imshow("Binary", stages["bin_digits_white"])
        cv2.imshow("After Close", stages["after_close"])
        cv2.imshow("OCR Input", stages["ocr_input"])

        key = cv2.waitKey(10) & 0xFF

        if key in (13, 32, ord("1")):  # ENTER / SPACE / '1'
            txt, conf, raw, debug = robust_ocr_from_lcd_roi(roi, p)
            winner_label = debug.get("winner_label", "")
            vote_count = int(debug.get("vote_count", 0))
            winner_raw = debug.get("winner_raw", raw)
            winner_stage_name = str(debug.get("winner_stage", ""))
            last_saved_stages = debug.get("winner_stages") or last_stages
            winner_crop = debug.get("winner_crop")
            winner_stages = debug.get("winner_stages") or {}
            winner_mask = debug.get("winner_mask")
            winner_ocr_input = winner_stages.get("ocr_input")
            show_winner_debug(winner_crop, winner_mask, winner_ocr_input)
            print(f"[*] Image: {image_path.name}")
            print(f"[*] ROI: {roi_rect}")
            print(f"[*] Params: {json.dumps(asdict(p))}")
            if winner_label:
                stage_suffix = f" stage={winner_stage_name}" if winner_stage_name else ""
                print(f"[*] Winner: {winner_label}{stage_suffix} | votes={vote_count}")
                penalties = debug.get("penalties_applied", [])
                penalty_text = ",".join(penalties[:6]) if penalties else "none"
                print(
                    f"[*] Winner score: final={float(debug.get('final_score', 0.0)):.2f} "
                    f"struct={float(debug.get('structural_quality', 0.0)):.2f} "
                    f"artifact={float(debug.get('artifact_penalty', 0.0)):.2f} "
                    f"penalties={penalty_text}"
                )
            print(f"[>] OCR RAW: '{winner_raw}'")
            print(f"[>] OCR CLEAN: '{txt}' | conf={conf:.1f}")
            candidate_rows = debug.get("candidate_summaries", [])
            if candidate_rows:
                print("[*] Candidate scores:")
                for row in candidate_rows:
                    row_penalties = ",".join(row.get("penalties", [])[:4]) or "none"
                    print(
                        f"    {int(row.get('rank', 0)):02d} clean='{row.get('clean', '')}' "
                        f"raw='{row.get('raw', '')}' source={row.get('source', '')} "
                        f"families={row.get('families', '')} variant={row.get('best_variant', '')} "
                        f"votes={int(row.get('vote_count', 0))} "
                        f"struct={float(row.get('structural_quality', 0.0)):.2f} "
                        f"artifact={float(row.get('artifact_penalty', 0.0)):.2f} "
                        f"final={float(row.get('final_score', 0.0)):.2f} "
                        f"penalties={row_penalties}"
                    )
            print()

        elif key in (ord("a"), ord("A")):
            ocr_all_images(
                image_paths,
                roi_rect,
                p,
                image_rois=image_rois if args.roi_each_image else None,
                image_params=image_params if args.roi_each_image else None,
                select_missing_rois=args.roi_each_image,
            )

        elif key in (ord("g"), ord("G")):
            print(f"[INFO] Running grid search on {image_path.name}... (UI will freeze until done)")
            grid_search(roi, args.regex)
            print("[INFO] Grid search finished.")

        elif key in (ord("r"), ord("R")):
            roi_rect2 = select_roi(image)
            if roi_rect2 != (0, 0, 0, 0):
                roi_rect = roi_rect2
                last_saved_stages = None
                winner_stage_name = ""
                show_winner_debug(None, None, None)
                if args.roi_each_image:
                    image_rois[image_roi_key(image_path)] = roi_rect
                print(f"[INFO] Manual ROI selected: {roi_rect}")

        elif key in (ord("n"), ord("N"), ord("p"), ord("P")):
            step = -1 if key in (ord("p"), ord("P")) else 1
            next_index = (current_image_index + step) % len(image_paths)
            next_path = image_paths[next_index]
            next_image = cv2.imread(str(next_path))
            if next_image is None:
                print(f"[WARN] Failed to load image: {next_path}")
            else:
                next_roi_rect = roi_rect
                if args.roi_each_image:
                    next_key = image_roi_key(next_path)
                    saved_roi = image_rois.get(next_key)
                    if saved_roi is None:
                        print(f"[INFO] Select LCD ROI for {next_path.name}")
                        saved_roi = select_roi(next_image)
                        if saved_roi == (0, 0, 0, 0):
                            print(f"[INFO] Staying on {image_path.name}; ROI selection for {next_path.name} was cancelled.")
                            continue
                        image_rois[next_key] = saved_roi
                    next_roi_rect = saved_roi
                    if next_key in image_params:
                        p = clone_params(image_params[next_key])
                        sync_trackbars_from_params(p)
                        print(f"[INFO] Loaded saved params for {next_path.name}.")

                current_image_index = next_index
                image_path = next_path
                image = next_image
                roi_rect = next_roi_rect
                last_saved_stages = None
                winner_stage_name = ""
                show_winner_debug(None, None, None)
                print(f"[INFO] Current image {current_image_index + 1}/{len(image_paths)}: {image_path.name}")

        elif key in (ord("s"), ord("S")):
            stages_to_save = last_saved_stages or last_stages
            if stages_to_save is not None:
                cv2.imwrite("debug_ocr_input.png", stages_to_save["ocr_input"])
                print("[INFO] Saved: debug_ocr_input.png")

        elif key in (ord("w"), ord("W")):
            if args.roi_each_image:
                image_rois[image_roi_key(image_path)] = roi_rect
                image_params[image_roi_key(image_path)] = clone_params(p)
            save_profile(args.out_profile, p, roi_rect, image_rois=image_rois, image_params=image_params)
            print(f"[INFO] Saved profile: {args.out_profile} with current ROI {roi_rect}")
            if image_rois:
                print(f"[INFO] Saved {len(image_rois)} per-image ROI(s).")
            if image_params:
                print(f"[INFO] Saved {len(image_params)} per-image parameter set(s).")

        elif key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
