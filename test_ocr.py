import cv2
import pytesseract
import re
import json
import argparse
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
)
FULL_ROI_READING_REGION_VARIANTS = (
    ("raw_band1", (0.00, 0.00, 0.90, 0.85)),
    ("raw_band2", (0.00, 0.00, 0.95, 0.90)),
    ("raw_band3", (0.00, 0.05, 0.95, 0.92)),
)


def default_test_dir() -> Path:
    return Path(__file__).resolve().parent / "test"


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
    seen_shapes = set()
    for name, region in READING_REGION_VARIANTS:
        cropped = crop_relative_region(lcd_bgr, region, margin_x_ratio=0.03, margin_y_ratio=0.05)
        if cropped.size == 0:
            continue
        shape_key = tuple(cropped.shape[:2])
        if shape_key in seen_shapes:
            continue
        seen_shapes.add(shape_key)
        candidates.append((name, cropped))

    for name, region in FULL_ROI_READING_REGION_VARIANTS:
        cropped = crop_relative_region(lcd_bgr, region, margin_x_ratio=0.0, margin_y_ratio=0.0)
        if cropped.size == 0:
            continue
        shape_key = tuple(cropped.shape[:2])
        if shape_key in seen_shapes:
            continue
        seen_shapes.add(shape_key)
        candidates.append((name, cropped))

    if not candidates:
        candidates.append(("fallback", lcd_bgr))

    return candidates


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
        vote_count = int(debug.get("vote_count", 0))
        winner_suffix = f" via={winner_label} votes={vote_count}" if winner_label else ""
        print(
            f"{i:02d}) {image_path.name}: roi={roi_rect} clean='{txt}' conf={conf:.1f} "
            f"raw='{raw}'{winner_suffix} params={json.dumps(asdict(current_params))}"
        )

    print("=" * 70 + "\n")


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

    return cleaned.strip(".")


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


def clean_7seg_mask(mask):
    cleaned = mask.copy()
    cleaned[:] = 0
    img_h, img_w = mask.shape[:2]
    min_area = max(20, int(img_h * img_w * 0.001))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])
        if area < min_area:
            continue
        if w <= 2 or h <= 2:
            continue
        cleaned[labels == i] = 255

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
    for x1, x2 in runs[1:]:
        prev_x1, prev_x2 = merged[-1]
        if x1 - prev_x2 <= max_gap:
            merged[-1] = (prev_x1, x2)
        else:
            merged.append((x1, x2))

    return merged


def segment_is_active(digit_mask, label: str, zone: Tuple[float, float, float, float]) -> bool:
    h, w = digit_mask.shape[:2]
    left, top, right, bottom = zone
    x1 = clamp(int(w * left), 0, w - 1)
    y1 = clamp(int(h * top), 0, h - 1)
    x2 = clamp(int(w * right), x1 + 1, w)
    y2 = clamp(int(h * bottom), y1 + 1, h)
    segment = digit_mask[y1:y2, x1:x2]
    threshold = SEGMENT_ACTIVE_THRESHOLDS.get(label, 0.25)
    return cv2.countNonZero(segment) / float(segment.size) >= threshold


def classify_7seg_digit(digit_mask) -> Optional[Tuple[str, str]]:
    h, w = digit_mask.shape[:2]
    if h < 10 or w < 3:
        return None

    aspect_ratio = w / float(h)

    # A seven-segment "1" is intentionally narrow: only the right-side bars.
    if aspect_ratio <= 0.30:
        return "1", "narrow"

    active = frozenset(
        label
        for label, zone in SEGMENT_ZONES.items()
        if segment_is_active(digit_mask, label, zone)
    )
    # When the crop merges most of the display into one blob, only the right-side
    # bars may survive the segment test. That can look like "bc", but the shape
    # is far too wide to be a real 7-segment "1".
    if active == frozenset("bc") and aspect_ratio >= 0.50:
        return None
    if active == frozenset("bcdg") and aspect_ratio >= 0.32:
        return "4", "bcdg-loose"
    digit = SEGMENT_DIGITS.get(active)
    if digit is None:
        return None

    return digit, "".join(sorted(active))


def read_7seg_from_mask(mask) -> Optional[Tuple[str, float, str]]:
    cleaned = clean_7seg_mask(mask)
    if cv2.countNonZero(cleaned) == 0:
        return None

    img_h = cleaned.shape[0]
    img_w = cleaned.shape[1]
    candidates: List[Dict[str, Any]] = []

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
            return None

        digit, segments = result
        candidates.append({
            "digit": digit,
            "segments": segments,
            "x1": x1,
            "x2": x2,
            "w": digit_mask.shape[1],
            "h": digit_mask.shape[0],
            "area": cv2.countNonZero(digit_mask),
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
            abnormal_shape = item["h"] > max_h * 1.10 or item["area"] < max_area * 0.75
            tiny_artifact = item["area"] < max_area * 0.18 and item["h"] < max_h * 0.80
            huge_merged_artifact = item["h"] > max_h * 1.45 or item["area"] > max_area * 2.20
            is_edge_narrow_artifact = (
                item["digit"] == "1"
                and item["segments"] == "narrow"
                and edge_side
                and (
                    (
                        item["w"] <= max(12, int(max_w * 0.35))
                        and (tiny_artifact or (sparse_gap and abnormal_shape))
                    )
                    or huge_merged_artifact
                )
            )
            if not is_edge_narrow_artifact:
                filtered.append(item)
        candidates = filtered

    if not candidates:
        return None

    digits = [item["digit"] for item in candidates]
    details = [f"{item['digit']}:{item['segments']}" for item in candidates]
    confidence = 95.0 if all(":" in item for item in details) else 85.0
    return "".join(digits), confidence, f"[7seg:{','.join(details)}]"


def read_7seg_from_stages(stages: Dict[str, Any]) -> Optional[Tuple[str, float, str]]:
    stage_priority = [
        "after_filters",
        "after_close",
        "bin_digits_white",
    ]

    votes: Dict[str, Dict[str, Any]] = {}
    for stage_name in stage_priority:
        mask = stages.get(stage_name)
        if mask is None:
            continue

        result = read_7seg_from_mask(mask)
        if result is None:
            continue

        text, conf, raw = result
        bucket = votes.setdefault(text, {"count": 0, "best_conf": 0.0, "raw": raw, "stage_rank": 999})
        bucket["count"] += 1
        bucket["best_conf"] = max(bucket["best_conf"], conf)
        bucket["raw"] = raw
        bucket["stage_rank"] = min(bucket["stage_rank"], stage_priority.index(stage_name))

    if not votes:
        return None

    best_text, best_info = max(
        votes.items(),
        key=lambda item: (item[1]["count"], item[1]["best_conf"], -item[1]["stage_rank"]),
    )
    conf = 95.0 if best_info["count"] >= 2 else 85.0
    return best_text, conf, best_info["raw"]


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


def vote_quality_score(raw: str, conf: float) -> float:
    normalized_conf = max(0.0, min(float(conf), 95.0)) / 95.0
    if raw.startswith("[7seg:"):
        return 1.0 + normalized_conf
    return max(0.15, normalized_conf)


def robust_ocr_from_lcd_roi(lcd_roi_bgr, p: Params) -> Tuple[str, float, str, Dict[str, Any]]:
    votes: Dict[str, Dict[str, Any]] = {}

    for crop_name, reading_roi in build_reading_roi_candidates(lcd_roi_bgr):
        for variant_name, p_variant in build_param_variants(p):
            stages = preprocess(reading_roi, p_variant)
            txt, conf, raw = ocr_from_stages(stages, p_variant)
            if not txt:
                continue

            bucket = votes.setdefault(
                txt,
                {
                    "count": 0,
                    "score_sum": 0.0,
                    "best_conf": 0.0,
                    "raw": raw,
                    "variants": [],
                    "sample_conf": -1.0,
                    "sample_label": "",
                    "sample_crop": None,
                    "sample_stages": None,
                    "sample_raw": "",
                },
            )
            bucket["count"] += 1
            bucket["score_sum"] += vote_quality_score(raw, conf)
            bucket["best_conf"] = max(bucket["best_conf"], conf)
            bucket["raw"] = raw
            bucket["variants"].append(f"{crop_name}/{variant_name}")
            if conf > bucket["sample_conf"]:
                bucket["sample_conf"] = conf
                bucket["sample_label"] = f"{crop_name}/{variant_name}"
                bucket["sample_crop"] = reading_roi.copy()
                bucket["sample_stages"] = stages
                bucket["sample_raw"] = raw

    if not votes:
        return "", 0.0, "", {}

    best_text, best_info = max(
        votes.items(),
        key=lambda item: (item[1]["score_sum"], item[1]["count"], item[1]["best_conf"], -len(item[0])),
    )
    conf = 95.0 if best_info["count"] >= 2 else best_info["best_conf"]
    debug = {
        "winner_label": best_info.get("sample_label", ""),
        "winner_crop": best_info.get("sample_crop"),
        "winner_stages": best_info.get("sample_stages"),
        "winner_raw": best_info.get("sample_raw", best_info["raw"]),
        "vote_count": int(best_info["count"]),
        "variants": list(best_info["variants"]),
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
    ap.add_argument("--image", default=None, help="Optional single image path. Overrides --image-dir.")
    ap.add_argument("--image-dir", default=str(default_test_dir()), help="Directory of test images to load.")
    ap.add_argument("--roi", default=None, help="Optional ROI as x,y,w,h. This should cover the whole LCD.")
    ap.add_argument("--tess", default=r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    ap.add_argument("--profile", default=None, help="Optional JSON profile to load at startup.")
    ap.add_argument("--out-profile", default="ocr_profile.json", help="Where to save profile with W key.")
    ap.add_argument("--regex", default=r"\d+(\.\d+)?", help="Regex for grid search hits (e.g. 0\\.16).")
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

    current_image_index = 0
    image_path = image_paths[current_image_index]
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[ERROR] Failed to load image: {image_path}")
        return

    p = Params()
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
            last_saved_stages = debug.get("winner_stages") or last_stages
            print(f"[*] Image: {image_path.name}")
            print(f"[*] ROI: {roi_rect}")
            print(f"[*] Params: {json.dumps(asdict(p))}")
            if winner_label:
                print(f"[*] Winner: {winner_label} | votes={vote_count}")
            print(f"[>] OCR RAW: '{winner_raw}'")
            print(f"[>] OCR CLEAN: '{txt}' | conf={conf:.1f}\n")

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
