import cv2
import json
import os
import sys
import time
import re
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

# Import pytesseract (ensure Tesseract-OCR is installed on your system PATH)
try:
    import pytesseract
except ImportError:
    print("Error: pytesseract not found. Please install it using 'pip install pytesseract'")
    print("Also ensure Tesseract-OCR engine is installed on your system and its path is in environment variables.")
    sys.exit(1)

# --- GLOBAL CONFIGURATION (from setup.py or for run.py specific logic) ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(PROJECT_DIR, 'config.json')

# FOR PC_TEST_MODE IN run.py:
# If you want to simulate live capture on PC, you can specify a directory
# containing multiple test images and the script will cycle through them.
# If set to None, it will expect a single test_image_path from config.json (used by setup.py)
PC_TEST_IMAGE_DIR = os.path.join(PROJECT_DIR, 'test_v2') # Set to a directory with test images, or None
PC_TEST_IMAGE_INDEX = 0 # To keep track of which image to load next if using PC_TEST_IMAGE_DIR
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
DARK_PIXEL_THRESHOLD = 57
OCR_SCALE = 3
OCR_BLUR = 0
ROBUST_PREPROCESS_VARIANTS = (
    ("low2", 53, 3, 0, 1),
    ("low", 55, 3, 0, 1),
    ("base", 57, 3, 0, 1),
    ("base_blur", 57, 3, 2, 1),
    ("noise_safe", 57, 1, 2, 0),
    ("mid", 60, 3, 0, 1),
    ("noise_safe_hi", 62, 1, 2, 0),
)

# --- Functions for Image Acquisition ---

def _get_image_from_pi_camera(resolution: tuple) -> cv2.Mat | None:
    """
    Captures a live image from the Raspberry Pi camera.
    Returns an OpenCV image (numpy array) or None on failure.
    """
    try:
        from picamera2 import Picamera2
        print('Capturing image from Raspberry Pi Camera...')
        picam2 = Picamera2()
        camera_config = picam2.create_still_configuration(main={'size': resolution})
        picam2.configure(camera_config)
        picam2.start()
        time.sleep(1) # Warm-up time
        image_np = picam2.capture_array() # Capture as a NumPy array (OpenCV format)
        picam2.stop()
        picam2.close()
        print('Image captured from Pi camera.')
        return cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) # Convert to BGR for OpenCV compatibility if needed
    except ImportError:
        print('Error: `picamera2` not found. Cannot capture from Pi camera.')
        return None
    except Exception as e:
        print(f'Error capturing image from Pi camera: {e}')
        return None

def _get_image_for_pc_test_mode(image_path_or_dir: str | None) -> cv2.Mat | None:
    """
    Loads an image for PC test mode. If image_path_or_dir is a directory,
    it cycles through images in that directory.
    """
    global PC_TEST_IMAGE_INDEX # We need to modify this global variable

    if image_path_or_dir is None: # Fallback to single image from config if no test dir specified
        config_data = load_configuration()
        if config_data and "initial_image_for_roi" in config_data:
            img_path = config_data["initial_image_for_roi"]
            print(f"Loading single test image from config: {img_path}")
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error: Could not load image at {img_path}.")
            return img
        else:
            print("Error: No test image path found in config for PC Test Mode and no directory specified.")
            return None

    if os.path.isdir(image_path_or_dir):
        image_files = sorted([f for f in os.listdir(image_path_or_dir)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        if not image_files:
            print(f"Error: No image files found in test directory: {image_path_or_dir}")
            return None

        current_image_file = image_files[PC_TEST_IMAGE_INDEX % len(image_files)]
        img_path = os.path.join(image_path_or_dir, current_image_file)
        PC_TEST_IMAGE_INDEX += 1 # Move to the next image for the next cycle

        print(f"Loading test image from directory: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not load image at {img_path}.")
        return img
    elif os.path.isfile(image_path_or_dir): # If it's a single file path
        print(f"Loading test image from path: {image_path_or_dir}")
        img = cv2.imread(image_path_or_dir)
        if img is None:
            print(f"Error: Could not load image at {image_path_or_dir}.")
        return img
    else:
        print(f"Error: Invalid PC test image path or directory: {image_path_or_dir}")
        return None

def get_image(config: dict) -> cv2.Mat | None:
    """
    Acquires an image based on the PC_TEST_MODE setting from the configuration.
    """
    if config.get("PC_TEST_MODE", False):
        return _get_image_for_pc_test_mode(PC_TEST_IMAGE_DIR)
    else:
        return _get_image_from_pi_camera(config.get("rpi_camera_resolution", (1920, 1080)))

# --- Functions for Configuration Management ---

def load_configuration() -> dict | None:
    """Loads configuration from config.json."""
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: Configuration file '{CONFIG_FILE}' not found.")
        print("Please run setup.py first to create the configuration.")
        return None
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            return config
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{CONFIG_FILE}': {e}")
        return None
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None

# --- Functions for OCR and Data Processing ---

def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def crop_relative_region(
    lcd_image: cv2.Mat,
    region: tuple[float, float, float, float],
    margin_x_ratio: float = 0.03,
    margin_y_ratio: float = 0.05,
) -> cv2.Mat:
    h, w = lcd_image.shape[:2]
    if h < 20 or w < 20:
        return lcd_image

    if margin_x_ratio > 0.0 or margin_y_ratio > 0.0:
        margin_x = max(2, int(w * margin_x_ratio))
        margin_y = max(2, int(h * margin_y_ratio))
        inner = lcd_image[margin_y:h - margin_y, margin_x:w - margin_x]
        if inner.size == 0:
            return lcd_image
    else:
        inner = lcd_image

    inner_h, inner_w = inner.shape[:2]
    left, top, right, bottom = region
    x1 = _clamp(int(inner_w * left), 0, inner_w)
    y1 = _clamp(int(inner_h * top), 0, inner_h)
    x2 = _clamp(int(inner_w * right), 0, inner_w)
    y2 = _clamp(int(inner_h * bottom), 0, inner_h)

    if x2 <= x1 or y2 <= y1:
        return inner

    return inner[y1:y2, x1:x2].copy()


def crop_lcd_reading_area(lcd_image: cv2.Mat) -> cv2.Mat:
    """Backward-compatible default reading-area crop."""
    return crop_relative_region(lcd_image, LCD_READING_REGION)


def build_reading_roi_candidates(lcd_image: cv2.Mat) -> list[tuple[str, cv2.Mat]]:
    candidates: list[tuple[str, cv2.Mat]] = []
    seen_shapes: set[tuple[int, int]] = set()
    for name, region in READING_REGION_VARIANTS:
        cropped = crop_relative_region(lcd_image, region, margin_x_ratio=0.03, margin_y_ratio=0.05)
        if cropped.size == 0:
            continue
        shape_key = tuple(cropped.shape[:2])
        if shape_key in seen_shapes:
            continue
        seen_shapes.add(shape_key)
        candidates.append((name, cropped))

    for name, region in FULL_ROI_READING_REGION_VARIANTS:
        cropped = crop_relative_region(lcd_image, region, margin_x_ratio=0.0, margin_y_ratio=0.0)
        if cropped.size == 0:
            continue
        shape_key = tuple(cropped.shape[:2])
        if shape_key in seen_shapes:
            continue
        seen_shapes.add(shape_key)
        candidates.append((name, cropped))

    if not candidates:
        candidates.append(("fallback", lcd_image))

    return candidates


def remove_edge_components(bin_img: cv2.Mat) -> cv2.Mat:
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


def preprocess_roi_for_ocr(
    roi_image: cv2.Mat,
    dark_threshold: int = DARK_PIXEL_THRESHOLD,
    scale: int = OCR_SCALE,
    blur: int = OCR_BLUR,
    close_enable: int = 1,
) -> cv2.Mat:
    """
    Applies image processing techniques to enhance the ROI for OCR.
    """
    # המרה לשחור לבן
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

    scale = _clamp(int(scale), 1, 6)
    if scale != 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # טשטוש קל להורדת רעשים
    blur = _clamp(int(blur), 0, 2)
    if blur == 1:
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    elif blur == 2:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    else:
        blurred = gray

    # הפעלת סף (Threshold)
    threshold_value = _clamp(int(dark_threshold), 0, 255)
    _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)

    edge = max(2, int(min(thresh.shape[:2]) * 0.025))
    thresh[:edge, :] = 0
    thresh[-edge:, :] = 0
    thresh[:, :edge] = 0
    thresh[:, -edge:] = 0
    thresh = remove_edge_components(thresh)

    # פעולה מורפולוגית: חיבור המקטעים של ה-7-segments
    if int(close_enable) == 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    else:
        processed = thresh

    # היפוך צבעים: Tesseract צריך טקסט שחור על רקע לבן
    final_processed = cv2.bitwise_not(processed)

    return final_processed


def normalize_numeric_text(text: str) -> str:
    cleaned = re.sub(r'[^0-9.]', '', text)
    if not cleaned:
        return ""

    if cleaned.count('.') > 1:
        head, tail = cleaned.split('.', 1)
        cleaned = head + '.' + tail.replace('.', '')

    if cleaned == ".":
        return ""

    return cleaned


SEGMENT_ZONES = {
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


def clean_7seg_mask(mask: cv2.Mat) -> cv2.Mat:
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


def active_column_runs(mask: cv2.Mat) -> list[tuple[int, int]]:
    img_h, img_w = mask.shape[:2]
    min_col_pixels = max(1, int(img_h * 0.01))
    active_cols = [cv2.countNonZero(mask[:, x]) >= min_col_pixels for x in range(img_w)]

    runs: list[tuple[int, int]] = []
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
    small_run = max(6, int(img_w * 0.04))
    for x1, x2 in runs[1:]:
        prev_x1, prev_x2 = merged[-1]
        gap = x1 - prev_x2
        prev_w = prev_x2 - prev_x1
        curr_w = x2 - x1
        should_merge = gap <= tiny_gap or (
            gap <= max_gap and (prev_w <= small_run or curr_w <= small_run)
        )
        if should_merge:
            merged[-1] = (prev_x1, x2)
        else:
            merged.append((x1, x2))

    return merged


def segment_is_active(digit_mask: cv2.Mat, label: str, zone: tuple[float, float, float, float]) -> bool:
    h, w = digit_mask.shape[:2]
    left, top, right, bottom = zone
    x1 = _clamp(int(w * left), 0, w - 1)
    y1 = _clamp(int(h * top), 0, h - 1)
    x2 = _clamp(int(w * right), x1 + 1, w)
    y2 = _clamp(int(h * bottom), y1 + 1, h)
    segment = digit_mask[y1:y2, x1:x2]
    threshold = SEGMENT_ACTIVE_THRESHOLDS.get(label, 0.25)
    return cv2.countNonZero(segment) / float(segment.size) >= threshold


def classify_7seg_digit(digit_mask: cv2.Mat) -> tuple[str, str] | None:
    h, w = digit_mask.shape[:2]
    if h < 10 or w < 3:
        return None

    aspect_ratio = w / float(h)

    if aspect_ratio <= 0.30:
        return "1", "narrow"

    active = frozenset(
        label
        for label, zone in SEGMENT_ZONES.items()
        if segment_is_active(digit_mask, label, zone)
    )
    if active == frozenset("bc") and aspect_ratio >= 0.50:
        return None
    if active == frozenset("bcdg") and aspect_ratio >= 0.32:
        return "4", "bcdg-loose"
    digit = SEGMENT_DIGITS.get(active)
    if digit is None:
        return None

    return digit, "".join(sorted(active))


def read_7seg_from_mask(mask: cv2.Mat) -> tuple[str, str] | None:
    cleaned = clean_7seg_mask(mask)
    if cv2.countNonZero(cleaned) == 0:
        return None

    img_h = cleaned.shape[0]
    img_w = cleaned.shape[1]
    candidates = []

    for x1, x2 in active_column_runs(cleaned):
        column_slice = cleaned[:, x1:x2]
        points = cv2.findNonZero(column_slice)
        if points is None:
            continue

        _, y, _, h = cv2.boundingRect(points)
        if h < img_h * 0.25:
            continue

        result = classify_7seg_digit(column_slice[y:y + h, :])
        if result is None:
            return None

        digit, segments = result
        digit_mask = column_slice[y:y + h, :]
        candidates.append({
            "digit": digit,
            "segments": segments,
            "x1": x1,
            "x2": x2,
            "y": y,
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
            abnormal_shape = item["h"] > max_h * 1.10 or item["area"] < max_area * 0.45
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

    max_area = max(item["area"] for item in candidates)
    max_h = max(item["h"] for item in candidates)
    max_w = max(item["w"] for item in candidates)
    digit_top = min(item["y"] for item in candidates)
    digit_bottom = max(item["y"] + item["h"] for item in candidates)

    dot_candidates: list[tuple[int, dict[str, int]]] = []
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

    dot_slots: dict[int, dict[str, int]] = {}
    if dot_candidates:
        best_slot, best_info = max(dot_candidates, key=lambda item: item[1]["area"])
        dot_slots[best_slot] = best_info

    text_parts: list[str] = []
    details: list[str] = []
    for index, item in enumerate(candidates):
        text_parts.append(item["digit"])
        details.append(f"{item['digit']}:{item['segments']}")
        if (index + 1) in dot_slots:
            text_parts.append(".")
            details.append(".:dot")

    return "".join(text_parts), f"[7seg:{','.join(details)}]"


def read_7seg_from_ocr_input(ocr_image: cv2.Mat) -> tuple[str, str] | None:
    _, mask = cv2.threshold(ocr_image, 128, 255, cv2.THRESH_BINARY_INV)
    return read_7seg_from_mask(mask)


def read_number_candidate_from_processed(ocr_image: cv2.Mat) -> tuple[str, float, str] | None:
    seven_seg = read_7seg_from_ocr_input(ocr_image)
    if seven_seg is not None:
        value, details = seven_seg
        return value, 95.0, details

    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.'
    data = pytesseract.image_to_data(ocr_image, config=custom_config, output_type=pytesseract.Output.DICT)
    tokens = [t for t in data.get("text", []) if t and t.strip()]
    raw_joined = "".join(tokens)
    cleaned_text = normalize_numeric_text(raw_joined)

    confs = []
    for conf in data.get("conf", []):
        try:
            conf_value = float(conf)
            if conf_value >= 0:
                confs.append(conf_value)
        except Exception:
            pass

    mean_conf = sum(confs) / len(confs) if confs else 0.0
    if cleaned_text:
        return cleaned_text, mean_conf, f"[tesseract:{raw_joined}]"

    return None


def vote_quality_score(text: str, raw: str, conf: float) -> float:
    normalized_conf = max(0.0, min(float(conf), 95.0)) / 95.0
    if raw.startswith("[7seg:"):
        digit_count = sum(ch.isdigit() for ch in text)
        score = 1.0 + normalized_conf + digit_count * 0.40
        if "." in text:
            score += 0.45
        if "loose" in raw:
            score -= 0.50
        return score

    score = max(0.15, normalized_conf) + sum(ch.isdigit() for ch in text) * 0.20
    if "." in text:
        score += 0.20
    return score


def select_best_vote(votes: dict[str, dict[str, object]]) -> tuple[str, dict[str, object]]:
    return max(
        votes.items(),
        key=lambda item: (
            float(item[1]["best_score"]),
            float(item[1]["score_sum"]),
            int(item[1]["count"]),
            float(item[1]["best_conf"]),
            len(item[0]),
        ),
    )


def is_stable_best_vote(text: str, info: dict[str, object]) -> bool:
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


def extract_number_from_roi(roi_image: cv2.Mat) -> float | None:
    """
    Extracts a number from the whole LCD ROI using a fixed internal multi-pass OCR strategy.
    """
    votes: dict[str, dict[str, object]] = {}
    tesseract_fallbacks: list[tuple[str, str, cv2.Mat]] = []

    def record_vote(text: str, conf: float, raw: str, crop_name: str, variant_name: str) -> None:
        bucket = votes.setdefault(
            text,
            {"count": 0, "score_sum": 0.0, "best_score": float("-inf"), "best_conf": 0.0, "raw": raw, "variants": []},
        )
        quality = vote_quality_score(text, raw, conf)
        bucket["count"] = int(bucket["count"]) + 1
        bucket["score_sum"] = float(bucket["score_sum"]) + quality
        bucket["best_score"] = max(float(bucket["best_score"]), quality)
        bucket["best_conf"] = max(float(bucket["best_conf"]), float(conf))
        bucket["raw"] = raw
        if isinstance(bucket["variants"], list):
            bucket["variants"].append(f"{crop_name}/{variant_name}")

    for crop_name, reading_roi in build_reading_roi_candidates(roi_image):
        for variant_name, dark_threshold, scale, blur, close_enable in ROBUST_PREPROCESS_VARIANTS:
            processed = preprocess_roi_for_ocr(
                reading_roi,
                dark_threshold=dark_threshold,
                scale=scale,
                blur=blur,
                close_enable=close_enable,
            )
            seven_seg = read_7seg_from_ocr_input(processed)
            if seven_seg is not None:
                text, raw = seven_seg
                record_vote(text, 95.0, raw, crop_name, variant_name)
            else:
                tesseract_fallbacks.append((crop_name, variant_name, processed))

    if not votes:
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.'
        for crop_name, variant_name, processed in tesseract_fallbacks:
            data = pytesseract.image_to_data(processed, config=custom_config, output_type=pytesseract.Output.DICT)
            tokens = [t for t in data.get("text", []) if t and t.strip()]
            raw_joined = "".join(tokens)
            cleaned_text = normalize_numeric_text(raw_joined)
            if not cleaned_text:
                continue

            confs = []
            for conf in data.get("conf", []):
                try:
                    conf_value = float(conf)
                    if conf_value >= 0:
                        confs.append(conf_value)
                except Exception:
                    pass
            mean_conf = sum(confs) / len(confs) if confs else 0.0
            record_vote(cleaned_text, mean_conf, f"[tesseract:{raw_joined}]", crop_name, variant_name)

    if not votes:
        return None

    best_text, best_info = select_best_vote(votes)

    try:
        vote_count = len(best_info["variants"]) if isinstance(best_info["variants"], list) else int(best_info["count"])
        print(f"OCR robust pick: {best_text} {best_info['raw']} via {vote_count} votes")
        return float(best_text)
    except ValueError:
        print(f"Warning: Could not convert OCR output '{best_text}' to a number.")
        return None

# --- Functions for Alerting ---

def send_email_alert(sender_email: str, sender_password: str, recipient_email: str,
                     smtp_server: str, smtp_port: int, subject: str, body: str):
    """Sends an email alert."""
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = recipient_email

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Secure the connection
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print(f"Email alert sent to {recipient_email}: '{subject}'")
    except Exception as e:
        print(f"Error sending email alert: {e}")
        print("Please check your email settings, app password, and internet connection.")


# --- Main Monitoring Loop ---

def run_monitoring():
    """Main function to run the Geiger counter monitoring."""
    config = load_configuration()
    if config is None:
        return

    # Extract configuration parameters
    roi_coords = tuple(config.get('roi_coordinates'))
    warning_threshold = config.get('warning_threshold')
    critical_threshold = config.get('critical_threshold')
    measurement_interval_seconds = config.get('measurement_interval_seconds')
    email_settings = config.get('email_settings')
    log_directory = config.get('log_directory', './logs/') # Default if not in config
    if not all([roi_coords, warning_threshold is not None, critical_threshold is not None, measurement_interval_seconds is not None]):
        print("Error: Missing essential configuration parameters. Please re-run setup.py.")
        return

    # Ensure log directory exists
    os.makedirs(log_directory, exist_ok=True)
    log_file_path = os.path.join(log_directory, 'geiger_monitor.log')

    print("\n--- Starting Geiger Counter Monitoring ---")
    print(f"ROI: {roi_coords}")
    print(f"Warning Threshold: {warning_threshold}")
    print(f"Critical Threshold: {critical_threshold}")
    print(f"Measurement Interval: {measurement_interval_seconds} seconds")
    print(
        "OCR Strategy: fixed internal multi-pass "
        f"{[(dark, scale, blur, close_enable) for _, dark, scale, blur, close_enable in ROBUST_PREPROCESS_VARIANTS]}"
    )
    print(f"Email Alerts {'Enabled' if email_settings else 'Disabled'}")
    print(f"Log File: {log_file_path}")
    print("------------------------------------------")

    while True:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] Taking measurement...")

        # 1. Image Acquisition
        full_image = get_image(config)
        if full_image is None:
            print("Failed to acquire image. Skipping this measurement.")
            with open(log_file_path, 'a') as f:
                f.write(f"[{timestamp}] ERROR: Failed to acquire image.\n")
            time.sleep(measurement_interval_seconds)
            continue

        # 2. Crop ROI
        x1, y1, x2, y2 = roi_coords
        roi_image = full_image[y1:y2, x1:x2]

        if roi_image.size == 0:
            print(f"Error: ROI is empty. Check ROI coordinates: {roi_coords} and image dimensions.")
            with open(log_file_path, 'a') as f:
                f.write(f"[{timestamp}] ERROR: Empty ROI. Check config.\n")
            time.sleep(measurement_interval_seconds)
            continue

        # 3. Extract Number using the fixed internal robust OCR strategy
        radiation_value = extract_number_from_roi(roi_image)

        log_entry = f"[{timestamp}] Reading: "
        if radiation_value is not None:
            print(f"Extracted Reading: {radiation_value}")
            log_entry += f"{radiation_value}"

            # 6. Threshold Monitoring and Alerting
            alert_status = "NORMAL"
            if radiation_value >= critical_threshold:
                alert_status = "CRITICAL"
                print(f"ALERT: CRITICAL RADIATION LEVEL DETECTED: {radiation_value}")
                if email_settings:
                    send_email_alert(
                        sender_email=email_settings['sender_email'],
                        sender_password=email_settings['sender_app_password'],
                        recipient_email=email_settings['recipient_email'],
                        smtp_server=email_settings['smtp_server'],
                        smtp_port=email_settings['smtp_port'],
                        subject=f"CRITICAL RADIATION ALERT: {radiation_value}",
                        body=f"Geiger counter reading is {radiation_value}, which is at or above the critical threshold of {critical_threshold}."
                    )
            elif radiation_value >= warning_threshold:
                alert_status = "WARNING"
                print(f"ALERT: WARNING RADIATION LEVEL DETECTED: {radiation_value}")
                if email_settings:
                    send_email_alert(
                        sender_email=email_settings['sender_email'],
                        sender_password=email_settings['sender_app_password'],
                        recipient_email=email_settings['recipient_email'],
                        smtp_server=email_settings['smtp_server'],
                        smtp_port=email_settings['smtp_port'],
                        subject=f"WARNING RADIATION ALERT: {radiation_value}",
                        body=f"Geiger counter reading is {radiation_value}, which is at or above the warning threshold of {warning_threshold}."
                    )
            log_entry += f" | Status: {alert_status}"
        else:
            print("Could not read value from display.")
            log_entry += "N/A | Status: UNREADABLE"

        # 7. Logging
        with open(log_file_path, 'a') as f:
            f.write(log_entry + "\n")

        # 8. Pause before next measurement
        print(f"Waiting for {measurement_interval_seconds} seconds...")
        time.sleep(measurement_interval_seconds)

if __name__ == '__main__':
    run_monitoring()
