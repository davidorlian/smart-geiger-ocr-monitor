from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import smtplib
import subprocess
import tempfile
import time
from datetime import datetime
from email.mime.text import MIMEText
from typing import Any, Dict, Optional

import cv2

import engine


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(PROJECT_DIR, "config.json")

CAMERA_BACKEND_PICAMERA2 = "picamera2"
CAMERA_BACKEND_LIBCAMERA_STILL = "libcamera-still"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
EXPECTED_NUMERIC_RE = re.compile(r"^\d+(?:\.\d+)?$")

_SELECTED_CAPTURE_BACKEND: str | None = None
_CAPTURE_BACKEND_ANNOUNCED = False
_PICAMERA2_FALLBACK_ANNOUNCED = False


def load_configuration() -> dict | None:
    """Load the local runtime configuration written by setup.py."""
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: Configuration file '{CONFIG_FILE}' not found.")
        print("Please run setup.py first to create the configuration.")
        return None

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{CONFIG_FILE}': {e}")
        return None
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _normalize_roi_bounds(
    image: cv2.Mat,
    roi_coords: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    img_h, img_w = image.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in roi_coords)
    x1 = _clamp(x1, 0, img_w)
    y1 = _clamp(y1, 0, img_h)
    x2 = _clamp(x2, 0, img_w)
    y2 = _clamp(y2, 0, img_h)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_configured_roi(
    image: cv2.Mat,
    roi_coords: tuple[int, int, int, int],
    allow_cropped_test_image_fallback: bool = False,
) -> tuple[cv2.Mat | None, str]:
    normalized = _normalize_roi_bounds(image, roi_coords)
    if normalized is None:
        return None, "invalid_roi"

    x1, y1, x2, y2 = normalized
    img_h, img_w = image.shape[:2]

    if allow_cropped_test_image_fallback:
        roi_w = max(0, x2 - x1)
        roi_h = max(0, y2 - y1)
        near_full_image = (
            x1 <= max(20, int(img_w * 0.08))
            and y1 <= max(20, int(img_h * 0.08))
            and roi_w >= int(img_w * 0.80)
            and roi_h >= int(img_h * 0.80)
        )
        looks_like_cropped_reading = img_h > 0 and (img_w / float(img_h)) >= 1.60
        if near_full_image and looks_like_cropped_reading:
            return image.copy(), "full_image_fallback"

    roi_image = image[y1:y2, x1:x2]
    if roi_image.size == 0:
        return None, "empty_roi"

    return roi_image.copy(), "configured_roi"


def send_email_alert(
    sender_email: str,
    sender_password: str,
    recipient_email: str,
    smtp_server: str,
    smtp_port: int,
    subject: str,
    body: str,
) -> None:
    """Send an email alert for warning or critical readings."""
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = recipient_email

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print(f"Email alert sent to {recipient_email}: '{subject}'")
    except Exception as e:
        print(f"Error sending email alert: {e}")
        print("Please check your email settings, app password, and internet connection.")


def _read_number_from_roi(roi_image: cv2.Mat) -> Dict[str, Any]:
    result = engine.read_number_from_roi(roi_image, mode="fast")
    text = str(result.get("text", ""))
    raw = str(result.get("raw", ""))
    debug = result.get("debug", {})

    if text:
        attempts = int(debug.get("attempt_count", 0)) if isinstance(debug, dict) else 0
        source = str(debug.get("source", "")) if isinstance(debug, dict) else ""
        print(f"OCR Pi fast pick: {text} {raw} source={source} attempts={attempts}")
        return result

    winner_raw = debug.get("winner_raw", raw) if isinstance(debug, dict) else raw
    rejected = debug.get("rejected", "no_read") if isinstance(debug, dict) else "no_read"
    print(f"OCR Pi fast pick rejected: {winner_raw} ({rejected})")
    return result


def _picamera2_available() -> bool:
    return importlib.util.find_spec("picamera2") is not None


def _select_capture_backend() -> str:
    return CAMERA_BACKEND_LIBCAMERA_STILL


def _resolve_capture_backend() -> str:
    global _SELECTED_CAPTURE_BACKEND
    if _SELECTED_CAPTURE_BACKEND is None:
        _SELECTED_CAPTURE_BACKEND = _select_capture_backend()
    return _SELECTED_CAPTURE_BACKEND


def _capture_backend_description(backend: str) -> str:
    if backend == CAMERA_BACKEND_PICAMERA2:
        return "picamera2"
    return "libcamera-still"


def _announce_capture_backend_once(backend: str) -> None:
    global _CAPTURE_BACKEND_ANNOUNCED
    if _CAPTURE_BACKEND_ANNOUNCED:
        return
    print(f"Camera Capture Backend: {_capture_backend_description(backend)}")
    _CAPTURE_BACKEND_ANNOUNCED = True


def _capture_image_with_picamera2(resolution: tuple[int, int]) -> cv2.Mat | None:
    try:
        from picamera2 import Picamera2
    except ImportError:
        return None

    picam2 = None
    try:
        picam2 = Picamera2()
        camera_config = picam2.create_still_configuration(main={"size": resolution})
        picam2.configure(camera_config)
        picam2.start()
        time.sleep(1)
        image_np = picam2.capture_array()
        return cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        print(f"picamera2 capture failed: {exc}")
        return None
    finally:
        if picam2 is not None:
            try:
                picam2.stop()
            except Exception:
                pass
            try:
                picam2.close()
            except Exception:
                pass


def _capture_image_with_libcamera_still(resolution: tuple[int, int]) -> cv2.Mat | None:
    width, height = (int(resolution[0]), int(resolution[1]))
    with tempfile.TemporaryDirectory(prefix="meter_camera_") as tmp_dir:
        image_path = os.path.join(tmp_dir, "capture.jpg")
        cmd = [
            "libcamera-still",
            "--nopreview",
            "--width",
            str(width),
            "--height",
            str(height),
            "--timeout",
            "2000",
            "--output",
            image_path,
        ]
        try:
            completed = subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            print("Error: `libcamera-still` not found. Install/enable Raspberry Pi camera tools.")
            return None
        except Exception as exc:
            print(f"Error running libcamera-still: {exc}")
            return None

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            print(f"Error: libcamera-still failed with exit code {completed.returncode}.")
            if stderr:
                print(stderr)
            return None

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: libcamera-still completed but OpenCV could not read {image_path}.")
            return None

        return image


def _get_image_from_pi_camera(resolution: tuple[int, int]) -> cv2.Mat | None:
    global _SELECTED_CAPTURE_BACKEND, _PICAMERA2_FALLBACK_ANNOUNCED

    backend = _resolve_capture_backend()
    _announce_capture_backend_once(backend)

    if backend == CAMERA_BACKEND_PICAMERA2:
        image = _capture_image_with_picamera2(resolution)
        if image is not None:
            return image

        _SELECTED_CAPTURE_BACKEND = CAMERA_BACKEND_LIBCAMERA_STILL
        if not _PICAMERA2_FALLBACK_ANNOUNCED:
            print("picamera2 capture is unavailable; switching to libcamera-still for this run.")
            _PICAMERA2_FALLBACK_ANNOUNCED = True

    return _capture_image_with_libcamera_still(resolution)


def extract_number_from_image_with_roi(
    image: cv2.Mat,
    roi_coords: tuple[int, int, int, int],
    pc_test_mode: bool = False,
) -> float | None:
    roi_image, _roi_mode = crop_configured_roi(
        image,
        roi_coords,
        allow_cropped_test_image_fallback=pc_test_mode,
    )
    if roi_image is None:
        print(f"OCR error: ROI is invalid for image dimensions. roi={roi_coords} shape={image.shape[:2]}")
        return None

    result = _read_number_from_roi(roi_image)
    value = result.get("value")
    return float(value) if value is not None else None


def _require_config_value(config: Dict[str, Any], key: str) -> Any:
    value = config.get(key)
    if value is None:
        raise ValueError(f"Missing required configuration value: {key}")
    return value


def _debug_summary(debug: Any) -> str:
    if not isinstance(debug, dict):
        return ""
    parts = []
    for key in ("source", "winner_label", "winner_stage", "attempt_count", "rejected"):
        value = debug.get(key)
        if value not in (None, ""):
            parts.append(f"{key}={value}")
    return " ".join(parts)


def _safe_timestamp_for_filename(timestamp: str) -> str:
    return timestamp.replace(":", "").replace("-", "").replace(" ", "_")


def _safe_path_token(value: str, fallback: str = "debug") -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    token = token.strip("._-")
    if not token:
        token = fallback
    return token[:120]


def _debug_json_safe(value: Any) -> Any:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return f"<array shape={tuple(value.shape)} dtype={value.dtype}>"
    if isinstance(value, dict):
        return {str(key): _debug_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_debug_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _attempt_count_from_debug(debug: Any) -> int:
    if not isinstance(debug, dict):
        return 0
    try:
        return int(debug.get("attempt_count", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _decimal_readings_expected(warning_threshold: float, critical_threshold: float) -> bool:
    return (
        not float(warning_threshold).is_integer()
        or not float(critical_threshold).is_integer()
        or abs(float(warning_threshold)) < 10.0
        or abs(float(critical_threshold)) < 10.0
    )


def _suspicious_ocr_reasons(
    text: str,
    value: Any,
    debug: Any,
    warning_threshold: float,
    critical_threshold: float,
) -> list[str]:
    reasons: list[str] = []
    debug_dict = debug if isinstance(debug, dict) else {}
    rejected = str(debug_dict.get("rejected", "") or "")
    attempts = _attempt_count_from_debug(debug_dict)

    if value is None or not text or rejected:
        reasons.append(rejected or "ocr_unreadable")
    if text and "." not in text and _decimal_readings_expected(warning_threshold, critical_threshold):
        reasons.append("missing_decimal_point")
    if attempts >= 20:
        reasons.append(f"attempts_ge_20_{attempts}")
    if value is not None:
        try:
            if float(value) >= float(critical_threshold) * 5.0:
                reasons.append("unusually_high_reading")
        except (TypeError, ValueError):
            pass

    deduped: list[str] = []
    for reason in reasons:
        if reason and reason not in deduped:
            deduped.append(reason)
    return deduped


def _save_debug_images(
    full_image: cv2.Mat,
    roi_image: cv2.Mat,
    timestamp: str,
    debug_dir: str = "debug_captures",
) -> None:
    os.makedirs(debug_dir, exist_ok=True)
    name_token = _safe_timestamp_for_filename(timestamp)
    full_path = os.path.join(debug_dir, f"full_{name_token}.jpg")
    roi_path = os.path.join(debug_dir, f"roi_{name_token}.jpg")

    full_ok = cv2.imwrite(full_path, full_image)
    roi_ok = cv2.imwrite(roi_path, roi_image)
    if full_ok:
        print(f"Saved full capture debug image: {full_path}")
    else:
        print(f"Failed to save full capture debug image: {full_path}")
    if roi_ok:
        print(f"Saved ROI debug image: {roi_path}")
    else:
        print(f"Failed to save ROI debug image: {roi_path}")


def _save_suspicious_debug_capture(
    full_image: cv2.Mat,
    roi_image: cv2.Mat | None,
    timestamp: str,
    reasons: list[str],
    result: Dict[str, Any],
    roi_coords: tuple[int, int, int, int],
    debug_dir: str = "debug_captures",
) -> str | None:
    reason_text = "+".join(reasons) if reasons else "suspicious"
    folder_name = f"{_safe_timestamp_for_filename(timestamp)}_{_safe_path_token(reason_text)}"
    folder_path = os.path.join(debug_dir, folder_name)

    try:
        os.makedirs(folder_path, exist_ok=True)
    except Exception as exc:
        print(f"Failed to create suspicious OCR debug folder: {exc}")
        return None

    full_path = os.path.join(folder_path, "full.jpg")
    roi_path = os.path.join(folder_path, "roi_crop.jpg")
    debug_path = os.path.join(folder_path, "debug.txt")

    full_ok = cv2.imwrite(full_path, full_image)
    if not full_ok:
        print(f"Failed to save suspicious full capture: {full_path}")

    roi_to_write = roi_image
    if roi_to_write is None or getattr(roi_to_write, "size", 0) == 0:
        roi_to_write = full_image[0:1, 0:1].copy()
    roi_ok = cv2.imwrite(roi_path, roi_to_write)
    if not roi_ok:
        print(f"Failed to save suspicious ROI crop: {roi_path}")

    debug = result.get("debug", {})
    attempts = _attempt_count_from_debug(debug)
    raw = str(result.get("raw", ""))
    text = str(result.get("text", ""))
    value = result.get("value")
    conf = float(result.get("conf", 0.0) or 0.0)
    debug_summary = _debug_summary(debug)
    debug_payload = json.dumps(_debug_json_safe(debug), indent=2, sort_keys=True)

    try:
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(f"timestamp: {timestamp}\n")
            f.write(f"reason: {reason_text}\n")
            f.write(f"roi_coordinates: {list(roi_coords)}\n")
            f.write(f"attempts_count: {attempts}\n")
            f.write(f"ocr_text: {text}\n")
            f.write(f"parsed_value: {value if value is not None else 'N/A'}\n")
            f.write(f"confidence: {conf:.1f}\n")
            f.write(f"raw_ocr: {raw}\n")
            f.write(f"debug_summary: {debug_summary}\n")
            f.write("debug_details:\n")
            f.write(debug_payload)
            f.write("\n")
    except Exception as exc:
        print(f"Failed to write suspicious OCR debug text: {exc}")
        return folder_path

    print(f"Saved suspicious OCR debug evidence: {folder_path}")
    return folder_path


def _read_number_from_image_with_roi_result(
    image: cv2.Mat,
    roi_coords: tuple[int, int, int, int],
    save_debug_images: bool = False,
    timestamp: str = "",
) -> Dict[str, Any]:
    roi_image, _roi_mode = crop_configured_roi(
        image,
        roi_coords,
        allow_cropped_test_image_fallback=False,
    )
    if roi_image is None:
        print(f"OCR error: ROI is invalid for image dimensions. roi={roi_coords} shape={image.shape[:2]}")
        return {"value": None, "text": "", "conf": 0.0, "raw": "", "debug": {"rejected": "invalid_roi"}}
    if save_debug_images:
        _save_debug_images(image, roi_image, timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return _read_number_from_roi(roi_image)


def _collect_saved_image_paths(image_path: str | None, image_dir: str | None) -> list[str]:
    if image_path:
        if not os.path.isfile(image_path):
            print(f"Error: Image file not found: {image_path}")
            return []
        return [image_path]

    if not image_dir:
        return []
    if not os.path.isdir(image_dir):
        print(f"Error: Image directory not found: {image_dir}")
        return []

    paths = [
        os.path.join(image_dir, name)
        for name in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, name))
        and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=lambda path: os.path.basename(path).lower())


def _configured_roi_from_config(config: Dict[str, Any]) -> tuple[int, int, int, int] | None:
    try:
        roi_coords = tuple(int(v) for v in _require_config_value(config, "roi_coordinates"))
        if len(roi_coords) != 4:
            raise ValueError("roi_coordinates must contain four values: [x1, y1, x2, y2]")
        return roi_coords
    except Exception as exc:
        print(f"Error: Invalid configuration. {exc}")
        print("Please re-run setup.py or use --image-is-roi for already-cropped images.")
        return None


def _format_result_value(value: Any) -> str:
    return "N/A" if value is None else str(value)


def _expected_text_from_filename(image_path: str) -> str | None:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    prefix = "ram_gene_"
    if stem.startswith(prefix):
        token = stem[len(prefix):]
    else:
        match = re.search(r"(\d+(?:p\d+)?)$", stem)
        if not match:
            return None
        token = match.group(1)

    expected = token.replace("p", ".", 1)
    return expected if EXPECTED_NUMERIC_RE.fullmatch(expected) else None


def _saved_image_status(text: str, expected: str | None) -> str:
    if expected is None:
        return "UNK"
    return "OK " if text == expected else "BAD"


def _run_saved_image_case(
    image_path: str,
    mode: str,
    image_is_roi: bool,
    roi_coords: tuple[int, int, int, int] | None,
) -> Dict[str, Any]:
    image = cv2.imread(image_path)
    name = os.path.basename(image_path)
    if image is None:
        print(f"[ERR] {name}: failed to load image")
        return {"readable": False, "comparable": False, "exact": False}

    if image_is_roi:
        roi_image = image.copy()
        roi_mode = "image_is_roi"
    else:
        if roi_coords is None:
            print(f"[ERR] {name}: full-frame image requires roi_coordinates from config.json")
            return {"readable": False, "comparable": False, "exact": False}
        roi_image, roi_mode = crop_configured_roi(
            image,
            roi_coords,
            allow_cropped_test_image_fallback=False,
        )
        if roi_image is None:
            print(f"[ERR] {name}: invalid ROI {roi_coords} for image shape={image.shape[:2]}")
            return {"readable": False, "comparable": False, "exact": False}

    start = time.perf_counter()
    result = engine.read_number_from_roi(roi_image, mode=mode)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    text = str(result.get("text", ""))
    value = result.get("value")
    conf = float(result.get("conf", result.get("confidence", 0.0)) or 0.0)
    source = str(result.get("source", "") or "")
    raw = str(result.get("raw", ""))
    debug_summary = _debug_summary(result.get("debug", {}))
    expected = _expected_text_from_filename(image_path)
    status = _saved_image_status(text, expected)
    expected_text = f" expected='{expected}'" if expected is not None else ""

    print(
        f"[{status}] {name}{expected_text} got='{text}' value={_format_result_value(value)} "
        f"conf={conf:.1f} source={source or 'unknown'} mode={mode} roi={roi_mode} "
        f"elapsed={elapsed_ms:.1f}ms debug='{debug_summary}'"
    )
    return {
        "readable": bool(text),
        "comparable": expected is not None,
        "exact": expected is not None and text == expected,
        "raw": raw,
    }


def run_saved_image_mode(
    image_path: str | None,
    image_dir: str | None,
    image_is_roi: bool,
    mode: str,
) -> None:
    image_paths = _collect_saved_image_paths(image_path, image_dir)
    if not image_paths:
        print("No saved images to process.")
        return

    roi_coords = None
    if not image_is_roi:
        config = load_configuration()
        if config is None:
            return
        roi_coords = _configured_roi_from_config(config)
        if roi_coords is None:
            return

    print("\n--- Running Saved-Image OCR Through run.py ---")
    print(f"Images: {len(image_paths)}")
    print(f"OCR Mode: {mode}")
    print(f"Input: {'already-cropped ROI' if image_is_roi else f'full frame using ROI {roi_coords}'}")
    print("OCR Interface: engine.read_number_from_roi")
    print("------------------------------------------------")

    readable = 0
    comparable = 0
    exact = 0
    for path in image_paths:
        case_result = _run_saved_image_case(path, mode, image_is_roi, roi_coords)
        readable += int(bool(case_result.get("readable", False)))
        comparable += int(bool(case_result.get("comparable", False)))
        exact += int(bool(case_result.get("exact", False)))

    if comparable:
        print(
            f"Saved-image run complete: exact={exact}/{comparable} "
            f"readable={readable}/{len(image_paths)} images={len(image_paths)}"
        )
    else:
        print(f"Saved-image run complete: readable={readable}/{len(image_paths)} images={len(image_paths)}")


def _run_single_measurement(
    roi_coords: tuple[int, int, int, int],
    warning_threshold: float,
    critical_threshold: float,
    camera_resolution: tuple[int, int],
    log_file_path: str,
    email_settings: Optional[Dict[str, Any]],
    no_alerts: bool = False,
    print_ocr_summary: bool = False,
    save_debug_images: bool = False,
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] Taking measurement...")

    full_image = _get_image_from_pi_camera(camera_resolution)
    if full_image is None:
        print("Failed to acquire image. Skipping this measurement.")
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] ERROR: Failed to acquire image.\n")
        return

    start = time.perf_counter()
    roi_image, _roi_mode = crop_configured_roi(
        full_image,
        roi_coords,
        allow_cropped_test_image_fallback=False,
    )
    if roi_image is None:
        print(f"OCR error: ROI is invalid for image dimensions. roi={roi_coords} shape={full_image.shape[:2]}")
        result = {"value": None, "text": "", "conf": 0.0, "raw": "", "debug": {"rejected": "invalid_roi"}}
    else:
        if save_debug_images:
            _save_debug_images(full_image, roi_image, timestamp)
        result = _read_number_from_roi(roi_image)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    value = result.get("value")
    text = str(result.get("text", ""))
    conf = float(result.get("conf", 0.0) or 0.0)
    raw = str(result.get("raw", ""))
    debug = result.get("debug", {})
    debug_text = _debug_summary(debug)

    if print_ocr_summary:
        print(
            f"OCR result: text='{text}' value={value if value is not None else 'N/A'} "
            f"conf={conf:.1f} elapsed={elapsed_ms:.1f}ms raw='{raw}'"
        )
        if debug_text:
            print(f"OCR debug: {debug_text}")

    log_entry = f"[{timestamp}] Reading: "
    if value is not None:
        measurement_value = float(value)
        print(f"Extracted Reading: {measurement_value}")
        log_entry += f"{measurement_value}"

        alert_status = "NORMAL"
        if measurement_value >= critical_threshold:
            alert_status = "CRITICAL"
            if not no_alerts:
                print(f"ALERT: CRITICAL MULTIMETER READING: {measurement_value}")
                if email_settings:
                    send_email_alert(
                        sender_email=email_settings["sender_email"],
                        sender_password=email_settings["sender_app_password"],
                        recipient_email=email_settings["recipient_email"],
                        smtp_server=email_settings["smtp_server"],
                        smtp_port=email_settings["smtp_port"],
                        subject=f"CRITICAL MULTIMETER ALERT: {measurement_value}",
                        body=(
                            f"Multimeter reading is {measurement_value}, which is at or above "
                            f"the critical threshold of {critical_threshold}."
                        ),
                    )
        elif measurement_value >= warning_threshold:
            alert_status = "WARNING"
            if not no_alerts:
                print(f"ALERT: WARNING MULTIMETER READING: {measurement_value}")
                if email_settings:
                    send_email_alert(
                        sender_email=email_settings["sender_email"],
                        sender_password=email_settings["sender_app_password"],
                        recipient_email=email_settings["recipient_email"],
                        smtp_server=email_settings["smtp_server"],
                        smtp_port=email_settings["smtp_port"],
                        subject=f"WARNING MULTIMETER ALERT: {measurement_value}",
                        body=(
                            f"Multimeter reading is {measurement_value}, which is at or above "
                            f"the warning threshold of {warning_threshold}."
                        ),
                    )
        if no_alerts and alert_status != "NORMAL":
            print(f"Status: {alert_status} (alerts disabled)")
        log_entry += f" | Status: {alert_status}"
    else:
        print("Could not read value from display.")
        log_entry += "N/A | Status: UNREADABLE"

    suspicious_reasons = _suspicious_ocr_reasons(
        text,
        value,
        debug,
        warning_threshold,
        critical_threshold,
    )
    if suspicious_reasons:
        debug_folder = _save_suspicious_debug_capture(
            full_image,
            roi_image,
            timestamp,
            suspicious_reasons,
            result,
            roi_coords,
        )
        if debug_folder:
            log_entry += f" | Debug: {debug_folder}"

    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")


def run_monitoring(once: bool = False, no_alerts: bool = False, save_debug_images: bool = False) -> None:
    """Raspberry Pi runtime entry point using the lightweight OCR strategy."""
    config = load_configuration()
    if config is None:
        return

    try:
        roi_coords = tuple(int(v) for v in _require_config_value(config, "roi_coordinates"))
        if len(roi_coords) != 4:
            raise ValueError("roi_coordinates must contain four values: [x1, y1, x2, y2]")
        warning_threshold = float(_require_config_value(config, "warning_threshold"))
        critical_threshold = float(_require_config_value(config, "critical_threshold"))
        measurement_interval_seconds = int(_require_config_value(config, "measurement_interval_seconds"))
    except Exception as exc:
        print(f"Error: Invalid configuration. {exc}")
        print("Please re-run setup.py.")
        return

    email_settings = None if no_alerts else config.get("email_settings")
    log_directory = str(config.get("log_directory", "./logs/"))
    if not os.path.isabs(log_directory):
        log_directory = os.path.join(PROJECT_DIR, log_directory)
    camera_resolution = tuple(config.get("rpi_camera_resolution", (1920, 1080)))

    os.makedirs(log_directory, exist_ok=True)
    log_file_path = os.path.join(log_directory, "meter_monitor.log")

    print("\n--- Starting Multimeter Monitoring on Raspberry Pi ---")
    print(f"ROI: {roi_coords}")
    print(f"Warning Threshold: {warning_threshold}")
    print(f"Critical Threshold: {critical_threshold}")
    print(f"Measurement Interval: {measurement_interval_seconds} seconds")
    print("OCR Strategy: Raspberry Pi lightweight OCR (7-segment fast path)")
    print(f"Camera Resolution: {camera_resolution}")
    _announce_capture_backend_once(_resolve_capture_backend())
    if config.get("PC_TEST_MODE", False):
        print("Note: config PC_TEST_MODE is true, but run.py always captures from the Raspberry Pi camera.")
    print(f"Email Alerts {'Disabled by --no-alerts' if no_alerts else 'Enabled' if email_settings else 'Disabled'}")
    print(f"Log File: {log_file_path}")
    if once:
        print("Mode: one-shot measurement")
    if save_debug_images:
        print("Debug image saving: enabled")
    print("---------------------------------------------------------")

    while True:
        _run_single_measurement(
            roi_coords,
            warning_threshold,
            critical_threshold,
            camera_resolution,
            log_file_path,
            email_settings,
            no_alerts=no_alerts,
            print_ocr_summary=once,
            save_debug_images=save_debug_images,
        )
        if once:
            print("One-shot measurement complete.")
            return

        print(f"Waiting for {measurement_interval_seconds} seconds...")
        time.sleep(measurement_interval_seconds)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Meter OCR runtime and saved-image system test entry point. "
            "Primary saved-image example: python -u run.py --image-dir "
            "test_sets/green_multimeter_v3_cleaned/cropped --image-is-roi --mode fast --no-alerts."
        )
    )
    image_input = parser.add_mutually_exclusive_group()
    image_input.add_argument(
        "--image",
        default=None,
        help=(
            "Run OCR on one saved image instead of live capture, e.g. "
            "test_sets/green_multimeter_v3_cleaned/cropped/meter_hold_1p2309.jpg."
        ),
    )
    image_input.add_argument(
        "--image-dir",
        default=None,
        help="Run OCR on saved images in a directory, e.g. test_sets/green_multimeter_v3_cleaned/cropped.",
    )
    parser.add_argument(
        "--image-is-roi",
        action="store_true",
        help="Treat saved image inputs as already-cropped display ROIs.",
    )
    parser.add_argument(
        "--mode",
        choices=("fast", "full"),
        default="fast",
        help="OCR engine mode for saved-image inputs.",
    )
    parser.add_argument("--once", action="store_true", help="Capture, OCR, log one measurement, then exit.")
    parser.add_argument("--no-alerts", action="store_true", help="Disable alert printing and email sending.")
    parser.add_argument(
        "--save-debug-images",
        action="store_true",
        help="Save the full capture and cropped ROI images for diagnostics.",
    )
    args = parser.parse_args(argv)
    if args.image or args.image_dir:
        run_saved_image_mode(
            image_path=args.image,
            image_dir=args.image_dir,
            image_is_roi=bool(args.image_is_roi),
            mode=str(args.mode),
        )
        return

    run_monitoring(
        once=bool(args.once),
        no_alerts=bool(args.no_alerts),
        save_debug_images=bool(args.save_debug_images),
    )


if __name__ == "__main__":
    main()
