from __future__ import annotations

import cv2
import json
import os
import smtplib
import time
from datetime import datetime
from email.mime.text import MIMEText


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(PROJECT_DIR, "config.json")

# If you want to simulate live capture on PC, specify a directory containing
# fixed-camera test images and the script will cycle through them.
# If set to None, PC test mode uses the setup image saved in config.json.
PC_TEST_IMAGE_DIR = None
PC_TEST_IMAGE_INDEX = 0


def _get_image_from_pi_camera(resolution: tuple) -> cv2.Mat | None:
    """
    Captures a live image from the Raspberry Pi camera.
    Returns an OpenCV image or None on failure.
    """
    try:
        from picamera2 import Picamera2

        print("Capturing image from Raspberry Pi Camera...")
        picam2 = Picamera2()
        camera_config = picam2.create_still_configuration(main={"size": resolution})
        picam2.configure(camera_config)
        picam2.start()
        time.sleep(1)
        image_np = picam2.capture_array()
        picam2.stop()
        picam2.close()
        print("Image captured from Pi camera.")
        return cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    except ImportError:
        print("Error: `picamera2` not found. Cannot capture from Pi camera.")
        return None
    except Exception as e:
        print(f"Error capturing image from Pi camera: {e}")
        return None


def _get_image_for_pc_test_mode(image_path_or_dir: str | None) -> cv2.Mat | None:
    """
    Loads an image for PC test mode. If image_path_or_dir is a directory,
    it cycles through images in that directory.
    """
    global PC_TEST_IMAGE_INDEX

    if image_path_or_dir is None:
        config_data = load_configuration()
        if config_data and "initial_image_for_roi" in config_data:
            img_path = config_data["initial_image_for_roi"]
            print(f"Loading single test image from config: {img_path}")
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error: Could not load image at {img_path}.")
            return img

        print("Error: No test image path found in config for PC Test Mode and no directory specified.")
        return None

    if os.path.isdir(image_path_or_dir):
        image_files = sorted(
            [
                f
                for f in os.listdir(image_path_or_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
            ]
        )
        if not image_files:
            print(f"Error: No image files found in test directory: {image_path_or_dir}")
            return None

        current_image_file = image_files[PC_TEST_IMAGE_INDEX % len(image_files)]
        img_path = os.path.join(image_path_or_dir, current_image_file)
        PC_TEST_IMAGE_INDEX += 1

        print(f"Loading test image from directory: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not load image at {img_path}.")
        return img

    if os.path.isfile(image_path_or_dir):
        print(f"Loading test image from path: {image_path_or_dir}")
        img = cv2.imread(image_path_or_dir)
        if img is None:
            print(f"Error: Could not load image at {image_path_or_dir}.")
        return img

    print(f"Error: Invalid PC test image path or directory: {image_path_or_dir}")
    return None


def get_image(config: dict) -> cv2.Mat | None:
    """
    Acquires an image based on the PC_TEST_MODE setting from the configuration.
    """
    if config.get("PC_TEST_MODE", False):
        return _get_image_for_pc_test_mode(PC_TEST_IMAGE_DIR)

    return _get_image_from_pi_camera(config.get("rpi_camera_resolution", (1920, 1080)))


def load_configuration() -> dict | None:
    """Loads configuration from config.json."""
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: Configuration file '{CONFIG_FILE}' not found.")
        print("Please run setup.py first to create the configuration.")
        return None

    try:
        with open(CONFIG_FILE, "r") as f:
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


def extract_number_from_roi(roi_image: cv2.Mat) -> float | None:
    """
    Extracts a number from a selected LCD ROI using the shared robust OCR engine.
    """
    result = _read_number_from_roi(roi_image)
    text = result["text"]
    raw = result["raw"]
    debug = result["debug"]
    value = result["value"]

    if text:
        vote_count = int(debug.get("vote_count", 0)) if isinstance(debug, dict) else 0
        print(f"OCR robust pick: {text} {raw} via {vote_count} votes")
        return value

    winner_raw = debug.get("winner_raw", raw) if isinstance(debug, dict) else raw
    rejected = debug.get("rejected", "no_read") if isinstance(debug, dict) else "no_read"
    print(f"OCR robust pick rejected: {winner_raw} ({rejected})")
    return None


def _read_number_from_roi(roi_image: cv2.Mat) -> dict[str, object]:
    try:
        import ocr_engine
    except Exception as exc:
        return {"value": None, "text": "", "raw": "", "debug": {"rejected": f"import_error:{exc}"}}

    text, conf, raw, debug = ocr_engine.robust_ocr_from_lcd_roi(roi_image, ocr_engine.Params())
    if not text:
        return {"value": None, "text": text, "raw": raw, "debug": debug, "conf": conf}

    try:
        return {"value": float(text), "text": text, "raw": raw, "debug": debug, "conf": conf}
    except ValueError:
        rejected_debug = debug if isinstance(debug, dict) else {}
        rejected_debug["rejected"] = "float_conversion"
        return {"value": None, "text": text, "raw": raw, "debug": rejected_debug, "conf": conf}


def extract_number_from_image_with_roi(
    image: cv2.Mat,
    roi_coords: tuple[int, int, int, int],
    pc_test_mode: bool = False,
) -> float | None:
    roi_image, roi_mode = crop_configured_roi(
        image,
        roi_coords,
        allow_cropped_test_image_fallback=pc_test_mode,
    )
    if roi_image is None:
        print(f"OCR error: ROI is invalid for image dimensions. roi={roi_coords} shape={image.shape[:2]}")
        return None

    result = _read_number_from_roi(roi_image)
    text = str(result.get("text", ""))
    raw = str(result.get("raw", ""))
    debug = result.get("debug", {})
    value = result.get("value")
    if text and value is not None:
        vote_count = int(debug.get("vote_count", 0)) if isinstance(debug, dict) else 0
        if roi_mode == "full_image_fallback":
            print("PC test mode: input image already looks cropped; using full image instead of configured ROI.")
        print(f"OCR robust pick: {text} {raw} via {vote_count} votes")
        return float(value)

    winner_raw = debug.get("winner_raw", raw) if isinstance(debug, dict) else raw
    rejected = debug.get("rejected", "no_read") if isinstance(debug, dict) else "no_read"
    print(f"OCR robust pick rejected: {winner_raw} ({rejected})")
    return None


def send_email_alert(
    sender_email: str,
    sender_password: str,
    recipient_email: str,
    smtp_server: str,
    smtp_port: int,
    subject: str,
    body: str,
) -> None:
    """Sends an email alert."""
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


def run_monitoring() -> None:
    """Main function to run the Geiger counter monitoring."""
    config = load_configuration()
    if config is None:
        return

    roi_coords = tuple(config.get("roi_coordinates"))
    warning_threshold = config.get("warning_threshold")
    critical_threshold = config.get("critical_threshold")
    measurement_interval_seconds = config.get("measurement_interval_seconds")
    email_settings = config.get("email_settings")
    log_directory = config.get("log_directory", "./logs/")
    if not all([roi_coords, warning_threshold is not None, critical_threshold is not None, measurement_interval_seconds is not None]):
        print("Error: Missing essential configuration parameters. Please re-run setup.py.")
        return

    os.makedirs(log_directory, exist_ok=True)
    log_file_path = os.path.join(log_directory, "geiger_monitor.log")

    print("\n--- Starting Geiger Counter Monitoring ---")
    print(f"ROI: {roi_coords}")
    print(f"Warning Threshold: {warning_threshold}")
    print(f"Critical Threshold: {critical_threshold}")
    print(f"Measurement Interval: {measurement_interval_seconds} seconds")
    print("OCR Strategy: shared robust OCR engine (fixed internal multi-pass)")
    print(f"Email Alerts {'Enabled' if email_settings else 'Disabled'}")
    print(f"Log File: {log_file_path}")
    print("------------------------------------------")

    while True:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] Taking measurement...")

        full_image = get_image(config)
        if full_image is None:
            print("Failed to acquire image. Skipping this measurement.")
            with open(log_file_path, "a") as f:
                f.write(f"[{timestamp}] ERROR: Failed to acquire image.\n")
            time.sleep(measurement_interval_seconds)
            continue

        radiation_value = extract_number_from_image_with_roi(
            full_image,
            roi_coords,
            pc_test_mode=bool(config.get("PC_TEST_MODE", False)),
        )

        log_entry = f"[{timestamp}] Reading: "
        if radiation_value is not None:
            print(f"Extracted Reading: {radiation_value}")
            log_entry += f"{radiation_value}"

            alert_status = "NORMAL"
            if radiation_value >= critical_threshold:
                alert_status = "CRITICAL"
                print(f"ALERT: CRITICAL RADIATION LEVEL DETECTED: {radiation_value}")
                if email_settings:
                    send_email_alert(
                        sender_email=email_settings["sender_email"],
                        sender_password=email_settings["sender_app_password"],
                        recipient_email=email_settings["recipient_email"],
                        smtp_server=email_settings["smtp_server"],
                        smtp_port=email_settings["smtp_port"],
                        subject=f"CRITICAL RADIATION ALERT: {radiation_value}",
                        body=(
                            f"Geiger counter reading is {radiation_value}, which is at or above "
                            f"the critical threshold of {critical_threshold}."
                        ),
                    )
            elif radiation_value >= warning_threshold:
                alert_status = "WARNING"
                print(f"ALERT: WARNING RADIATION LEVEL DETECTED: {radiation_value}")
                if email_settings:
                    send_email_alert(
                        sender_email=email_settings["sender_email"],
                        sender_password=email_settings["sender_app_password"],
                        recipient_email=email_settings["recipient_email"],
                        smtp_server=email_settings["smtp_server"],
                        smtp_port=email_settings["smtp_port"],
                        subject=f"WARNING RADIATION ALERT: {radiation_value}",
                        body=(
                            f"Geiger counter reading is {radiation_value}, which is at or above "
                            f"the warning threshold of {warning_threshold}."
                        ),
                    )
            log_entry += f" | Status: {alert_status}"
        else:
            print("Could not read value from display.")
            log_entry += "N/A | Status: UNREADABLE"

        with open(log_file_path, "a") as f:
            f.write(log_entry + "\n")

        print(f"Waiting for {measurement_interval_seconds} seconds...")
        time.sleep(measurement_interval_seconds)


if __name__ == "__main__":
    run_monitoring()
