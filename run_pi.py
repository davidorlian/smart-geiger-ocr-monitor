from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict

import cv2

import ocr_engine
import ocr_pi
import run as _shared_runtime


def _read_number_from_roi(roi_image: cv2.Mat) -> Dict[str, Any]:
    result = ocr_pi.read_number_from_lcd_roi(roi_image, ocr_engine.Params())
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


def extract_number_from_image_with_roi(
    image: cv2.Mat,
    roi_coords: tuple[int, int, int, int],
) -> float | None:
    roi_image, _roi_mode = _shared_runtime.crop_configured_roi(
        image,
        roi_coords,
        allow_cropped_test_image_fallback=False,
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


def run_monitoring() -> None:
    """Raspberry Pi runtime entry point using the lightweight OCR strategy."""
    config = _shared_runtime.load_configuration()
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

    email_settings = config.get("email_settings")
    log_directory = config.get("log_directory", "./logs/")
    camera_resolution = tuple(config.get("rpi_camera_resolution", (1920, 1080)))

    os.makedirs(log_directory, exist_ok=True)
    log_file_path = os.path.join(log_directory, "geiger_monitor.log")

    print("\n--- Starting Geiger Counter Monitoring on Raspberry Pi ---")
    print(f"ROI: {roi_coords}")
    print(f"Warning Threshold: {warning_threshold}")
    print(f"Critical Threshold: {critical_threshold}")
    print(f"Measurement Interval: {measurement_interval_seconds} seconds")
    print("OCR Strategy: Raspberry Pi lightweight OCR (7-segment fast path)")
    print(f"Camera Resolution: {camera_resolution}")
    if config.get("PC_TEST_MODE", False):
        print("Note: config PC_TEST_MODE is true, but run_pi.py always captures from the Raspberry Pi camera.")
    print(f"Email Alerts {'Enabled' if email_settings else 'Disabled'}")
    print(f"Log File: {log_file_path}")
    print("---------------------------------------------------------")

    while True:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] Taking measurement...")

        full_image = _shared_runtime._get_image_from_pi_camera(camera_resolution)
        if full_image is None:
            print("Failed to acquire image. Skipping this measurement.")
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] ERROR: Failed to acquire image.\n")
            time.sleep(measurement_interval_seconds)
            continue

        radiation_value = extract_number_from_image_with_roi(full_image, roi_coords)

        log_entry = f"[{timestamp}] Reading: "
        if radiation_value is not None:
            print(f"Extracted Reading: {radiation_value}")
            log_entry += f"{radiation_value}"

            alert_status = "NORMAL"
            if radiation_value >= critical_threshold:
                alert_status = "CRITICAL"
                print(f"ALERT: CRITICAL RADIATION LEVEL DETECTED: {radiation_value}")
                if email_settings:
                    _shared_runtime.send_email_alert(
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
                    _shared_runtime.send_email_alert(
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

        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

        print(f"Waiting for {measurement_interval_seconds} seconds...")
        time.sleep(measurement_interval_seconds)


if __name__ == "__main__":
    run_monitoring()
