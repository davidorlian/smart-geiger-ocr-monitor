from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import time
from datetime import datetime
from typing import Any, Dict, Optional

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


def _capture_image_with_libcamera_still(resolution: tuple[int, int]) -> cv2.Mat | None:
    width, height = (int(resolution[0]), int(resolution[1]))
    with tempfile.TemporaryDirectory(prefix="geiger_camera_") as tmp_dir:
        image_path = os.path.join(tmp_dir, "capture.jpg")
        cmd = [
            "libcamera-still",
            "--nopreview",
            "--timeout",
            "1000",
            "--width",
            str(width),
            "--height",
            str(height),
            "--output",
            image_path,
        ]
        print("Capturing image with libcamera-still...")
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

        print("Image captured with libcamera-still.")
        return image


def _get_image_from_pi_camera(resolution: tuple[int, int]) -> cv2.Mat | None:
    image = _shared_runtime._get_image_from_pi_camera(resolution)
    if image is not None:
        return image

    print("Falling back to libcamera-still capture.")
    return _capture_image_with_libcamera_still(resolution)


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


def _debug_summary(debug: Any) -> str:
    if not isinstance(debug, dict):
        return ""
    parts = []
    for key in ("source", "winner_label", "winner_stage", "attempt_count", "rejected"):
        value = debug.get(key)
        if value not in (None, ""):
            parts.append(f"{key}={value}")
    return " ".join(parts)


def _read_number_from_image_with_roi_result(
    image: cv2.Mat,
    roi_coords: tuple[int, int, int, int],
) -> Dict[str, Any]:
    roi_image, _roi_mode = _shared_runtime.crop_configured_roi(
        image,
        roi_coords,
        allow_cropped_test_image_fallback=False,
    )
    if roi_image is None:
        print(f"OCR error: ROI is invalid for image dimensions. roi={roi_coords} shape={image.shape[:2]}")
        return {"value": None, "text": "", "conf": 0.0, "raw": "", "debug": {"rejected": "invalid_roi"}}
    return _read_number_from_roi(roi_image)


def _run_single_measurement(
    roi_coords: tuple[int, int, int, int],
    warning_threshold: float,
    critical_threshold: float,
    camera_resolution: tuple[int, int],
    log_file_path: str,
    email_settings: Optional[Dict[str, Any]],
    no_alerts: bool = False,
    print_ocr_summary: bool = False,
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
    result = _read_number_from_image_with_roi_result(full_image, roi_coords)
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
        radiation_value = float(value)
        print(f"Extracted Reading: {radiation_value}")
        log_entry += f"{radiation_value}"

        alert_status = "NORMAL"
        if radiation_value >= critical_threshold:
            alert_status = "CRITICAL"
            if not no_alerts:
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
            if not no_alerts:
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
        if no_alerts and alert_status != "NORMAL":
            print(f"Status: {alert_status} (alerts disabled)")
        log_entry += f" | Status: {alert_status}"
    else:
        print("Could not read value from display.")
        log_entry += "N/A | Status: UNREADABLE"

    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")


def run_monitoring(once: bool = False, no_alerts: bool = False) -> None:
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

    email_settings = None if no_alerts else config.get("email_settings")
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
    print(f"Email Alerts {'Disabled by --no-alerts' if no_alerts else 'Enabled' if email_settings else 'Disabled'}")
    print(f"Log File: {log_file_path}")
    if once:
        print("Mode: one-shot measurement")
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
        )
        if once:
            print("One-shot measurement complete.")
            return

        print(f"Waiting for {measurement_interval_seconds} seconds...")
        time.sleep(measurement_interval_seconds)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Raspberry Pi Geiger monitor runtime.")
    parser.add_argument("--once", action="store_true", help="Capture, OCR, log one measurement, then exit.")
    parser.add_argument("--no-alerts", action="store_true", help="Disable alert printing and email sending.")
    args = parser.parse_args(argv)
    run_monitoring(once=bool(args.once), no_alerts=bool(args.no_alerts))


if __name__ == "__main__":
    main()
