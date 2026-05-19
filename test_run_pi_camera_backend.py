from __future__ import annotations

import io
import shutil
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

import run_pi


ROOT = Path(__file__).resolve().parent


class RunPiCameraBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_picamera2_available = run_pi._picamera2_available
        self.original_picamera2_capture = run_pi._capture_image_with_picamera2
        self.original_libcamera_capture = run_pi._capture_image_with_libcamera_still
        self.original_get_image = run_pi._get_image_from_pi_camera
        self.original_read_number = run_pi._read_number_from_roi
        self.original_load_config = run_pi._shared_runtime.load_configuration
        self.original_selected_backend = run_pi._SELECTED_CAPTURE_BACKEND
        self.original_backend_announced = run_pi._CAPTURE_BACKEND_ANNOUNCED
        self.original_fallback_announced = run_pi._PICAMERA2_FALLBACK_ANNOUNCED
        self.case_dir = ROOT / "logs" / "run_pi_camera_backend_tests" / uuid.uuid4().hex
        run_pi._SELECTED_CAPTURE_BACKEND = None
        run_pi._CAPTURE_BACKEND_ANNOUNCED = False
        run_pi._PICAMERA2_FALLBACK_ANNOUNCED = False

    def tearDown(self) -> None:
        run_pi._picamera2_available = self.original_picamera2_available
        run_pi._capture_image_with_picamera2 = self.original_picamera2_capture
        run_pi._capture_image_with_libcamera_still = self.original_libcamera_capture
        run_pi._get_image_from_pi_camera = self.original_get_image
        run_pi._read_number_from_roi = self.original_read_number
        run_pi._shared_runtime.load_configuration = self.original_load_config
        run_pi._SELECTED_CAPTURE_BACKEND = self.original_selected_backend
        run_pi._CAPTURE_BACKEND_ANNOUNCED = self.original_backend_announced
        run_pi._PICAMERA2_FALLBACK_ANNOUNCED = self.original_fallback_announced
        shutil.rmtree(self.case_dir, ignore_errors=True)

    def test_selects_libcamera_when_picamera2_is_missing(self) -> None:
        run_pi._picamera2_available = lambda: False

        self.assertEqual(run_pi._resolve_capture_backend(), run_pi.CAMERA_BACKEND_LIBCAMERA_STILL)

    def test_selects_picamera2_when_available(self) -> None:
        run_pi._picamera2_available = lambda: True

        self.assertEqual(run_pi._resolve_capture_backend(), run_pi.CAMERA_BACKEND_PICAMERA2)

    def test_libcamera_fallback_does_not_print_picamera2_error(self) -> None:
        expected = np.zeros((4, 5, 3), dtype=np.uint8)
        run_pi._picamera2_available = lambda: False
        run_pi._capture_image_with_picamera2 = lambda _resolution: self.fail("picamera2 should not be called")
        run_pi._capture_image_with_libcamera_still = lambda _resolution: expected

        output = io.StringIO()
        with redirect_stdout(output):
            first = run_pi._get_image_from_pi_camera((5, 4))
            second = run_pi._get_image_from_pi_camera((5, 4))

        self.assertIs(first, expected)
        self.assertIs(second, expected)
        text = output.getvalue()
        self.assertEqual(text.count("Camera Capture Backend:"), 1)
        self.assertIn("libcamera-still", text)
        self.assertNotIn("picamera2` not found", text)
        self.assertNotIn("Cannot capture from Pi camera", text)

    def test_picamera2_backend_is_used_when_available(self) -> None:
        expected = np.zeros((4, 5, 3), dtype=np.uint8)
        run_pi._picamera2_available = lambda: True
        run_pi._capture_image_with_picamera2 = lambda _resolution: expected
        run_pi._capture_image_with_libcamera_still = lambda _resolution: self.fail("libcamera should not be called")

        output = io.StringIO()
        with redirect_stdout(output):
            image = run_pi._get_image_from_pi_camera((5, 4))

        self.assertIs(image, expected)
        self.assertIn("Camera Capture Backend: picamera2", output.getvalue())

    def test_picamera2_runtime_failure_switches_to_libcamera_once(self) -> None:
        expected = np.zeros((4, 5, 3), dtype=np.uint8)
        calls = {"picamera2": 0, "libcamera": 0}

        def picamera2_failure(_resolution):
            calls["picamera2"] += 1
            return None

        def libcamera_success(_resolution):
            calls["libcamera"] += 1
            return expected

        run_pi._picamera2_available = lambda: True
        run_pi._capture_image_with_picamera2 = picamera2_failure
        run_pi._capture_image_with_libcamera_still = libcamera_success

        output = io.StringIO()
        with redirect_stdout(output):
            first = run_pi._get_image_from_pi_camera((5, 4))
            second = run_pi._get_image_from_pi_camera((5, 4))

        self.assertIs(first, expected)
        self.assertIs(second, expected)
        self.assertEqual(calls["picamera2"], 1)
        self.assertEqual(calls["libcamera"], 2)
        text = output.getvalue()
        self.assertEqual(text.count("Camera Capture Backend:"), 1)
        self.assertEqual(text.count("switching to libcamera-still"), 1)

    def test_run_monitoring_prints_selected_backend_once_at_startup(self) -> None:
        run_pi._picamera2_available = lambda: False
        run_pi._shared_runtime.load_configuration = lambda: {
            "roi_coordinates": [0, 0, 4, 4],
            "warning_threshold": 0.6,
            "critical_threshold": 1.2,
            "measurement_interval_seconds": 5,
            "log_directory": str(self.case_dir),
            "rpi_camera_resolution": (4, 4),
            "email_settings": None,
        }
        run_pi._get_image_from_pi_camera = lambda _resolution: np.zeros((4, 4, 3), dtype=np.uint8)
        run_pi._read_number_from_roi = lambda _roi_image: {
            "value": 0.1,
            "text": "0.100",
            "conf": 95.0,
            "raw": "[test]",
            "debug": {"attempt_count": 1, "source": "test"},
        }

        output = io.StringIO()
        with redirect_stdout(output):
            run_pi.run_monitoring(once=True, no_alerts=True, save_debug_images=False)

        text = output.getvalue()
        self.assertEqual(text.count("Camera Capture Backend:"), 1)
        self.assertIn("Camera Capture Backend: libcamera-still", text)
        self.assertLess(text.index("Camera Capture Backend:"), text.index("Taking measurement"))


if __name__ == "__main__":
    unittest.main()
