from __future__ import annotations

import io
import shutil
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

import run


ROOT = Path(__file__).resolve().parents[1]


class RunCameraBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_picamera2_available = run._picamera2_available
        self.original_picamera2_capture = run._capture_image_with_picamera2
        self.original_libcamera_capture = run._capture_image_with_libcamera_still
        self.original_get_image = run._get_image_from_pi_camera
        self.original_read_number = run._read_number_from_roi
        self.original_load_config = run.load_configuration
        self.original_selected_backend = run._SELECTED_CAPTURE_BACKEND
        self.original_backend_announced = run._CAPTURE_BACKEND_ANNOUNCED
        self.original_fallback_announced = run._PICAMERA2_FALLBACK_ANNOUNCED
        self.case_dir = ROOT / "logs" / "run_camera_backend_tests" / uuid.uuid4().hex
        run._SELECTED_CAPTURE_BACKEND = None
        run._CAPTURE_BACKEND_ANNOUNCED = False
        run._PICAMERA2_FALLBACK_ANNOUNCED = False

    def tearDown(self) -> None:
        run._picamera2_available = self.original_picamera2_available
        run._capture_image_with_picamera2 = self.original_picamera2_capture
        run._capture_image_with_libcamera_still = self.original_libcamera_capture
        run._get_image_from_pi_camera = self.original_get_image
        run._read_number_from_roi = self.original_read_number
        run.load_configuration = self.original_load_config
        run._SELECTED_CAPTURE_BACKEND = self.original_selected_backend
        run._CAPTURE_BACKEND_ANNOUNCED = self.original_backend_announced
        run._PICAMERA2_FALLBACK_ANNOUNCED = self.original_fallback_announced
        shutil.rmtree(self.case_dir, ignore_errors=True)

    def test_selects_libcamera_when_picamera2_is_missing(self) -> None:
        run._picamera2_available = lambda: False

        self.assertEqual(run._resolve_capture_backend(), run.CAMERA_BACKEND_LIBCAMERA_STILL)

    def test_defaults_to_libcamera_even_when_picamera2_is_available(self) -> None:
        run._picamera2_available = lambda: True

        self.assertEqual(run._resolve_capture_backend(), run.CAMERA_BACKEND_LIBCAMERA_STILL)

    def test_libcamera_fallback_does_not_print_picamera2_error(self) -> None:
        expected = np.zeros((4, 5, 3), dtype=np.uint8)
        run._picamera2_available = lambda: False
        run._capture_image_with_picamera2 = lambda _resolution: self.fail("picamera2 should not be called")
        run._capture_image_with_libcamera_still = lambda _resolution: expected

        output = io.StringIO()
        with redirect_stdout(output):
            first = run._get_image_from_pi_camera((5, 4))
            second = run._get_image_from_pi_camera((5, 4))

        self.assertIs(first, expected)
        self.assertIs(second, expected)
        text = output.getvalue()
        self.assertEqual(text.count("Camera Capture Backend:"), 1)
        self.assertIn("libcamera-still", text)
        self.assertNotIn("picamera2` not found", text)
        self.assertNotIn("Cannot capture from Pi camera", text)

    def test_libcamera_backend_is_used_even_when_picamera2_is_available(self) -> None:
        expected = np.zeros((4, 5, 3), dtype=np.uint8)
        run._picamera2_available = lambda: True
        run._capture_image_with_picamera2 = lambda _resolution: self.fail("picamera2 should not be called")
        run._capture_image_with_libcamera_still = lambda _resolution: expected

        output = io.StringIO()
        with redirect_stdout(output):
            image = run._get_image_from_pi_camera((5, 4))

        self.assertIs(image, expected)
        self.assertIn("Camera Capture Backend: libcamera-still", output.getvalue())

    def test_picamera2_manual_backend_runtime_failure_switches_to_libcamera_once(self) -> None:
        expected = np.zeros((4, 5, 3), dtype=np.uint8)
        calls = {"picamera2": 0, "libcamera": 0}

        def picamera2_failure(_resolution):
            calls["picamera2"] += 1
            return None

        def libcamera_success(_resolution):
            calls["libcamera"] += 1
            return expected

        run._picamera2_available = lambda: True
        run._SELECTED_CAPTURE_BACKEND = run.CAMERA_BACKEND_PICAMERA2
        run._capture_image_with_picamera2 = picamera2_failure
        run._capture_image_with_libcamera_still = libcamera_success

        output = io.StringIO()
        with redirect_stdout(output):
            first = run._get_image_from_pi_camera((5, 4))
            second = run._get_image_from_pi_camera((5, 4))

        self.assertIs(first, expected)
        self.assertIs(second, expected)
        self.assertEqual(calls["picamera2"], 1)
        self.assertEqual(calls["libcamera"], 2)
        text = output.getvalue()
        self.assertEqual(text.count("Camera Capture Backend:"), 1)
        self.assertEqual(text.count("switching to libcamera-still"), 1)

    def test_libcamera_still_uses_configured_resolution_and_two_second_timeout(self) -> None:
        expected = np.zeros((720, 1280, 3), dtype=np.uint8)
        completed = mock.Mock()
        completed.returncode = 0
        completed.stderr = ""
        tmp_dir = self.case_dir / "tmp_capture"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        with (
            mock.patch.object(run.tempfile, "TemporaryDirectory", return_value=mock.MagicMock(
                __enter__=mock.Mock(return_value=str(tmp_dir)),
                __exit__=mock.Mock(return_value=False),
            )),
            mock.patch.object(run.subprocess, "run", return_value=completed) as subprocess_run,
            mock.patch.object(run.cv2, "imread", return_value=expected),
        ):
            image = run._capture_image_with_libcamera_still((1280, 720))

        self.assertIs(image, expected)
        cmd = subprocess_run.call_args.args[0]
        self.assertEqual(cmd[0:8], [
            "libcamera-still",
            "--nopreview",
            "--width",
            "1280",
            "--height",
            "720",
            "--timeout",
            "2000",
        ])
        self.assertIn("--output", cmd)
        self.assertTrue(subprocess_run.call_args.kwargs["stdout"] is run.subprocess.PIPE)
        self.assertTrue(subprocess_run.call_args.kwargs["stderr"] is run.subprocess.PIPE)

    def test_run_monitoring_prints_selected_backend_once_at_startup(self) -> None:
        run._picamera2_available = lambda: False
        run.load_configuration = lambda: {
            "roi_coordinates": [0, 0, 4, 4],
            "warning_threshold": 0.6,
            "critical_threshold": 1.2,
            "measurement_interval_seconds": 5,
            "log_directory": str(self.case_dir),
            "rpi_camera_resolution": (4, 4),
            "email_settings": None,
        }
        run._get_image_from_pi_camera = lambda _resolution: np.zeros((4, 4, 3), dtype=np.uint8)
        run._read_number_from_roi = lambda _roi_image: {
            "value": 0.1,
            "text": "0.100",
            "conf": 95.0,
            "raw": "[test]",
            "debug": {"attempt_count": 1, "source": "test"},
        }

        output = io.StringIO()
        with redirect_stdout(output):
            run.run_monitoring(once=True, no_alerts=True, save_debug_images=False)

        text = output.getvalue()
        self.assertEqual(text.count("Camera Capture Backend:"), 1)
        self.assertIn("Camera Capture Backend: libcamera-still", text)
        self.assertLess(text.index("Camera Capture Backend:"), text.index("Taking measurement"))


class RunEntrypointRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_get_image = run._get_image_from_pi_camera
        self.original_read_number = run._read_number_from_roi
        self.original_load_config = run.load_configuration
        self.original_picamera2_available = run._picamera2_available
        self.original_selected_backend = run._SELECTED_CAPTURE_BACKEND
        self.original_backend_announced = run._CAPTURE_BACKEND_ANNOUNCED
        self.original_fallback_announced = run._PICAMERA2_FALLBACK_ANNOUNCED
        self.case_dir = ROOT / "logs" / "run_entrypoint_runtime_tests" / uuid.uuid4().hex
        run._SELECTED_CAPTURE_BACKEND = None
        run._CAPTURE_BACKEND_ANNOUNCED = False
        run._PICAMERA2_FALLBACK_ANNOUNCED = False

    def tearDown(self) -> None:
        run._get_image_from_pi_camera = self.original_get_image
        run._read_number_from_roi = self.original_read_number
        run.load_configuration = self.original_load_config
        run._picamera2_available = self.original_picamera2_available
        run._SELECTED_CAPTURE_BACKEND = self.original_selected_backend
        run._CAPTURE_BACKEND_ANNOUNCED = self.original_backend_announced
        run._PICAMERA2_FALLBACK_ANNOUNCED = self.original_fallback_announced
        shutil.rmtree(self.case_dir, ignore_errors=True)

    def test_run_py_is_raspberry_runtime_entrypoint(self) -> None:
        calls = {"ocr": 0}
        run._picamera2_available = lambda: False
        run.load_configuration = lambda: {
            "roi_coordinates": [0, 0, 4, 4],
            "warning_threshold": 0.6,
            "critical_threshold": 1.2,
            "measurement_interval_seconds": 5,
            "log_directory": str(self.case_dir),
            "rpi_camera_resolution": (4, 4),
            "email_settings": None,
        }
        run._get_image_from_pi_camera = lambda _resolution: np.zeros((4, 4, 3), dtype=np.uint8)

        def read_number(roi_image):
            calls["ocr"] += 1
            self.assertEqual(roi_image.shape, (4, 4, 3))
            return {
                "value": 0.1,
                "text": "0.100",
                "conf": 95.0,
                "raw": "[test]",
                "debug": {"attempt_count": 1, "source": "test"},
            }

        run._read_number_from_roi = read_number

        output = io.StringIO()
        with redirect_stdout(output):
            run.run_monitoring(once=True, no_alerts=True, save_debug_images=False)

        text = output.getvalue()
        self.assertEqual(calls["ocr"], 1)
        self.assertIn("Starting Multimeter Monitoring on Raspberry Pi", text)
        self.assertIn("OCR Strategy: Raspberry Pi lightweight OCR", text)
        self.assertIn("One-shot measurement complete.", text)


if __name__ == "__main__":
    unittest.main()
