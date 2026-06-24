
import argparse
import cv2
import json
import os
import subprocess
import sys
import time
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- GLOBAL CONFIGURATION / MODE SELECTION (EASILY ACCESSIBLE) ---
# Raspberry Pi setup is the default. Use --pc-test for PC fixture setup.
PC_TEST_MODE = False
# ------------------------------------------------------------------

# --- PC TEST MODE DEFAULT SETUP PARAMETERS (ONLY USED IF PC_TEST_MODE IS TRUE) ---
# When PC_TEST_MODE is True, these values will be used automatically.
PC_TEST_SETUP_DEFAULTS = {
    # Path to a test image for ROI selection in PC_TEST_MODE.
    "test_image_path": os.path.join(
        PROJECT_DIR,
        "test_sets",
        "green_multimeter_v3_cleaned",
        "cropped",
        "meter_hold_1p2309.jpg",
    ),
    "warning_threshold": 0.6,
    "critical_threshold": 1.2,
    "measurement_interval_seconds": 5, # 5 seconds for faster PC test-mode iteration

    # Set to True to include dummy email settings in test config, False to skip.
    "email_setup_enabled": False, # Set to True to test email config saving
    "email_settings": {
        "sender_email": "test_sender@example.com",
        "sender_app_password": "dummy_app_password",
        "recipient_email": "test_recipient@example.com",
        "smtp_server": "smtp.dummy.com",
        "smtp_port": 587
    },
    # Default ROI coordinates for testing.
    # If set to None, it will still open the GUI for manual ROI selection even in PC_TEST_MODE.
    # If you want fully headless PC testing, provide pre-determined ROI coordinates here.
    "roi_coordinates": None # Set to (x1, y1, x2, y2) tuple for headless ROI in PC test mode, e.g., (315, 445, 780, 655)
}
# ----------------------------------------------------------------------------------


# --- Other Global Configuration / Output File Names ---
OUTPUT_CONFIG_FILE = os.path.join(PROJECT_DIR, 'config.json')
OUTPUT_INITIAL_IMAGE_NAME = 'initial_display.jpg'
SETUP_IMAGE_DIR = os.path.join(PROJECT_DIR, 'setup_images')

RPI_CAMERA_RESOLUTION = (1280, 720)
LIBCAMERA_TIMEOUT_MS = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Multimeter OCR setup.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--pc-test', action='store_true', help='Use PC test image/default setup behavior.')
    mode.add_argument('--pi', action='store_true', help='Use Raspberry Pi camera setup behavior.')
    return parser.parse_args()


def resolve_pc_test_mode(args: argparse.Namespace) -> bool:
    if args.pi:
        return False
    if args.pc_test:
        return True
    return PC_TEST_MODE


def _stop_preview_process(preview_proc) -> None:
    if preview_proc is None:
        return
    if preview_proc.poll() is not None:
        return
    print('Stopping live camera preview...')
    preview_proc.terminate()
    try:
        preview_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print('Preview did not stop promptly; killing it.')
        preview_proc.kill()
        preview_proc.wait(timeout=5)


def _start_libcamera_preview(resolution: tuple):
    width, height = (int(resolution[0]), int(resolution[1]))
    cmd = [
        'libcamera-hello',
        '--timeout',
        '0',
        '--width',
        str(width),
        '--height',
        str(height),
    ]
    print('\n--- Raspberry Pi Camera Live Preview ---')
    print(f'Starting live preview at {width}x{height} with libcamera-hello.')
    print('Adjust camera position, focus, and lighting until the display is clear.')
    print('Press Enter in this terminal when ready to capture a still image.')
    print('Press Ctrl+C to cancel setup.')
    try:
        return subprocess.Popen(cmd)
    except FileNotFoundError:
        print('Error: `libcamera-hello` not found. Install/enable Raspberry Pi camera tools.')
        return None
    except Exception as e:
        print(f'Error starting libcamera preview: {e}')
        return None


def _preview_then_capture_setup_image(output_filename: str, resolution: tuple) -> bool:
    preview_proc = _start_libcamera_preview(resolution)
    if preview_proc is None:
        print('Live preview is unavailable; continuing with still capture.')
    else:
        try:
            input('Press Enter to capture setup image...')
        except KeyboardInterrupt:
            print('\nSetup cancelled during camera preview.')
            _stop_preview_process(preview_proc)
            return False
        finally:
            _stop_preview_process(preview_proc)
            time.sleep(0.5)

    print('\n--- Capturing Setup Still Image ---')
    return _capture_image_with_libcamera_still(output_filename, resolution)


def _capture_image_with_libcamera_still(output_filename: str, resolution: tuple) -> bool:
    width, height = (int(resolution[0]), int(resolution[1]))
    cmd = [
        'libcamera-still',
        '--nopreview',
        '--timeout',
        str(LIBCAMERA_TIMEOUT_MS),
        '--width',
        str(width),
        '--height',
        str(height),
        '--output',
        output_filename,
    ]

    print(f"Capturing image with libcamera-still to '{output_filename}' at resolution {resolution}...")
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        print('Error: `libcamera-still` not found. Install/enable Raspberry Pi camera tools.')
        return False
    except Exception as e:
        print(f'Error running libcamera-still: {e}')
        return False

    if completed.returncode != 0:
        print(f'Error: libcamera-still failed with exit code {completed.returncode}.')
        stderr = (completed.stderr or '').strip()
        if stderr:
            print(stderr)
        return False

    image = cv2.imread(output_filename)
    if image is None:
        print(f"Error: libcamera-still completed, but OpenCV could not read '{output_filename}'.")
        return False

    if image.shape[1] != width or image.shape[0] != height:
        print(
            f'Warning: captured image size is {image.shape[1]}x{image.shape[0]}, '
            f'expected {width}x{height}.'
        )

    print(f"Image captured successfully with libcamera-still to '{output_filename}'.")
    return True


def _save_setup_debug_copy(image_path: str) -> None:
    """Keep a timestamped copy of the setup capture for manual ROI/debug use."""
    if not os.path.exists(image_path):
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: could not reload setup image for debug copy: {image_path}")
        return

    setup_dir = os.path.dirname(image_path) or "."
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_path = os.path.join(setup_dir, f"setup_capture_{timestamp}.jpg")
    if cv2.imwrite(debug_path, image):
        print(f"Saved setup/debug image: {debug_path}")
    else:
        print(f"Warning: failed to save setup/debug image: {debug_path}")


def get_initial_image_for_roi_selection() -> str:
    """Return a PC test image or capture a Raspberry Pi setup image."""
    os.makedirs(SETUP_IMAGE_DIR, exist_ok=True)

    if PC_TEST_MODE:
        initial_image_path = os.path.abspath(PC_TEST_SETUP_DEFAULTS["test_image_path"])
        print('\n--- Running in PC Test Mode: Using predefined test image ---')
        print(f"Test image path: '{initial_image_path}'")
        if not os.path.exists(initial_image_path):
            print(f"Error: Test image file not found at '{initial_image_path}'.")
            print("Update PC_TEST_SETUP_DEFAULTS['test_image_path'] with a correct path.")
            sys.exit(1)
    else:
        print('\n--- Running in Raspberry Pi Mode: Preview and capture camera image ---')
        initial_image_path = os.path.join(SETUP_IMAGE_DIR, OUTPUT_INITIAL_IMAGE_NAME)
        success = _preview_then_capture_setup_image(initial_image_path, RPI_CAMERA_RESOLUTION)
        if not success:
            print('Failed to capture image from Raspberry Pi camera. Exiting setup.')
            sys.exit(1)
        _save_setup_debug_copy(initial_image_path)

    print(f"Image ready for ROI selection: '{initial_image_path}'")
    return initial_image_path


def _safe_destroy_window(window_name: str) -> None:
    try:
        cv2.destroyWindow(window_name)
    except Exception:
        pass


def _safe_destroy_all_windows() -> None:
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


def _roi_from_xywh(x: int, y: int, w: int, h: int, image_shape: tuple) -> tuple:
    img_h, img_w = image_shape[:2]
    if w <= 0 or h <= 0:
        print("Invalid ROI: width and height must be positive.")
        return ()

    x1 = max(0, min(img_w, int(x)))
    y1 = max(0, min(img_h, int(y)))
    x2 = max(0, min(img_w, int(x + w)))
    y2 = max(0, min(img_h, int(y + h)))
    if x2 <= x1 or y2 <= y1:
        print(f"Invalid ROI after clipping to image bounds {img_w}x{img_h}.")
        return ()

    if (x1, y1, x2, y2) != (x, y, x + w, y + h):
        print(f"ROI clipped to image bounds: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    return (x1, y1, x2, y2)


def _prompt_manual_roi(image_shape: tuple, image_path: str) -> tuple:
    img_h, img_w = image_shape[:2]
    print('\n--- Manual ROI Entry ---')
    print(f"Setup image: {image_path}")
    print(f"Image size: width={img_w}, height={img_h}")
    print("Enter ROI as x,y,w,h in image pixels. Example: 315,445,465,210")
    print("This will be saved to config.json as x1,y1,x2,y2 for run.py compatibility.")

    while True:
        raw = input("Manual ROI x,y,w,h, or Q to quit: ").strip()
        if raw.lower() in ("q", "quit", "exit"):
            print("Manual ROI entry cancelled.")
            return ()
        if not raw:
            print("Please enter four numbers, or Q to quit.")
            continue

        parts = raw.replace(",", " ").split()
        if len(parts) != 4:
            print("Invalid format. Enter exactly four values: x,y,w,h")
            continue

        try:
            x, y, w, h = (int(float(part)) for part in parts)
        except ValueError:
            print("Invalid format. ROI values must be numbers.")
            continue

        roi = _roi_from_xywh(x, y, w, h, image_shape)
        if not roi:
            continue

        x1, y1, x2, y2 = roi
        print(f"Manual ROI selected: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        return roi


def _select_roi_with_mouse(image):
    window_name = "Setup ROI Selection"
    img_h, img_w = image.shape[:2]
    max_display_w = 1280
    max_display_h = 800
    scale = min(1.0, max_display_w / float(img_w), max_display_h / float(img_h))
    if scale < 1.0:
        display = cv2.resize(
            image,
            (max(1, int(img_w * scale)), max(1, int(img_h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        display = image.copy()

    state = {
        "drawing": False,
        "start": None,
        "current": None,
        "roi": None,
    }

    def normalize_rect(p1, p2):
        x1 = min(int(p1[0]), int(p2[0]))
        y1 = min(int(p1[1]), int(p2[1]))
        x2 = max(int(p1[0]), int(p2[0]))
        y2 = max(int(p1[1]), int(p2[1]))
        if x2 - x1 < 3 or y2 - y1 < 3:
            return None
        return (x1, y1, x2, y2)

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["start"] = (x, y)
            state["current"] = (x, y)
            state["roi"] = None
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["current"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state["drawing"] = False
            state["current"] = (x, y)
            rect = normalize_rect(state["start"], state["current"]) if state["start"] else None
            state["roi"] = rect

    def display_to_original(rect):
        x1, y1, x2, y2 = rect
        if scale != 1.0:
            x1 = int(round(x1 / scale))
            y1 = int(round(y1 / scale))
            x2 = int(round(x2 / scale))
            y2 = int(round(y2 / scale))
        return (
            max(0, min(img_w, x1)),
            max(0, min(img_h, y1)),
            max(0, min(img_w, x2)),
            max(0, min(img_h, y2)),
        )

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display.shape[1], display.shape[0])
        cv2.setMouseCallback(window_name, on_mouse)
    except Exception as e:
        print(f"OpenCV GUI ROI window could not be opened: {e}")
        _safe_destroy_window(window_name)
        return None

    print('\nOpenCV ROI window opened.')
    print('Drag with the left mouse button around the numeric LCD window.')
    print('ENTER or SPACE = accept, R = clear/redraw, Q/ESC = cancel.')

    try:
        while True:
            frame = display.copy()
            rect = None
            if state["drawing"] and state["start"] and state["current"]:
                rect = normalize_rect(state["start"], state["current"])
            elif state["roi"]:
                rect = state["roi"]

            if rect:
                color = (0, 255, 255) if state["drawing"] else (0, 255, 0)
                cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), color, 2)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10, 32):
                if not state["roi"]:
                    print("No ROI was selected.")
                    return None
                x1, y1, x2, y2 = display_to_original(state["roi"])
                if x2 <= x1 or y2 <= y1:
                    print("Selected ROI is empty.")
                    return None
                print(f"ROI selected: (x1={x1}, y1={y1}, x2={x2}, y2={y2})")
                return (x1, y1, x2, y2)
            if key in (ord("r"), ord("R")):
                state["drawing"] = False
                state["start"] = None
                state["current"] = None
                state["roi"] = None
                print("ROI cleared. Draw again.")
            elif key in (27, ord("q"), ord("Q")):
                print("ROI window cancelled.")
                return ()
    except KeyboardInterrupt:
        print("\nROI selection interrupted.")
        return ()
    finally:
        _safe_destroy_window(window_name)


def select_roi(image_path: str) -> tuple:
    """
    Select a Region of Interest (ROI) from the setup image.

    On Raspberry Pi, OpenCV's built-in blocking ROI helper can freeze under
    some desktop/camera combinations, so this uses a small mouse callback loop
    with a manual x,y,w,h fallback.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: (x1, y1, x2, y2) coordinates of the selected ROI, or an empty tuple if selection was cancelled.
    """
    print('\n--- Stage 2: Interactive ROI Selection ---')
    try:
        if PC_TEST_MODE and PC_TEST_SETUP_DEFAULTS["roi_coordinates"] is not None:
            print("Using predefined ROI coordinates for PC Test Mode (headless).")
            return PC_TEST_SETUP_DEFAULTS["roi_coordinates"]

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f'Could not read image at: {image_path}')

        print(f"Setup image saved at: {image_path}")
        print(f"Image size: width={image.shape[1]}, height={image.shape[0]}")
        print('Include all digits and the decimal point, with a little LCD background margin.')
        print('Opening captured still image for ROI selection.')

        roi = _select_roi_with_mouse(image)
        if roi:
            return roi
        if roi == ():
            return ()

        print("GUI ROI selection did not complete. Falling back to manual coordinate entry.")
        return _prompt_manual_roi(image.shape, image_path)

    except Exception as e:
         print(f'Error during ROI selection: {e}')
         print('Please check:')
         print('   - The image path is correct.')
         print('   - OpenCV is installed (`pip install opencv-python`).')
         _safe_destroy_all_windows()
         return () # Return an empty tuple on error


def crop_roi_from_image(image, roi_coordinates: tuple) -> any:
    x1, y1, x2, y2 = roi_coordinates
    img_h, img_w = image.shape[:2]
    x1 = max(0, min(img_w, int(x1)))
    y1 = max(0, min(img_h, int(y1)))
    x2 = max(0, min(img_w, int(x2)))
    y2 = max(0, min(img_h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2].copy()


def format_detected_value(value) -> str:
    if value is None:
        return "<no reading>"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def confirm_roi_readback(image_path: str, roi_coordinates: tuple) -> bool:
    """Validate a PC-test ROI and let the user accept or redraw it."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print('Warning: Could not reload image for ROI confirmation.')
            return False

        roi_image = crop_roi_from_image(image, roi_coordinates)
        if roi_image is None or roi_image.size == 0:
            print('Warning: Selected ROI is empty.')
            return False

        import run

        detected_value = run.extract_number_from_image_with_roi(
            image,
            roi_coordinates,
            pc_test_mode=True,
        )
        detected_text = format_detected_value(detected_value)
        print(f'\nSetup check: detected reading inside ROI = {detected_text}')

        while True:
            choice = input('Accept this ROI? [Y]es / [R]edraw / [Q]uit: ').strip().lower()
            if choice in ('', 'y', 'yes'):
                return True
            if choice in ('r', 'redraw'):
                return False
            if choice in ('q', 'quit'):
                print('Setup cancelled.')
                sys.exit(1)
            print('Please enter Y, R, or Q.')

    except Exception as e:
        print(f'Warning: ROI confirmation failed: {e}')
        return False


def _read_pi_setup_ocr_result(image, roi_coordinates: tuple) -> dict:
    try:
        import run

        return run._read_number_from_image_with_roi_result(image, roi_coordinates)
    except Exception as e:
        return {
            'value': None,
            'text': '',
            'conf': 0.0,
            'raw': '',
            'debug': {'rejected': str(e)},
        }


def confirm_roi_readback_pi(image_path: str, roi_coordinates: tuple) -> str:
    """
    Validate the selected ROI with the same lightweight OCR path used by run.py.
    Returns one of: accept, redraw, preview, quit.
    """
    try:
        print('\n--- Stage 2b: Raspberry Pi OCR ROI Validation ---')
        image = cv2.imread(image_path)
        if image is None:
            print("Warning: Could not reload image for ROI confirmation.")
            return "redraw"

        roi_image = crop_roi_from_image(image, roi_coordinates)
        if roi_image is None or roi_image.size == 0:
            print("Warning: Selected ROI is empty.")
            return "redraw"

        result = _read_pi_setup_ocr_result(image, roi_coordinates)
        value = result.get("value")
        text = str(result.get("text", ""))
        conf = float(result.get("conf", 0.0) or 0.0)
        raw = str(result.get("raw", ""))
        debug = result.get("debug", {})
        rejected = debug.get("rejected", "") if isinstance(debug, dict) else ""
        detected_text = format_detected_value(value)
        print(f"Setup check using Pi OCR: detected reading inside ROI = {detected_text}")
        print(f"OCR text='{text}' conf={conf:.1f} raw='{raw}'")
        if rejected:
            print(f"OCR rejected reason: {rejected}")

        while True:
            choice = input(
                "Accept this ROI/config? [Y]es / [D]raw again / [P]review-capture again / [Q]uit: "
            ).strip().lower()
            if choice in ("", "y", "yes"):
                return "accept"
            if choice in ("d", "draw", "redraw"):
                return "redraw"
            if choice in ("p", "preview"):
                return "preview"
            if choice in ("q", "quit"):
                return "quit"
            print("Please enter Y, D, P, or Q.")

    except Exception as e:
        print(f"Warning: ROI confirmation failed: {e}")
        return "redraw"


def select_and_confirm_roi(image_path: str) -> tuple:
    while True:
        roi_coordinates = select_roi(image_path)
        if not roi_coordinates:
            return ()
        if confirm_roi_readback(image_path, roi_coordinates):
            return roi_coordinates


def select_and_confirm_roi_pi(image_path: str) -> tuple:
    current_image_path = image_path
    while True:
        roi_coordinates = select_roi(current_image_path)
        if not roi_coordinates:
            return "", ()

        action = confirm_roi_readback_pi(current_image_path, roi_coordinates)
        if action == "accept":
            return current_image_path, roi_coordinates
        if action == "redraw":
            print("Redraw ROI on the same captured image.")
            continue
        if action == "preview":
            print("Restarting live preview and capturing a new setup image.")
            current_image_path = get_initial_image_for_roi_selection()
            continue
        if action == "quit":
            print("Setup cancelled.")
            sys.exit(1)

def get_thresholds_and_email_settings() -> dict:
    """Build a complete configuration only for explicit PC test mode."""
    if not PC_TEST_MODE:
        raise RuntimeError('Raspberry Pi setup must preserve the existing config.json.')

    print('\n--- Stage 3: Building PC Test Configuration ---')
    settings = {
        'warning_threshold': PC_TEST_SETUP_DEFAULTS['warning_threshold'],
        'critical_threshold': PC_TEST_SETUP_DEFAULTS['critical_threshold'],
        'measurement_interval_seconds': PC_TEST_SETUP_DEFAULTS['measurement_interval_seconds'],
        'email_settings': (
            dict(PC_TEST_SETUP_DEFAULTS['email_settings'])
            if PC_TEST_SETUP_DEFAULTS['email_setup_enabled']
            else None
        ),
        'log_directory': './logs/',
    }
    return settings


def save_configuration(config_data: dict, output_file: str) -> bool:
    """Atomically save configuration data to JSON."""
    print('\n--- Saving Configuration ---')
    output_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(output_dir, exist_ok=True)
    temp_file = f'{output_file}.tmp'

    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4)
            f.write('\n')
        os.replace(temp_file, output_file)
        print(f'Configuration saved successfully to: {output_file}')
        return True
    except Exception as e:
        print(f'Error saving configuration: {e}')
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except OSError:
            pass
        return False


def _portable_project_path(path: str) -> str:
    absolute_path = os.path.abspath(path)
    try:
        if os.path.commonpath((PROJECT_DIR, absolute_path)) == PROJECT_DIR:
            return os.path.relpath(absolute_path, PROJECT_DIR).replace(os.sep, '/')
    except ValueError:
        pass
    return path


# --- Main execution block for setup.py ---
def main() -> int:
    global PC_TEST_MODE

    args = parse_args()
    PC_TEST_MODE = resolve_pc_test_mode(args)

    print('--- Multimeter OCR Setup ---')
    print(f"Current mode: {'PC Test Mode' if PC_TEST_MODE else 'Raspberry Pi Mode'}")

    initial_image_file_path = get_initial_image_for_roi_selection()

    if PC_TEST_MODE:
        roi_coordinates = select_and_confirm_roi(initial_image_file_path)
    else:
        initial_image_file_path, roi_coordinates = select_and_confirm_roi_pi(initial_image_file_path)

    if not roi_coordinates:
        print('Setup cancelled. Exiting.')
        return 1

    if PC_TEST_MODE:
        config_settings = get_thresholds_and_email_settings()
    else:
        try:
            with open(OUTPUT_CONFIG_FILE, 'r', encoding='utf-8') as config_file:
                config_settings = json.load(config_file)
            if not isinstance(config_settings, dict):
                raise ValueError('Existing configuration is not a JSON object.')
            print(f'Updating existing configuration: {OUTPUT_CONFIG_FILE}')
        except FileNotFoundError:
            print(f'Configuration file not found: {OUTPUT_CONFIG_FILE}')
            print('Create the base configuration before running Raspberry Pi setup.')
            return 1
        except Exception as e:
            print(f'Could not load existing configuration: {e}')
            return 1

    # Preserve all existing Raspberry Pi settings and update only setup-owned fields.
    config_settings['PC_TEST_MODE'] = PC_TEST_MODE
    config_settings['initial_image_for_roi'] = _portable_project_path(initial_image_file_path)
    config_settings['roi_coordinates'] = [int(value) for value in roi_coordinates]
    config_settings['rpi_camera_resolution'] = list(RPI_CAMERA_RESOLUTION)
    config_settings['camera_resolution'] = list(RPI_CAMERA_RESOLUTION)

    if not save_configuration(config_settings, OUTPUT_CONFIG_FILE):
        print('\nSetup failed. Check the error above.')
        return 1

    print('\nSetup completed successfully.')
    print(f'ROI: {tuple(config_settings["roi_coordinates"])}')
    print(f'Camera resolution: {tuple(config_settings["rpi_camera_resolution"])}')
    print('Existing thresholds, interval, email, and logging settings were preserved.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
