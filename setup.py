
import argparse
import cv2
import json
import os
import subprocess
import sys
import time
from datetime import datetime

from run import extract_number_from_image_with_roi

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- GLOBAL CONFIGURATION / MODE SELECTION (EASILY ACCESSIBLE) ---
# Set this to True for PC testing where setup parameters are read from defaults below.
# Set this to False for Raspberry Pi deployment, requiring interactive setup.
PC_TEST_MODE = True  # <--- TOGGLE THIS FOR YOUR DESIRED MODE
# ------------------------------------------------------------------

# --- PC TEST MODE DEFAULT SETUP PARAMETERS (ONLY USED IF PC_TEST_MODE IS TRUE) ---
# When PC_TEST_MODE is True, these values will be used automatically.
PC_TEST_SETUP_DEFAULTS = {
    # Path to a test image for ROI selection in PC_TEST_MODE.
    # >>> IMPORTANT: CHANGE THIS PATH to one of your collected test images <<<
    "test_image_path": os.path.join(PROJECT_DIR, "test_v2", "ram_gene_0p03.png"),
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

RPI_CAMERA_RESOLUTION = (1920, 1080)  # Camera resolution for Raspberry Pi capture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Smart Geiger Counter Interface setup.')
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


def _start_libcamera_preview():
    cmd = ['libcamera-hello', '-t', '0']
    print('\n--- Raspberry Pi Camera Live Preview ---')
    print('Starting live preview with libcamera-hello.')
    print('Adjust camera position, focus, and lighting until the LCD is clear.')
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
    preview_proc = _start_libcamera_preview()
    try:
        input('Press Enter to capture setup image...')
    except KeyboardInterrupt:
        print('\nSetup cancelled during camera preview.')
        _stop_preview_process(preview_proc)
        return False

    _stop_preview_process(preview_proc)
    time.sleep(0.5)
    print('\n--- Capturing Setup Still Image ---')
    return _capture_image_from_pi_camera(output_filename, resolution)


def _capture_image_from_pi_camera(output_filename: str, resolution: tuple) -> bool:
    """
    Internal helper function to capture an image using picamera2 on a Raspberry Pi.
    This function should only be called if PC_TEST_MODE is False.
    """
    picam2 = None
    started = False
    try:
        # Dynamically import picamera2 here to avoid ImportErrors on PC
        from picamera2 import Picamera2

        print('Initializing Raspberry Pi Camera...')
        picam2 = Picamera2()

        camera_config = picam2.create_still_configuration(main={'size': resolution})
        picam2.configure(camera_config)

        # Do not start a second GUI preview here. setup.py already offered a
        # libcamera preview before capture, and Picamera2 previews can conflict
        # with OpenCV windows on the Raspberry Pi desktop.
        picam2.start()
        started = True
        print('Camera started for still capture. Waiting 2 seconds for auto-adjustments...')
        time.sleep(2)  # Give camera time to adjust exposure/white balance

        print(f"Capturing image to '{output_filename}' at resolution {resolution}...")
        picam2.capture_file(output_filename)
        print(f"Image captured successfully to '{output_filename}'.")
        return True

    except ImportError:
        print('Error: `picamera2` not found. Is this a Raspberry Pi, and is picamera2 installed?')
    except Exception as e:
        print(f'Error capturing image from Pi camera: {e}')
        print('Please check camera connection, power, and permissions.')
    finally:
        if picam2 is not None:
            try:
                if started:
                    picam2.stop()
                picam2.close()
                print('Raspberry Pi Camera resources released.')
            except Exception as e:
                print(f'Error during camera cleanup: {e}')

    print('Falling back to libcamera-still for setup capture...')
    return _capture_image_with_libcamera_still(output_filename, resolution)


def _capture_image_with_libcamera_still(output_filename: str, resolution: tuple) -> bool:
    width, height = (int(resolution[0]), int(resolution[1]))
    cmd = [
        'libcamera-still',
        '--nopreview',
        '--timeout',
        '1000',
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
    """
    Determines how to get the initial image for ROI selection based on PC_TEST_MODE.
    Returns the path to the image file.
    """
    initial_image_path = ''
    target_image_dir = 'setup_images'  # Directory to save/look for setup images
    os.makedirs(target_image_dir, exist_ok=True)  # Ensure directory exists

    if PC_TEST_MODE:
        initial_image_path = PC_TEST_SETUP_DEFAULTS["test_image_path"]
        print(f'\n--- Running in PC Test Mode: Using predefined test image ---')
        print(f'Test image path: \'{initial_image_path}\'')
        if not os.path.exists(initial_image_path):
            print(f'Error: Test image file not found at \'{initial_image_path}\'.')
            print("Please update PC_TEST_SETUP_DEFAULTS['test_image_path'] with a correct path.")
            sys.exit(1)
    else:
        print('\n--- Running in Raspberry Pi Mode: Preview and capture camera image ---')
        initial_image_path = os.path.join(target_image_dir, OUTPUT_INITIAL_IMAGE_NAME)
        success = _preview_then_capture_setup_image(initial_image_path, RPI_CAMERA_RESOLUTION)
        if not success:
            print('Failed to capture image from Raspberry Pi camera. Exiting setup.')
            sys.exit(1)
        _save_setup_debug_copy(initial_image_path)

    print(f'Image ready for ROI selection: \'{initial_image_path}\'')
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
    print("This will be saved to config.json as x1,y1,x2,y2 for run_pi.py compatibility.")

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

            try:
                visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
            except Exception as e:
                print(f"ROI window check failed: {e}")
                return None
            if visible < 1:
                print("ROI window was closed.")
                return None

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
    """
    Reads the current display inside the selected ROI and lets the user accept or redraw.
    Returns True if the ROI is accepted, False if it should be redrawn or setup should stop.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print("Warning: Could not reload image for ROI confirmation.")
            return True

        roi_image = crop_roi_from_image(image, roi_coordinates)
        if roi_image is None or roi_image.size == 0:
            print("Warning: Selected ROI is empty.")
            return False

        detected_value = extract_number_from_image_with_roi(
            image,
            roi_coordinates,
            pc_test_mode=PC_TEST_MODE,
        )
        detected_text = format_detected_value(detected_value)
        print(f"\nSetup check: detected reading inside ROI = {detected_text}")

        while True:
            choice = input("Accept this ROI? [Y]es / [R]edraw / [Q]uit: ").strip().lower()
            if choice in ("", "y", "yes"):
                return True
            if choice in ("r", "redraw"):
                return False
            if choice in ("q", "quit"):
                print("Setup cancelled.")
                sys.exit(1)
            print("Please enter Y, R, or Q.")

    except Exception as e:
        print(f"Warning: ROI confirmation failed: {e}")
        print("Continuing without confirmation.")
        return True


def _read_pi_setup_ocr_result(image, roi_coordinates: tuple) -> dict:
    try:
        import run_pi
        return run_pi._read_number_from_image_with_roi_result(image, roi_coordinates)
    except Exception as e:
        print(f"Warning: Raspberry Pi OCR validation failed: {e}")
        return {"value": None, "text": "", "conf": 0.0, "raw": "", "debug": {"rejected": str(e)}}


def confirm_roi_readback_pi(image_path: str, roi_coordinates: tuple) -> str:
    """
    Validate the selected ROI with the same lightweight OCR path used by run_pi.py.
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
    """
    Gets warning/critical thresholds, email settings, and measurement interval.
    If PC_TEST_MODE is True, it uses predefined defaults.
    Otherwise, it prompts the user via the command line.

    Returns:
        dict: A dictionary containing the collected settings.
              Returns an empty dictionary if any input is invalid/missing.
    """
    print('\n--- Stage 3: Getting Thresholds and Optional Email Settings ---')
    settings = {}

    if PC_TEST_MODE:
        print("Using predefined thresholds, interval, and email settings for PC Test Mode.")
        settings['warning_threshold'] = PC_TEST_SETUP_DEFAULTS["warning_threshold"]
        settings['critical_threshold'] = PC_TEST_SETUP_DEFAULTS["critical_threshold"]
        settings['measurement_interval_seconds'] = PC_TEST_SETUP_DEFAULTS["measurement_interval_seconds"]

        if PC_TEST_SETUP_DEFAULTS["email_setup_enabled"]:
            settings['email_settings'] = PC_TEST_SETUP_DEFAULTS["email_settings"]
            print("Email setup enabled in test mode.")
        else:
            settings['email_settings'] = None
            print("Email setup skipped in test mode.")

    else: # Raspberry Pi Mode - CLI Prompts
        try:
            # Get Thresholds
            while True:
                warning_threshold_str = input('Enter warning threshold (e.g., 0.5): ').strip()
                try:
                    settings['warning_threshold'] = float(warning_threshold_str)
                    break
                except ValueError:
                    print('Invalid input. Please enter a numerical value.')

            while True:
                critical_threshold_str = input('Enter critical threshold (e.g., 1.0): ').strip()
                try:
                     settings['critical_threshold'] = float(critical_threshold_str)
                     if settings['critical_threshold'] <= settings['warning_threshold']:
                         print('Critical threshold must be greater than warning threshold.')
                     else:
                         break
                except ValueError:
                    print('Invalid input. Please enter a numerical value.')

            # Get Email Settings (Optional)
            email_setup_choice = input('\nDo you want to set up email alerts? (yes/no): ').strip().lower()
            if email_setup_choice == 'yes':
                print('\n--- Email Settings ---')
                settings['email_settings'] = {}
                settings['email_settings']['sender_email'] = input('Enter sender email address: ').strip()
                settings['email_settings']['sender_app_password'] = input('Enter sender app password (NOT your regular password): ').strip()
                settings['email_settings']['recipient_email'] = input('Enter recipient email address: ').strip()
                settings['email_settings']['smtp_server'] = input('Enter SMTP server (e.g., smtp.gmail.com): ').strip()
                while True:
                    smtp_port_str = input('Enter SMTP port (e.g., 587): ').strip()
                    try:
                        settings['email_settings']['smtp_port'] = int(smtp_port_str)
                        break
                    except ValueError:
                         print('Invalid input. Please enter an integer port number.')
            else:
                print('Skipping email setup.')
                settings['email_settings'] = None # Store None if user doesn't want emails

            # Get Measurement Interval
            while True:
                interval_str = input('Enter measurement interval in seconds (e.g., 300 for 5 minutes): ').strip()
                try:
                    settings['measurement_interval_seconds'] = int(interval_str)
                    if settings['measurement_interval_seconds'] <= 0:
                        print('Interval must be a positive number.')
                    else:
                        break
                except ValueError:
                    print('Invalid input. Please enter an integer number of seconds.')

            print(f"\nCollected settings: {settings}") # For debugging in CLI mode

        except Exception as e:
            print(f'Error getting configuration settings: {e}')
            return {}  # Return an empty dictionary to signal failure

    # Set Log Directory (Default for both modes)
    settings['log_directory'] = './logs/'
    print(f"Log directory will be: {settings['log_directory']}")

    return settings


def save_configuration(config_data: dict, output_file: str) -> bool:
    """
    Saves the collected configuration data to a JSON file.

    Args:
        config_data (dict): The dictionary containing the configuration.
        output_file (str): The path to the JSON file to create/overwrite.
    Returns:
        bool: True if the configuration was saved successfully, False otherwise.
    """
    print('\n--- Stage 4: Saving Configuration to JSON ---')
    try:
        with open(output_file, 'w') as f:
            json.dump(config_data, f, indent=4)
        print(f'Configuration saved successfully to: {output_file}')
        return True
    except Exception as e:
        print(f'Error saving configuration: {e}')
        return False

# --- Main execution block for setup.py ---
if __name__ == '__main__':
    args = parse_args()
    PC_TEST_MODE = resolve_pc_test_mode(args)

    print('--- Smart Geiger Counter Interface Setup ---')
    print(f"Current mode: {'PC Test Mode' if PC_TEST_MODE else 'Raspberry Pi Mode'}")

    # Stage 1: Get the initial image (either from camera or file based on mode)
    initial_image_file_path = get_initial_image_for_roi_selection()

    # Stage 2: Interactive ROI Selection
    # If PC_TEST_MODE is True and 'roi_coordinates' are provided, it will skip GUI.
    # Otherwise, it will open the GUI for manual selection.
    if PC_TEST_MODE:
        roi_coordinates = select_and_confirm_roi(initial_image_file_path)
    else:
        initial_image_file_path, roi_coordinates = select_and_confirm_roi_pi(initial_image_file_path)
    if not roi_coordinates:  # User cancelled or error
        print('Setup cancelled. Exiting.')
        sys.exit(1)

    # Stage 3: Get Thresholds and Email Settings (based on mode)
    config_settings = get_thresholds_and_email_settings()

    # If any settings are missing, exit
    if not config_settings:
        print('Failed to get configuration settings. Exiting.')
        sys.exit(1)

    # Add ROI coordinates to the settings
    config_settings['PC_TEST_MODE'] = PC_TEST_MODE
    config_settings['initial_image_for_roi'] = initial_image_file_path
    config_settings['roi_coordinates'] = list(roi_coordinates)  # Convert tuple to list for JSON

    # Stage 4: Save the configuration to config.json
    save_successful = save_configuration(config_settings, OUTPUT_CONFIG_FILE)

    if save_successful:
        print('\nSetup completed successfully.')
        print(f'Configuration saved to: {OUTPUT_CONFIG_FILE}')
    else:
        print('\nSetup failed. Check for errors above.')
