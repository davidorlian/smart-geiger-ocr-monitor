import cv2
import json
import os
import sys

from run import extract_number_from_roi

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
    "test_image_path": os.path.join(PROJECT_DIR, "test", "Picture1.png"),
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
    "roi_coordinates": None # Set to (x1, y1, x2, y2) tuple for headless ROI in PC test mode, e.g., (499, 636, 715, 729)
}
# ----------------------------------------------------------------------------------


# --- Other Global Configuration / Output File Names ---
OUTPUT_CONFIG_FILE = os.path.join(PROJECT_DIR, 'config.json')
OUTPUT_INITIAL_IMAGE_NAME = 'initial_display.jpg'

RPI_CAMERA_RESOLUTION = (1920, 1080)  # Camera resolution for Raspberry Pi capture


def _capture_image_from_pi_camera(output_filename: str, resolution: tuple) -> bool:
    """
    Internal helper function to capture an image using picamera2 on a Raspberry Pi.
    This function should only be called if PC_TEST_MODE is False.
    """
    try:
        # Dynamically import picamera2 here to avoid ImportErrors on PC
        from picamera2 import Picamera2
        import time # Import time here as it's only needed for Pi camera

        print('Initializing Raspberry Pi Camera...')
        picam2 = Picamera2()

        camera_config = picam2.create_still_configuration(main={'size': resolution})
        picam2.configure(camera_config)

        picam2.start_preview()
        print('Camera preview started. Waiting 2 seconds for auto-adjustments...')
        time.sleep(2)  # Give camera time to adjust exposure/white balance

        print(f"Capturing image to '{output_filename}' at resolution {resolution}...")
        picam2.capture_file(output_filename)
        print(f"Image captured successfully to '{output_filename}'.")
        return True

    except ImportError:
        print('Error: `picamera2` not found. Is this a Raspberry Pi, and is picamera2 installed?')
        return False
    except Exception as e:
        print(f'Error capturing image from Pi camera: {e}')
        print('Please check camera connection, power, and permissions.')
        return False
    finally:
        if 'picam2' in locals() and picam2 is not None:
            try:
                picam2.stop_preview()
                picam2.close()
                print('Raspberry Pi Camera resources released.')
            except Exception as e:
                print(f'Error during camera cleanup: {e}')

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
        print('\n--- Running in Raspberry Pi Mode: Capturing live image from camera ---')
        initial_image_path = os.path.join(target_image_dir, OUTPUT_INITIAL_IMAGE_NAME)
        success = _capture_image_from_pi_camera(initial_image_path, RPI_CAMERA_RESOLUTION)
        if not success:
            print('Failed to capture image from Raspberry Pi camera. Exiting setup.')
            sys.exit(1)

    print(f'Image ready for ROI selection: \'{initial_image_path}\'')
    return initial_image_path

def select_roi(image_path: str) -> tuple:
    """
    Opens a window to display the image and allows the user to select a Region of Interest (ROI).
    Uses OpenCV's cv2.selectROI() function for interactive selection.

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

        print('Displaying image. Draw a rectangle around the whole LCD.')
        print('Press ENTER or SPACE to confirm, ESC or Q to cancel.')
        # cv2.selectROI will open a new window and wait for user input
        roi = cv2.selectROI('Select ROI', image, fromCenter=False, showCrosshair=True)  # Show crosshair for precision
        cv2.destroyAllWindows()  # Clean up the selection window

        if roi == (0, 0, 0, 0):  # User cancelled
            print('ROI selection cancelled.')
            return ()  # Return an empty tuple to indicate cancellation
        else:
            x, y, w, h = roi
            x1, y1, x2, y2 = x, y, x + w, y + h
            print(f'ROI selected: (x1={x1}, y1={y1}, x2={x2}, y2={y2})')
            return (x1, y1, x2, y2)

    except Exception as e:
         print(f'Error during ROI selection: {e}')
         print('Please check:')
         print('   - The image path is correct.')
         print('   - OpenCV is installed (`pip install opencv-python`).')
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

        detected_value = extract_number_from_roi(roi_image)
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


def select_and_confirm_roi(image_path: str) -> tuple:
    while True:
        roi_coordinates = select_roi(image_path)
        if not roi_coordinates:
            return ()
        if confirm_roi_readback(image_path, roi_coordinates):
            return roi_coordinates

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
    print('--- Smart Geiger Counter Interface Setup ---')
    print(f"Current mode: {'PC Test Mode' if PC_TEST_MODE else 'Raspberry Pi Mode'}")

    # Stage 1: Get the initial image (either from camera or file based on mode)
    initial_image_file_path = get_initial_image_for_roi_selection()

    # Stage 2: Interactive ROI Selection
    # If PC_TEST_MODE is True and 'roi_coordinates' are provided, it will skip GUI.
    # Otherwise, it will open the GUI for manual selection.
    roi_coordinates = select_and_confirm_roi(initial_image_file_path)
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
