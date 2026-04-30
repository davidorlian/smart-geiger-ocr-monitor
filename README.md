# Smart Geiger Counter Interface

This project monitors the numeric LCD reading of a handheld Geiger counter using a camera and OCR.

The system is designed around a simple user flow:

1. Run setup once.
2. Draw a rectangle around the LCD.
3. Confirm the detected number.
4. Run the monitor.

The runtime path is headless. It does not open OCR debug windows.

## Background

This project was motivated by the need to read a Geiger counter display without modifying the meter itself.

The practical idea is:

- use a camera to observe the existing LCD
- extract the reading from the screen image
- avoid electrical integration into the meter

That matters for two reasons:

1. Measurement should come from the approved instrument itself, not from a separate recalibrated interface.
2. A non-invasive solution is more realistic for institutional acceptance because it does not require changing the certified device.

In other words, the system is intended to read the display of the existing Geiger counter rather than replace its measurement chain.

## Constraints

The main engineering constraints behind the project are:

- the Geiger counter is the authoritative measurement source
- the project should avoid modifying the device hardware
- the OCR must tolerate imperfect framing and placement
- the camera setup may be physically constrained because the device can be mounted near a wall

That last point directly affects the implementation. The OCR and ROI selection flow were built to tolerate non-ideal camera angle and framing, because the display may not always be photographed from a perfectly centered position.

## Goal

The goal of the project is to turn a camera-facing Geiger counter into a monitored sensor that can:

- read the displayed value from the LCD
- compare the reading against warning and critical thresholds
- log measurements over time
- optionally send email alerts

## Repository Structure

- [setup.py](c:/Users/david/PycharmProjects/EmbeddedMiniProject/setup.py)  
  Interactive setup. Handles ROI selection and saves configuration.

- [run.py](c:/Users/david/PycharmProjects/EmbeddedMiniProject/run.py)  
  Monitoring runtime. Uses the saved ROI and performs periodic measurements.

- [test_ocr.py](c:/Users/david/PycharmProjects/EmbeddedMiniProject/test_ocr.py)  
  OCR lab tool for debugging and tuning on sample images.

- [config.json](c:/Users/david/PycharmProjects/EmbeddedMiniProject/config.json)  
  Saved configuration used by `run.py`.

- [config.example.json](c:/Users/david/PycharmProjects/EmbeddedMiniProject/config.example.json)  
  Example configuration for version control. Copy or regenerate a real `config.json` locally with `setup.py`.

- [test](c:/Users/david/PycharmProjects/EmbeddedMiniProject/test)  
  Test images used in PC Test Mode.

- [logs](c:/Users/david/PycharmProjects/EmbeddedMiniProject/logs)  
  Monitoring log output.

## How the OCR Works

The OCR pipeline is built for 7-segment LCD digits.

At a high level:

1. The user selects the whole LCD ROI.
2. The script tries several internal reading-band crops inside that ROI.
3. It tries a small set of fixed preprocessing variants.
4. It reads the mask as a 7-segment display first.
5. If that fails, it falls back to Tesseract.
6. The final reading is chosen by weighted voting across the internal candidates.

Important design choices:

- The system does not rely on filename-based answers.
- The runtime uses fixed internal OCR logic.
- The user is not expected to tune OCR sliders during normal use.

## Modes

### PC Test Mode

Used for development on a PC with saved images.

Current behavior:

- `setup.py` loads a predefined test image
- `run.py` cycles through the images in the `test` directory
- the measurement interval is intentionally short for fast iteration

This mode is controlled by:

- `PC_TEST_MODE = True` in [setup.py](c:/Users/david/PycharmProjects/EmbeddedMiniProject/setup.py)
- `"PC_TEST_MODE": true` in [config.json](c:/Users/david/PycharmProjects/EmbeddedMiniProject/config.json)

### Raspberry Pi Mode

Used for the real deployed camera setup.

Current behavior:

- `setup.py` captures a live image from the Pi camera
- the user selects the LCD once
- `run.py` repeatedly captures live frames and reads the saved ROI

For deployment:

- set `PC_TEST_MODE = False` in [setup.py](c:/Users/david/PycharmProjects/EmbeddedMiniProject/setup.py)
- rerun `setup.py`
- choose real thresholds and a real measurement interval

## Setup Workflow

Run:

```powershell
python3 setup.py
```

What happens:

1. An image is loaded or captured.
2. You draw a rectangle around the whole LCD.
3. The script runs OCR on that rectangle.
4. It asks for confirmation:

```text
Setup check: detected reading inside ROI = ...
Accept this ROI? [Y]es / [R]edraw / [Q]uit:
```

If the result is wrong, redraw the ROI.

After confirmation, the script saves:

- ROI coordinates
- warning threshold
- critical threshold
- measurement interval
- optional email settings

into [config.json](c:/Users/david/PycharmProjects/EmbeddedMiniProject/config.json).

## Runtime Workflow

Run:

```powershell
python3 run.py
```

The runtime:

- loads the saved config
- acquires a frame
- crops the saved ROI
- extracts the number
- compares it to the thresholds
- logs the result
- waits for the configured interval

The runtime does not open OCR tuning windows.

## OCR Lab Tool

Run:

```powershell
python3 test_ocr.py
```

Use this only for OCR development and investigation.

It provides:

- manual ROI selection per image
- debug windows
- calibration sliders
- OCR comparison across sample images
- visibility into the internal winning crop and preprocessing variant

This tool is not part of the normal user workflow.

## Configuration

Example fields in [config.json](c:/Users/david/PycharmProjects/EmbeddedMiniProject/config.json):

- `warning_threshold`
- `critical_threshold`
- `measurement_interval_seconds`
- `email_settings`
- `log_directory`
- `PC_TEST_MODE`
- `initial_image_for_roi`
- `roi_coordinates`

For upload:

- commit [config.example.json](c:/Users/david/PycharmProjects/EmbeddedMiniProject/config.example.json)
- keep your local [config.json](c:/Users/david/PycharmProjects/EmbeddedMiniProject/config.json) out of git

## Dependencies

Python:

- `opencv-python`
- `pytesseract`

System:

- Tesseract OCR installed and available on `PATH`

Raspberry Pi deployment:

- `picamera2`

## Notes

- In PC Test Mode, the current default interval is short for quick testing.
- In Raspberry Pi mode, rerun setup and choose the real measurement interval you want.
- The OCR is tuned for the current LCD display style and camera framing approach.
- The best real-world validation is repeated live-camera testing after deployment.

## Current Status

The current implementation has been validated on the provided six-image test set using:

- one setup ROI
- the fixed runtime OCR strategy
- no runtime tuning UI

This is the intended architecture for the project.
