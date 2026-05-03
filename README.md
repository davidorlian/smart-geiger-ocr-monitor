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

- [setup.py](setup.py)  
  Interactive setup. Handles ROI selection and saves configuration.

- [run.py](run.py)  
  Monitoring runtime. Uses the saved ROI and performs periodic measurements.

- [test_ocr.py](test_ocr.py)  
  OCR development tool. It supports both automatic cropped-image batch testing and interactive calibration.

- [benchmark_ocr.py](benchmark_ocr.py)  
  OCR regression benchmark script for local datasets.

- `config.json`  
  Saved configuration used by `run.py`.

- [config.example.json](config.example.json)  
  Example configuration for version control. Copy or regenerate a real `config.json` locally with `setup.py`.

- [test_v1](test_v1)  
  Legacy first test set. Kept for historical comparison and regression checks.

- [test_v2](test_v2)  
  Current OCR test set. Images are cropped around the numeric LCD reading area, and filenames encode the expected value.

- `logs/`  
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

- `PC_TEST_MODE = True` in [setup.py](setup.py)
- `"PC_TEST_MODE": true` in `config.json`

### Raspberry Pi Mode

Used for the real deployed camera setup.

Current behavior:

- `setup.py` captures a live image from the Pi camera
- the user selects the LCD once
- `run.py` repeatedly captures live frames and reads the saved ROI

For deployment:

- set `PC_TEST_MODE = False` in [setup.py](setup.py)
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

into `config.json`.

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

By default, this runs batch cropped test mode on `test_v2`.

Batch mode expects images that are already cropped around the numeric LCD reading area. It does not open windows, ask for an ROI, or wait for key presses. Expected values are derived from filenames when possible, for example:

```text
ram_gene_25p70.png -> 25.70
```

Batch output includes:

- expected value
- actual OCR result
- OK/BAD status
- confidence
- winner source, crop, variant, and stage
- final score, vote count, and penalties
- top competing candidates for failures
- simple failure classification

Run interactive calibration mode explicitly when needed:

```powershell
python3 test_ocr.py --mode 0
```

Interactive mode is still useful for OCR development and investigation.

It provides:

- manual ROI selection per image
- debug windows
- calibration sliders
- OCR comparison across sample images
- visibility into the internal winning crop and preprocessing variant

This tool is not part of the normal user workflow.

## OCR Test Sets

There are currently two test dataset generations:

- `test_v1/` is the legacy first test set. It is useful for history and checking that old assumptions are not silently forgotten.
- `test_v2/` is the current primary OCR dataset. It contains decimal and integer readings, cropped around the numeric LCD reading area.

The current default OCR testing workflow is:

```powershell
python3 test_ocr.py --image-dir test_v2
```

For regression benchmarking with the same OCR engine:

```powershell
python3 benchmark_ocr.py --image-dir test_v2 --engine robust
```

The runtime still uses the saved camera/LCD ROI from `config.json`. The cropped-image batch workflow is for OCR validation and tuning, not for replacing setup on the mounted camera.

## Configuration

Example fields in `config.json`:

- `warning_threshold`
- `critical_threshold`
- `measurement_interval_seconds`
- `email_settings`
- `log_directory`
- `PC_TEST_MODE`
- `initial_image_for_roi`
- `roi_coordinates`

For upload:

- commit [config.example.json](config.example.json)
- keep your local `config.json` out of git

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

The project has moved from the original small integer-focused test set to a larger decimal-focused OCR validation set.

Current OCR validation status:

- `test_v1/` remains as the legacy dataset.
- `test_v2/` is the primary cropped-reading dataset.
- `test_ocr.py` batch mode validates cropped reading images automatically.
- Interactive calibration mode remains available for investigating ROI, preprocessing, and candidate-selection behavior.
- `run.py` uses the same robust OCR path for runtime extraction.

This is the intended architecture for the project.

Large refactoring is intentionally deferred until after saving this working checkpoint. The next likely cleanup is to split OCR engine, scoring/reporting, and interactive UI code into smaller modules without changing behavior.
