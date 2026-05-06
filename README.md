# Smart Geiger Counter Interface

This project reads the numeric LCD display of a handheld Geiger counter with a camera and OCR, then logs the value and raises alerts when thresholds are crossed. It is designed as a non-invasive monitor: the approved meter remains the measurement source, and the software only reads what the meter already displays.

The normal flow is:

1. Run setup once.
2. Draw a rectangle around the numeric LCD window.
3. Confirm the detected number.
4. Run the monitor.

The runtime path is headless. It does not open OCR debug windows.

## Quick Start

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install Tesseract OCR, then run setup:

```powershell
python setup.py
```

After setup writes `config.json`, start monitoring:

```powershell
python run.py
```

## Modes

### PC Test Mode

Used for development on a PC with saved images.

Current behavior:

- `setup.py` loads a predefined full-camera test image from `test_v2`.
- `run.py` uses the setup image saved in `config.json` by default.
- The measurement interval is intentionally short for fast iteration.
- `PC_TEST_IMAGE_DIR` in [run.py](run.py) can be set to a directory only when you have multiple fixed-camera images with the same framing.

This mode is controlled by:

- `PC_TEST_MODE = True` in [setup.py](setup.py)
- `"PC_TEST_MODE": true` in `config.json`

Do not use a mixed handheld folder as a runtime sequence unless all images share the same camera position and ROI. The saved ROI is a fixed-camera contract.

### Raspberry Pi Mode

Used for the real deployed camera setup.

Current behavior:

- `setup.py` captures a live image from the Pi camera.
- The user selects the numeric LCD window once.
- `run.py` repeatedly captures live frames and reads the saved ROI.

For deployment:

- Set `PC_TEST_MODE = False` in [setup.py](setup.py).
- Rerun `setup.py`.
- Choose real thresholds and a real measurement interval.

## Setup Workflow

Run:

```powershell
python setup.py
```

What happens:

1. An image is loaded or captured.
2. You draw a rectangle around the numeric LCD window.
3. The script runs OCR on that rectangle.
4. It asks for confirmation:

```text
Setup check: detected reading inside ROI = ...
Accept this ROI? [Y]es / [R]edraw / [Q]uit:
```

If the result is wrong, redraw the ROI.

### ROI Selection Rules

The rectangle matters. The OCR is robust to small framing mistakes, but it cannot reliably recover from a rectangle that captures the wrong part of the device.

Use these rules:

- Select the complete gray numeric LCD window, not only the dark digit strokes.
- Include all digits, the decimal point, and a small amount of LCD background around them.
- Do not crop through a digit, the decimal point, or the left/right edge of the LCD window.
- Do not select the whole Geiger counter body, buttons, LEDs, labels, desk, or wall.
- Prefer a slightly loose rectangle around the numeric LCD over a tight rectangle around the digits.
- If the setup readback is wrong, choose `R` and redraw. Do not accept a bad setup readback.

For the current test images, `test_v2/` contains full-camera images for practicing this setup flow, while `test_v2_cropped/` contains already-cropped numeric windows for OCR regression tests.

The full-camera examples in `test_v2/` are useful for setup practice, but they are not guaranteed to be aligned as one fixed-camera runtime sequence. Run setup for the image or camera position you actually use.

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
python run.py
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

## How the OCR Works

The OCR pipeline is built for 7-segment LCD digits.

At a high level:

1. The user selects the numeric LCD window ROI.
2. The script tries several internal reading-band crops inside that ROI.
3. It tries a small set of fixed preprocessing variants.
4. It reads the mask as a 7-segment display first.
5. If that fails, it falls back to Tesseract.
6. The final reading is chosen by weighted voting across the internal candidates.

## OCR Lab Tool

Run:

```powershell
python test_ocr.py
```

By default, this runs batch cropped test mode on `test_v2_cropped`.

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
python test_ocr.py --mode 0
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
- `test_v2/` contains full-camera test images. Use this for setup/runtime tests where an ROI must be selected.
- `test_v2_cropped/` is the current primary cropped OCR regression dataset. It contains decimal and integer readings cropped around the numeric LCD reading area.

The current default OCR testing workflow is:

```powershell
python test_ocr.py --image-dir test_v2_cropped
```

For regression benchmarking with the same OCR engine:

```powershell
python benchmark_ocr.py --image-dir test_v2_cropped --engine robust
```

The runtime still uses the saved camera/LCD ROI from `config.json`. The cropped-image batch workflow is for OCR validation and tuning, not for replacing setup on the mounted camera.

## Repository Structure

- [setup.py](setup.py): Interactive setup. Handles ROI selection and saves configuration.

- [run.py](run.py): Monitoring runtime. Uses the saved ROI and performs periodic measurements.

- [ocr_engine.py](ocr_engine.py): Shared OCR engine used by setup, runtime, batch testing, and benchmarking.

- [test_ocr.py](test_ocr.py): OCR development tool. It supports both automatic cropped-image batch testing and interactive calibration.

- [benchmark_ocr.py](benchmark_ocr.py): OCR regression benchmark script for local datasets.

- `requirements.txt`: Python package dependencies.

- `config.json`: Local saved configuration used by `run.py`.

- [config.example.json](config.example.json): Example configuration for version control. Copy or regenerate a real `config.json` locally with `setup.py`.

- [test_v1](test_v1): Legacy first test set. Kept for historical comparison and regression checks.

- [test_v2](test_v2): Full-camera test images used to exercise setup/runtime ROI selection.

- [test_v2_cropped](test_v2_cropped): Cropped numeric LCD reading images used for OCR batch regression. Filenames encode the expected value.

- `logs/`: Monitoring log output.

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

Python packages:

```powershell
python -m pip install -r requirements.txt
```

System package for OCR:

```bash
sudo apt install tesseract-ocr
```

On Windows, install Tesseract OCR and make sure its executable is available on `PATH`, or keep the default path used by `test_ocr.py`:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

Raspberry Pi deployment:

```bash
sudo apt install python3-picamera2
```
