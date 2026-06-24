# Raspberry Pi Multimeter OCR

This project reads a numeric multimeter display from a Raspberry Pi camera, converts the display to a number, logs measurements, and optionally sends threshold alerts. The OCR is tuned for seven-segment displays; it is not a general-purpose OCR system.

## Architecture

- `setup.py` captures a setup image, selects and validates the display ROI, and updates `config.json`.
- `run.py` provides the Raspberry Pi runtime and the saved-image test interface.
- `engine.py` contains the active OCR pipeline.
- `config.example.json` is the safe configuration template. `config.json` is local and ignored.

The normal runtime path is `run.py` -> configured ROI -> `engine.read_number_from_roi()` -> log and optional alert.

## Installation

Use Python 3.10 or newer:

```bash
python -m pip install -r requirements.txt
```

The Raspberry Pi also needs the camera tools (`libcamera-hello` and `libcamera-still`). Tesseract must be installed as an operating-system package for the OCR fallback:

```bash
sudo apt install tesseract-ocr
```

## Setup

Create the local configuration, review thresholds and optional email settings, then select the camera ROI:

```bash
cp config.example.json config.json
python setup.py --pi
```

`setup.py` preserves the existing thresholds, interval, email settings, and log directory in `config.json`; it updates the ROI, setup image, and camera resolution. Never commit `config.json` if it contains local paths or credentials.

For setup against the bundled saved image:

```bash
python setup.py --pc-test
```

## Raspberry Pi Runtime

```bash
python run.py
```

Useful one-shot diagnostics:

```bash
python run.py --once --no-alerts --save-debug-images
```

Relative log paths in `config.json` are resolved from the project directory.

## Saved-Image Testing

Run the current cleaned cropped dataset through the same `run.py`/`engine.py` interface:

```bash
python run.py --image-dir test_sets/green_multimeter_v3_cleaned/cropped --image-is-roi --mode fast --no-alerts
```

Single-image example:

```bash
python run.py --image test_sets/green_multimeter_v3_cleaned/cropped/meter_hold_1p2309.jpg --image-is-roi --mode fast --no-alerts
```

`--mode fast` uses the Raspberry Pi-oriented path. `--mode full` runs the broader and slower OCR search. Full-frame saved images require `roi_coordinates` from `config.json`; cropped images should use `--image-is-roi`.

## Tests

```bash
python -m unittest discover -s tests
```

The active tests exercise `engine.py` and `run.py`. Archived hardware and experiment tests are under `legacy/tests_reference/` and are not part of normal test discovery.

## Directory Structure

```text
setup.py
run.py
engine.py
config.example.json
requirements.txt
README.md
tests/                 active unit and runtime tests
test_sets/             curated saved-image fixtures
tools/                 reusable analysis/report helpers
legacy/                old architecture, manual OCR, and experiment references
```

Runtime logs, setup captures, debug evidence, generated reports, local configuration, IDE files, and secrets are excluded by `.gitignore`.

## Current Limitations

- Accuracy depends on stable framing, focus, lighting, and an ROI that includes the complete digits and decimal point.
- The engine is display-specific and can reject unfamiliar layouts or damaged/partially visible digits.
- Camera setup requires a Raspberry Pi desktop/display for interactive ROI selection.
- Email alerts require a local `config.json` containing valid SMTP settings; credentials are not stored in the repository.
