# Embedded Meter OCR

This project reads a numeric 7-segment display from a camera image, converts it to a number, logs measurements, and can raise alerts from configured thresholds. The normal deployed path is:

1. Run `setup.py` once to create or update `config.json`.
2. Select the display ROI.
3. Run `run.py` for live capture, OCR, logging, and alerts.

The OCR is tuned for 7-segment display images. Dataset quality and ROI consistency matter: crops should include the full numeric display, decimal point, and a little background without cutting through digits.

## Setup

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run setup:

```powershell
python setup.py
```

`setup.py` captures or loads an image, lets you choose the display ROI, checks the OCR result, and writes runtime settings to `config.json`. Use `config.example.json` as the versioned template; keep local machine-specific `config.json` out of commits.

## Run Normally

Start the runtime:

```powershell
python -u run.py
```

Useful runtime options:

```powershell
python -u run.py --once --no-alerts --save-debug-images
```

`run.py` loads `config.json`, captures a frame, crops the saved ROI, calls `engine.py`, logs the result, and handles alert output. On Raspberry Pi, live capture uses the configured camera resolution.

## Test Saved Images

Use `run.py` as the saved-image system test entry point. Current primary cropped dataset example:

```powershell
python -u run.py --image-dir test_sets/green_multimeter_v3_cleaned/cropped --image-is-roi --mode fast --no-alerts
```

For a single saved image:

```powershell
python -u run.py --image test_sets/green_multimeter_v3_cleaned/cropped/meter_hold_1p2309.jpg --image-is-roi --mode fast --no-alerts
```

Use `--mode full` when you need the full OCR path. Historical Geiger/red LCD datasets remain under `test_sets/` for regression and reference, but the green multimeter cleaned cropped set is the current primary saved-image example.

## Unit Tests

Active unit tests live in `tests/`:

```powershell
python -m unittest discover -s tests
```

## Project Structure

- `setup.py`: setup/configuration/ROI flow.
- `run.py`: live runtime and saved-image system test entry point.
- `engine.py`: active OCR engine with `fast` and `full` modes.
- `config.json`: local runtime configuration.
- `config.example.json`: versioned configuration template.
- `tests/`: active unit tests.
- `test_sets/`: saved image datasets.
- `legacy/`: old/manual/reference OCR and runtime tools.
- `tools/`: helper, inspection, benchmark, and search scripts.
- `logs/`: runtime/debug output, not part of the versioned structure.

## Legacy And Tools

Legacy/manual tools are kept for reference and investigation, not the normal runtime path. Prefer `run.py` for current live runs and saved-image checks.

Helper scripts in `tools/` are for analysis, inspection, benchmarking, and search experiments. They may have narrower assumptions than the runtime.
