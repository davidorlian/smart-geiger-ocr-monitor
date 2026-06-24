# Green Multimeter v3 Cleaned Dataset

This is the curated numeric OCR set derived from `test_sets/green_multimeter_v3`.

- Cropped numeric OCR images: 49
- Stable full-camera numeric images: 49
- Manual recovered crops from early full-only captures: 0
- Excluded poor manual crops from early full-only captures: 2
- Excluded non-numeric display: 1 (`--OL-` at `20260527_151213`)
- Excluded log-only rows with no saved images: 6

Cropped filenames end with the expected display value using `p` for the decimal point, matching the project convention. Example: `meter_hold_1p5446.jpg` means expected display text `1.5446`.

The two early full-only frames (`20260527_142511`, `20260527_142834`) came from a different full-camera framing and were too weak after manual crop, so they are excluded from the curated numeric image set.

See `manifest.csv` for per-image ground truth, original log result, and mistake classification. See `alignment.csv` for every timestamp/log/image row from the raw folder.

See `mistake_analysis.md` for the failure pattern analysis and proposed fix direction. The current fast-engine run is saved in `engine_fast_results.csv` and summarized in `engine_fast_summary.json`.
