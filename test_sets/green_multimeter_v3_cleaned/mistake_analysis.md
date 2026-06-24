# Green Multimeter v3 Mistake Analysis

## What Was Cleaned

- Raw inputs: 57 log rows, 52 full images, 50 cropped images.
- Cleaned outputs: 49 numeric cropped OCR tests and 49 stable full-camera numeric images.
- Removed after review: 2 weak manual crops from early full-only frames.
- Excluded from numeric OCR set: 6 log-only rows with no images, 1 non-numeric `--OL-` display, and the 2 weak manual crops.

The stable full-image ROI inferred from all original full/crop pairs is:

```text
x=309, y=167, width=389, height=114
```

The two early full-only frames do not share that framing. They were manually recoverable but too weak for a clean regression set, so they are excluded.

## Original Run Pattern

From the raw log/image alignment:

- 39 numeric readings matched the display value.
- 3 numeric readings had a `1`/`7` digit error in the logged result:
  - `08.671` was logged as `8.611`
  - `0.7023` was logged as `0.1023`
  - `2.4793` was logged as `2.4193`
- 4 numeric readings missed the decimal point:
  - `1.5446` was logged as `15446.0`
  - `1.4938` was logged as `14938.0`
  - `1.5130` was logged as `15130.0`
  - `1.6319` was logged as `16319.0`
- 2 stable cropped numeric displays were logged as unreadable:
  - `16.788`
  - `4.4682`
- 2 early full-only displays were logged unreadable and had no crop, but were too weak after manual crop:
  - `11231`
  - `06.521`
- 1 image had no matching log row:
  - `1.6597` at `20260527_151508`

Most `unusually_high_reading` debug folders were not OCR mistakes. In those cases the logged numeric value matched the display; the runtime only considered them suspicious relative to thresholds.

## Current Engine Check

I ran the current fast OCR path on the cleaned cropped set:

```text
exact: 39/49
numeric_equal: 39/49
```

Current failures group into three patterns:

- Missing decimal after a leading `1`: `1.4938`, `1.5130`, `1.5446`, `1.6319`, `1.6511`, and `1.6597` are read as digit-only strings.
- `7` is confused with `1`: `0.7023 -> 0.1023`, `16.272 -> 16.212`, `16.788 -> 16.188`, and `2.4793 -> 2.4193`.

## Fix Direction

1. Improve decimal-point detection for this green display.
   The repeated failure is a five-digit reading with the decimal after the first digit. The decimal is visible in the image but weak in the OCR mask. Add a secondary green-channel decimal-dot scan near the lower digit baseline, before morphology can erase the dot.

2. Prefer supported decimal candidates over digit-only candidates.
   If a digit-only candidate has the same digits as a weaker decimal candidate, and the decimal candidate has visual dot evidence, rank the decimal candidate higher. This should target `1.xxxx -> 1xxxx` without blindly inserting decimals.

3. Tighten `1` vs `7` slot scoring.
   Several misses come from a weak top segment. Before accepting `1`, measure top-horizontal energy in that digit slot. Promote to `7` only when the top segment is wide enough; avoid treating a small smear as a full `7`.

4. Add an ROI drift guard for full-camera runtime.
   The two early full-only images prove fixed ROI is only valid while camera framing is unchanged. Before OCR, verify the configured ROI contains the expected green digit band; if it does not, log `roi_drift_or_focus` and save evidence.

5. Recognize `--OL-` explicitly.
   `20260527_151213` is not a numeric OCR failure. It should be classified as overload/non-numeric, not folded into numeric regression data.
