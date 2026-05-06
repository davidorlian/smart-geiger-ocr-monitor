from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract

from ocr_engine import (
    Params,
    candidate_source,
    clamp,
    clone_params,
    collect_image_paths,
    crop_lcd_reading_area,
    crop_roi,
    default_test_dir,
    expected_text_from_filename,
    image_roi_key,
    is_valid_final_numeric_text,
    load_profile,
    make_odd,
    ocr_once,
    ocr_text_fast,
    parse_roi_arg,
    preprocess,
    robust_ocr_from_lcd_roi,
    save_profile,
)


def select_roi(image) -> Tuple[int, int, int, int]:
    roi_rect = cv2.selectROI("Select ROI", image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    return tuple(map(int, roi_rect))


def ocr_all_images(
    image_paths: List[Path],
    fallback_roi_rect: Optional[Tuple[int, int, int, int]],
    p: Params,
    image_rois: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
    image_params: Optional[Dict[str, Params]] = None,
    select_missing_rois: bool = False,
) -> None:
    print("\n" + "=" * 70)
    print("OCR ALL TEST IMAGES")
    print("=" * 70)

    for i, image_path in enumerate(image_paths, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"{i:02d}) {image_path.name}: ERROR failed to load image")
            continue

        key = image_roi_key(image_path)
        roi_rect = image_rois.get(key) if image_rois is not None else None

        if roi_rect is None and select_missing_rois:
            print(f"[INFO] Select LCD ROI for {image_path.name}")
            roi_rect = select_roi(image)
            if roi_rect == (0, 0, 0, 0):
                print(f"{i:02d}) {image_path.name}: SKIP ROI selection cancelled")
                continue
            if image_rois is not None:
                image_rois[key] = roi_rect

        if roi_rect is None:
            roi_rect = fallback_roi_rect

        if roi_rect is None:
            print(f"{i:02d}) {image_path.name}: SKIP no ROI selected")
            continue

        current_params = clone_params(image_params[key]) if image_params is not None and key in image_params else clone_params(p)

        roi = crop_roi(image, roi_rect)
        if roi is None:
            print(f"{i:02d}) {image_path.name}: ERROR ROI {roi_rect} is outside this image")
            continue

        txt, conf, raw, debug = robust_ocr_from_lcd_roi(roi, current_params)
        winner_label = debug.get("winner_label", "")
        winner_stage = debug.get("winner_stage", "")
        vote_count = int(debug.get("vote_count", 0))
        winner_suffix = ""
        if winner_label:
            stage_suffix = f" stage={winner_stage}" if winner_stage else ""
            winner_suffix = f" via={winner_label}{stage_suffix} votes={vote_count}"
        print(
            f"{i:02d}) {image_path.name}: roi={roi_rect} clean='{txt}' conf={conf:.1f} "
            f"raw='{raw}'{winner_suffix} params={json.dumps(asdict(current_params))}"
        )

    print("=" * 70 + "\n")


def short_penalty_text(penalties: Any, limit: int = 4) -> str:
    if not penalties:
        return "none"
    if isinstance(penalties, str):
        return penalties
    return ",".join(str(item) for item in list(penalties)[:limit]) or "none"


def candidate_rows_from_debug(debug: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = debug.get("candidate_summaries", []) if isinstance(debug, dict) else []
    return rows if isinstance(rows, list) else []


def classify_batch_failure(
    expected: Optional[str],
    actual: str,
    debug: Dict[str, Any],
) -> str:
    rows = candidate_rows_from_debug(debug)
    if not rows and not actual:
        return "no_candidate"

    if actual and not is_valid_final_numeric_text(actual):
        return "malformed_numeric"

    if expected:
        if any(str(row.get("clean", "")) == expected for row in rows):
            return "correct_candidate_lost"

        if actual and expected.replace(".", "") == actual.replace(".", "") and expected != actual:
            return "decimal_issue"

        if ("." in expected) != ("." in actual):
            return "decimal_issue"

    top = rows[0] if rows else {}
    penalties = [str(item) for item in top.get("penalties", [])]
    families = str(top.get("families", ""))
    artifact_penalty = float(top.get("artifact_penalty", debug.get("artifact_penalty", 0.0) or 0.0))
    structural = float(top.get("structural_quality", debug.get("structural_quality", 0.0) or 0.0))

    if actual and "." in actual and "+" not in families:
        if any("single_family_decimal" in item for item in penalties):
            return "single_family_decimal_won"

    if artifact_penalty >= 0.50 or any("right_border_artifact" in item for item in penalties):
        return "right_edge_artifact"

    if structural and structural < 0.55:
        return "weak_structural_candidate_won"

    if expected and actual:
        return "wrong_digit_classification"

    return "unknown"


def print_candidate_rows(rows: List[Dict[str, Any]], limit: int) -> None:
    for row in rows[:limit]:
        print(
            f"      cand#{int(row.get('rank', 0)):02d} "
            f"clean='{row.get('clean', '')}' raw='{row.get('raw', '')}' "
            f"source={row.get('source', '')} families={row.get('families', '')} "
            f"votes={int(row.get('vote_count', 0))} "
            f"struct={float(row.get('structural_quality', 0.0)):.2f} "
            f"artifact={float(row.get('artifact_penalty', 0.0)):.2f} "
            f"final={float(row.get('final_score', 0.0)):.2f} "
            f"penalties={short_penalty_text(row.get('penalties', []))}"
        )


def run_batch_cropped_mode(
    image_paths: List[Path],
    p: Params,
    image_params: Optional[Dict[str, Params]] = None,
    top_candidates: int = 5,
    show_ok_candidates: bool = False,
) -> None:
    print("\n" + "=" * 90)
    print("OCR BATCH CROPPED TEST MODE")
    print("Input images are treated as already-cropped numeric reading areas.")
    print("=" * 90)

    total = 0
    comparable = 0
    exact = 0
    failures: List[Dict[str, Any]] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[ERR] {image_path.name:<24} failed to load")
            failures.append({"image": image_path.name, "classification": "no_candidate"})
            continue

        expected = expected_text_from_filename(image_path)
        case_params = clone_params(image_params.get(image_roi_key(image_path), p)) if image_params else clone_params(p)
        actual, conf, raw, debug = robust_ocr_from_lcd_roi(image, case_params)
        rows = candidate_rows_from_debug(debug)
        top = rows[0] if rows else {}
        winner_label = str(debug.get("winner_label", "")) if isinstance(debug, dict) else ""
        winner_stage = str(debug.get("winner_stage", "")) if isinstance(debug, dict) else ""
        winner_source = str(top.get("source", candidate_source(raw, winner_label)))
        vote_count = int(debug.get("vote_count", top.get("vote_count", 0)) or 0) if isinstance(debug, dict) else 0
        final_score = float(debug.get("final_score", top.get("final_score", 0.0)) or 0.0) if isinstance(debug, dict) else 0.0
        penalties = debug.get("penalties_applied", top.get("penalties", [])) if isinstance(debug, dict) else []

        total += 1
        ok = expected is not None and actual == expected
        if expected is not None:
            comparable += 1
            exact += int(ok)

        status = "OK " if ok else "BAD" if expected is not None else "UNK"
        fail_class = "" if ok else classify_batch_failure(expected, actual, debug if isinstance(debug, dict) else {})
        expected_text = expected if expected is not None else "?"
        stage_suffix = f" stage={winner_stage}" if winner_stage else ""
        class_suffix = f" class={fail_class}" if fail_class else ""
        print(
            f"[{status}] {image_path.name:<24} expected='{expected_text}' got='{actual}' "
            f"conf={conf:.1f} source={winner_source} winner={winner_label}{stage_suffix} "
            f"votes={vote_count} final={final_score:.2f} "
            f"penalties={short_penalty_text(penalties)}{class_suffix}"
        )

        if not ok:
            failures.append(
                {
                    "image": image_path.name,
                    "expected": expected_text,
                    "actual": actual,
                    "raw": raw,
                    "classification": fail_class,
                    "winner": winner_label,
                    "stage": winner_stage,
                }
            )

        if rows and (show_ok_candidates or not ok):
            print_candidate_rows(rows, top_candidates)

    accuracy = exact / float(max(1, comparable))
    print("\n" + "=" * 90)
    print(f"BATCH SUMMARY: exact={exact}/{comparable} accuracy={accuracy:.3f} images={total}")
    if failures:
        counts = Counter(item["classification"] for item in failures)
        print("Failure classes:")
        for name, count in counts.most_common():
            print(f"  {name}: {count}")
        print("Failures:")
        for item in failures:
            print(
                f"  - {item['image']}: expected='{item.get('expected', '?')}' "
                f"got='{item.get('actual', '')}' class={item['classification']} "
                f"raw='{item.get('raw', '')}' winner={item.get('winner', '')} stage={item.get('stage', '')}"
            )
    print("=" * 90 + "\n")


def sync_trackbars_from_params(p: Params) -> None:
    cv2.setTrackbarPos("Method (0=Otsu,1=Adaptive,2=Dark)", "Calibration Panel", clamp(p.method, 0, 2))
    cv2.setTrackbarPos("Scale (1..6)", "Calibration Panel", clamp(p.scale, 1, 6))
    cv2.setTrackbarPos("Blur (0..2)", "Calibration Panel", clamp(p.blur, 0, 2))
    cv2.setTrackbarPos("Dark Threshold (0..255)", "Calibration Panel", clamp(p.dark_threshold, 0, 255))
    cv2.setTrackbarPos("Adaptive Block", "Calibration Panel", clamp(p.adaptive_block, 3, 151))
    cv2.setTrackbarPos("Adaptive C", "Calibration Panel", clamp(p.adaptive_c, 0, 50))
    cv2.setTrackbarPos("Close Enable (0/1)", "Calibration Panel", clamp(p.close_enable, 0, 1))
    cv2.setTrackbarPos("Close Iter (0..3)", "Calibration Panel", clamp(p.close_iter, 0, 3))
    cv2.setTrackbarPos("Close K (1..31)", "Calibration Panel", clamp(p.close_k, 1, 31))
    cv2.setTrackbarPos("Dilate (0..2)", "Calibration Panel", clamp(p.dilate_iter, 0, 2))
    cv2.setTrackbarPos("Erode (0..2)", "Calibration Panel", clamp(p.erode_iter, 0, 2))
    cv2.setTrackbarPos("Median (0/3)", "Calibration Panel", 0 if p.median == 0 else 1)
    cv2.setTrackbarPos("PSM (0=7,1=8,2=13)", "Calibration Panel", clamp(p.psm_mode, 0, 2))
    cv2.setTrackbarPos("Pad (0..80)", "Calibration Panel", clamp(p.pad, 0, 80))


def grid_search(roi_bgr, regex_pattern: str) -> None:
    """
    Grid search over a reasonable sweep that you can expand easily.
    IMPORTANT: No dot injection / no output fixing.
    Prints all regex hits + params.
    """

    rx = re.compile(regex_pattern)

    # Sweep lists (edit these if you want wider search)
    methods = [0, 1]
    scales = [1, 2, 3, 4, 5, 6]
    blurs = [0, 1, 2]

    close_enables = [0, 1]
    close_iters = [0, 1, 2, 3]
    close_ks = [5, 9, 13, 31]  # you can expand to range(1,32) if you insist (will be slower)

    dilates = [0, 1, 2]
    erodes = [0, 1, 2]
    medians = [0, 3]

    psms = [0, 1, 2]  # 7/8/13
    pads = [0, 10, 20, 30, 40, 60, 80]

    # Adaptive sweeps only matter if method=1
    adaptive_blocks = [31, 51, 91, 115, 151]
    adaptive_cs = [0, 12, 22, 32, 44]

    hits = []  # (conf, txt, params_dict, raw)
    out_counter = Counter()

    # Count total combos (for progress)
    total = 0
    for m in methods:
        for _ in scales:
            for _ in blurs:
                for _ in close_enables:
                    for _ in close_iters:
                        for _ in close_ks:
                            for _ in dilates:
                                for _ in erodes:
                                    for _ in medians:
                                        for _ in psms:
                                            for _ in pads:
                                                if m == 0:
                                                    total += 1
                                                else:
                                                    total += len(adaptive_blocks) * len(adaptive_cs)

    done = 0

    for method in methods:
        for scale in scales:
            for blur in blurs:
                for close_enable in close_enables:
                    for close_iter in close_iters:
                        for close_k in close_ks:
                            for dilate_iter in dilates:
                                for erode_iter in erodes:
                                    for median in medians:
                                        for psm_mode in psms:
                                            for pad in pads:
                                                if method == 0:
                                                    adapt_pairs = [(31, 12)]  # dummy; not used
                                                else:
                                                    adapt_pairs = [(ab, ac) for ab in adaptive_blocks for ac in adaptive_cs]

                                                for ab, ac in adapt_pairs:
                                                    done += 1
                                                    if done % 200 == 0:
                                                        print(f"[GRID] {done}/{total}")

                                                    p = Params()
                                                    p.method = method
                                                    p.scale = scale
                                                    p.blur = blur

                                                    p.close_enable = close_enable
                                                    p.close_iter = close_iter
                                                    p.close_k = close_k

                                                    p.dilate_iter = dilate_iter
                                                    p.erode_iter = erode_iter
                                                    p.median = median

                                                    p.psm_mode = psm_mode
                                                    p.pad = pad

                                                    if method == 1:
                                                        p.adaptive_block = make_odd(clamp(ab, 3, 151))
                                                        p.adaptive_c = clamp(ac, 0, 50)

                                                    stages = preprocess(roi_bgr, p)
                                                    txt = ocr_text_fast(stages["ocr_input"], p)
                                                    out_counter[txt] += 1

                                                    if rx.fullmatch(txt or ""):
                                                        txt2, conf, raw = ocr_once(stages["ocr_input"], p)
                                                        hits.append((conf, txt2, asdict(p), raw))

    hits.sort(key=lambda x: x[0], reverse=True)

    print("\n" + "=" * 90)
    print(f"GRID SEARCH DONE | regex='{regex_pattern}'")
    print(f"Hits: {len(hits)}")
    print("=" * 90)

    if hits:
        print("\nHITS (sorted by confidence):")
        for i, (conf, txt, pdict, raw) in enumerate(hits, 1):
            print(f"{i:02d}) conf={conf:.1f} txt='{txt}' raw='{raw}' params={json.dumps(pdict)}")
    else:
        print("\nNo hits found.")
        print("\nMost common outputs (top 15):")
        for txt, cnt in out_counter.most_common(15):
            print(f"  '{txt}' : {cnt}")

    print("=" * 90 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=("0", "1", "calibration", "batch"),
        default="1",
        help="0/calibration = interactive ROI calibration; 1/batch = automatic cropped-image test (default).",
    )
    ap.add_argument("--image", default=None, help="Optional single image path. Overrides --image-dir.")
    ap.add_argument("--image-dir", default=str(default_test_dir()), help="Directory of test images to load.")
    ap.add_argument("--roi", default=None, help="Optional ROI as x,y,w,h. This should cover the whole LCD.")
    ap.add_argument("--tess", default=r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    ap.add_argument("--profile", default=None, help="Optional JSON profile to load at startup.")
    ap.add_argument("--out-profile", default="ocr_profile.json", help="Where to save profile with W key.")
    ap.add_argument("--regex", default=r"\d+(\.\d+)?", help="Regex for grid search hits (e.g. 0\\.16).")
    ap.add_argument("--top-candidates", type=int, default=5, help="Batch mode: number of competing candidates to print.")
    ap.add_argument("--show-ok-candidates", action="store_true", help="Batch mode: also print candidates for OK cases.")
    roi_mode = ap.add_mutually_exclusive_group()
    roi_mode.add_argument(
        "--roi-each-image",
        dest="roi_each_image",
        action="store_true",
        help="For test photos, select and save a separate LCD ROI for each image.",
    )
    roi_mode.add_argument(
        "--fixed-roi",
        dest="roi_each_image",
        action="store_false",
        help="Use one fixed-camera LCD ROI for all loaded images.",
    )
    ap.set_defaults(roi_each_image=True)
    args = ap.parse_args()

    pytesseract.pytesseract.tesseract_cmd = args.tess

    image_paths = collect_image_paths(args.image, args.image_dir)
    if not image_paths:
        print(f"[ERROR] No image files found. Checked: {args.image or args.image_dir}")
        return

    p = Params()
    batch_image_params: Dict[str, Params] = {}
    if args.profile and Path(args.profile).exists():
        try:
            p, _profile_roi_rect, _image_rois, batch_image_params = load_profile(args.profile)
            if args.mode in ("1", "batch"):
                print(f"[INFO] Loaded batch params from profile: {args.profile}")
                if args.image and image_roi_key(Path(args.image)) in batch_image_params:
                    p = clone_params(batch_image_params[image_roi_key(Path(args.image))])
        except Exception as e:
            if args.mode in ("1", "batch"):
                print(f"[WARN] Failed to load profile params for batch mode: {e}")

    if args.mode in ("1", "batch"):
        run_batch_cropped_mode(
            image_paths,
            p,
            image_params=batch_image_params,
            top_candidates=max(0, int(args.top_candidates)),
            show_ok_candidates=bool(args.show_ok_candidates),
        )
        return

    current_image_index = 0
    image_path = image_paths[current_image_index]
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[ERROR] Failed to load image: {image_path}")
        return

    profile_roi_rect: Optional[Tuple[int, int, int, int]] = None
    image_rois: Dict[str, Tuple[int, int, int, int]] = {}
    image_params: Dict[str, Params] = {}
    if args.profile and Path(args.profile).exists():
        try:
            p, profile_roi_rect, image_rois, image_params = load_profile(args.profile)
            print(f"[INFO] Loaded profile: {args.profile}")
            if image_rois:
                print(f"[INFO] Loaded {len(image_rois)} per-image ROI(s) from profile.")
            if image_params:
                print(f"[INFO] Loaded {len(image_params)} per-image parameter set(s) from profile.")
        except Exception as e:
            print(f"[WARN] Failed to load profile: {e}")
    elif args.profile:
        print(f"[INFO] No profile found yet at: {args.profile}")

    print("\n" + "=" * 70)
    print("STABLE OCR CALIBRATION (SOLID + 7-SEG)")
    print(f"Loaded {len(image_paths)} image(s). Current: {image_path.name}")
    print(f"ROI mode: {'one LCD rectangle per test image' if args.roi_each_image else 'one fixed-camera LCD rectangle'}")
    if args.roi_each_image:
        print("Test photos: select the whole LCD again whenever a new image needs its own rectangle.")
    else:
        print("Setup: focus the fixed camera, then select the whole LCD once.")
    print("Use --fixed-roi only for the real mounted-camera setup.")
    print("Recommended starting preset: Method=2, Scale=3, Blur=0, Dark Threshold=57, Close K=7.")
    print("OCR once/all also checks a few nearby crop bands and threshold values to reduce brittleness.")
    print("The 7-segment reader checks LCD segments first; Tesseract is only used if that fails.")
    print("1 / ENTER / SPACE = OCR once (prints result)")
    print("A = OCR all loaded test images")
    print("G = Grid search on current ROI (prints all hits)")
    print("N / P = Next / previous loaded test image")
    print("R = Reselect ROI | S = Save OCR input image | W = Save profile JSON")
    print("Q / ESC = Quit")
    print("IMPORTANT: Click once on the 'OCR Input' window so it receives keyboard focus.")
    print("=" * 70 + "\n")

    try:
        roi_rect = parse_roi_arg(args.roi)
    except argparse.ArgumentTypeError as e:
        print(f"[ERROR] {e}")
        return

    if args.roi_each_image:
        current_key = image_roi_key(image_path)
        if current_key in image_params:
            p = clone_params(image_params[current_key])
            print(f"[INFO] Using saved params for {image_path.name}: {json.dumps(asdict(p))}")
        if roi_rect is not None:
            image_rois[current_key] = roi_rect
            print(f"[INFO] Using command-line ROI for {image_path.name}: {roi_rect}")
        else:
            roi_rect = image_rois.get(current_key)
            if roi_rect is not None:
                print(f"[INFO] Using saved ROI for {image_path.name}: {roi_rect}")
            else:
                print(f"[INFO] Select LCD ROI for {image_path.name}")
                roi_rect = select_roi(image)
                if roi_rect != (0, 0, 0, 0):
                    image_rois[current_key] = roi_rect
    elif roi_rect is None and profile_roi_rect is not None:
        roi_rect = profile_roi_rect
        print(f"[INFO] Using ROI from profile: {roi_rect}")
    elif roi_rect is not None:
        print(f"[INFO] Using ROI from command line: {roi_rect}")

    if roi_rect is None:
        roi_rect = select_roi(image)

    if roi_rect == (0, 0, 0, 0):
        print("[INFO] ROI selection cancelled.")
        return

    cv2.namedWindow("Calibration Panel", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration Panel", 620, 420)

    cv2.namedWindow("OCR Input", cv2.WINDOW_NORMAL)
    cv2.namedWindow("LCD ROI", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Reading ROI", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Binary", cv2.WINDOW_NORMAL)
    cv2.namedWindow("After Close", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Winner ROI", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Winner Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Winner OCR Input", cv2.WINDOW_NORMAL)

    cv2.createTrackbar("Method (0=Otsu,1=Adaptive,2=Dark)", "Calibration Panel", p.method, 2, lambda _: None)
    cv2.createTrackbar("Scale (1..6)", "Calibration Panel", p.scale, 6, lambda _: None)
    cv2.createTrackbar("Blur (0..2)", "Calibration Panel", p.blur, 2, lambda _: None)
    cv2.createTrackbar("Dark Threshold (0..255)", "Calibration Panel", p.dark_threshold, 255, lambda _: None)

    cv2.createTrackbar("Adaptive Block", "Calibration Panel", p.adaptive_block, 151, lambda _: None)
    cv2.createTrackbar("Adaptive C", "Calibration Panel", p.adaptive_c, 50, lambda _: None)

    cv2.createTrackbar("Close Enable (0/1)", "Calibration Panel", p.close_enable, 1, lambda _: None)
    cv2.createTrackbar("Close Iter (0..3)", "Calibration Panel", p.close_iter, 3, lambda _: None)
    cv2.createTrackbar("Close K (1..31)", "Calibration Panel", p.close_k, 31, lambda _: None)

    cv2.createTrackbar("Dilate (0..2)", "Calibration Panel", p.dilate_iter, 2, lambda _: None)
    cv2.createTrackbar("Erode (0..2)", "Calibration Panel", p.erode_iter, 2, lambda _: None)

    cv2.createTrackbar("Median (0/3)", "Calibration Panel", 0, 1, lambda _: None)
    cv2.createTrackbar("PSM (0=7,1=8,2=13)", "Calibration Panel", p.psm_mode, 2, lambda _: None)
    cv2.createTrackbar("Pad (0..80)", "Calibration Panel", p.pad, 80, lambda _: None)
    sync_trackbars_from_params(p)

    last_stages: Optional[Dict[str, Any]] = None
    last_saved_stages: Optional[Dict[str, Any]] = None
    winner_stage_name = ""

    def make_placeholder(label: str) -> np.ndarray:
        canvas = np.zeros((120, 240), dtype=np.uint8)
        cv2.putText(canvas, label, (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
        return canvas

    def show_winner_debug(
        winner_crop: Optional[np.ndarray],
        winner_mask: Optional[np.ndarray],
        winner_ocr_input: Optional[np.ndarray],
    ) -> None:
        cv2.imshow("Winner ROI", winner_crop if winner_crop is not None else make_placeholder("No winner"))
        cv2.imshow("Winner Mask", winner_mask if winner_mask is not None else make_placeholder("No mask"))
        cv2.imshow(
            "Winner OCR Input",
            winner_ocr_input if winner_ocr_input is not None else make_placeholder("No OCR input"),
        )

    show_winner_debug(None, None, None)

    while True:
        roi = crop_roi(image, roi_rect)
        if roi is None:
            print(f"[ERROR] ROI is outside the current image: {image_path}")
            break

        p.method = cv2.getTrackbarPos("Method (0=Otsu,1=Adaptive,2=Dark)", "Calibration Panel")
        p.scale = clamp(cv2.getTrackbarPos("Scale (1..6)", "Calibration Panel"), 1, 6)
        p.blur = cv2.getTrackbarPos("Blur (0..2)", "Calibration Panel")
        p.dark_threshold = clamp(cv2.getTrackbarPos("Dark Threshold (0..255)", "Calibration Panel"), 0, 255)

        p.adaptive_block = make_odd(clamp(cv2.getTrackbarPos("Adaptive Block", "Calibration Panel"), 3, 151))
        p.adaptive_c = clamp(cv2.getTrackbarPos("Adaptive C", "Calibration Panel"), 0, 50)

        p.close_enable = cv2.getTrackbarPos("Close Enable (0/1)", "Calibration Panel")
        p.close_iter = clamp(cv2.getTrackbarPos("Close Iter (0..3)", "Calibration Panel"), 0, 3)
        p.close_k = clamp(cv2.getTrackbarPos("Close K (1..31)", "Calibration Panel"), 1, 31)

        p.dilate_iter = clamp(cv2.getTrackbarPos("Dilate (0..2)", "Calibration Panel"), 0, 2)
        p.erode_iter = clamp(cv2.getTrackbarPos("Erode (0..2)", "Calibration Panel"), 0, 2)

        med_sel = cv2.getTrackbarPos("Median (0/3)", "Calibration Panel")
        p.median = 0 if med_sel == 0 else 3

        p.psm_mode = clamp(cv2.getTrackbarPos("PSM (0=7,1=8,2=13)", "Calibration Panel"), 0, 2)
        p.pad = clamp(cv2.getTrackbarPos("Pad (0..80)", "Calibration Panel"), 0, 80)

        reading_roi = crop_lcd_reading_area(roi)
        stages = preprocess(reading_roi, p)
        last_stages = stages

        cv2.imshow("LCD ROI", roi)
        cv2.imshow("Reading ROI", reading_roi)
        cv2.imshow("Binary", stages["bin_digits_white"])
        cv2.imshow("After Close", stages["after_close"])
        cv2.imshow("OCR Input", stages["ocr_input"])

        key = cv2.waitKey(10) & 0xFF

        if key in (13, 32, ord("1")):
            txt, conf, raw, debug = robust_ocr_from_lcd_roi(roi, p)
            winner_label = debug.get("winner_label", "")
            vote_count = int(debug.get("vote_count", 0))
            winner_raw = debug.get("winner_raw", raw)
            winner_stage_name = str(debug.get("winner_stage", ""))
            last_saved_stages = debug.get("winner_stages") or last_stages
            winner_crop = debug.get("winner_crop")
            winner_stages = debug.get("winner_stages") or {}
            winner_mask = debug.get("winner_mask")
            winner_ocr_input = winner_stages.get("ocr_input")
            show_winner_debug(winner_crop, winner_mask, winner_ocr_input)
            print(f"[*] Image: {image_path.name}")
            print(f"[*] ROI: {roi_rect}")
            print(f"[*] Params: {json.dumps(asdict(p))}")
            if winner_label:
                stage_suffix = f" stage={winner_stage_name}" if winner_stage_name else ""
                print(f"[*] Winner: {winner_label}{stage_suffix} | votes={vote_count}")
                penalties = debug.get("penalties_applied", [])
                penalty_text = ",".join(penalties[:6]) if penalties else "none"
                print(
                    f"[*] Winner score: final={float(debug.get('final_score', 0.0)):.2f} "
                    f"struct={float(debug.get('structural_quality', 0.0)):.2f} "
                    f"artifact={float(debug.get('artifact_penalty', 0.0)):.2f} "
                    f"penalties={penalty_text}"
                )
            print(f"[>] OCR RAW: '{winner_raw}'")
            print(f"[>] OCR CLEAN: '{txt}' | conf={conf:.1f}")
            candidate_rows = debug.get("candidate_summaries", [])
            if candidate_rows:
                print("[*] Candidate scores:")
                for row in candidate_rows:
                    row_penalties = ",".join(row.get("penalties", [])[:4]) or "none"
                    print(
                        f"    {int(row.get('rank', 0)):02d} clean='{row.get('clean', '')}' "
                        f"raw='{row.get('raw', '')}' source={row.get('source', '')} "
                        f"families={row.get('families', '')} variant={row.get('best_variant', '')} "
                        f"votes={int(row.get('vote_count', 0))} "
                        f"struct={float(row.get('structural_quality', 0.0)):.2f} "
                        f"artifact={float(row.get('artifact_penalty', 0.0)):.2f} "
                        f"final={float(row.get('final_score', 0.0)):.2f} "
                        f"penalties={row_penalties}"
                    )
            print()

        elif key in (ord("a"), ord("A")):
            ocr_all_images(
                image_paths,
                roi_rect,
                p,
                image_rois=image_rois if args.roi_each_image else None,
                image_params=image_params if args.roi_each_image else None,
                select_missing_rois=args.roi_each_image,
            )

        elif key in (ord("g"), ord("G")):
            print(f"[INFO] Running grid search on {image_path.name}... (UI will freeze until done)")
            grid_search(roi, args.regex)
            print("[INFO] Grid search finished.")

        elif key in (ord("r"), ord("R")):
            roi_rect2 = select_roi(image)
            if roi_rect2 != (0, 0, 0, 0):
                roi_rect = roi_rect2
                last_saved_stages = None
                winner_stage_name = ""
                show_winner_debug(None, None, None)
                if args.roi_each_image:
                    image_rois[image_roi_key(image_path)] = roi_rect
                print(f"[INFO] Manual ROI selected: {roi_rect}")

        elif key in (ord("n"), ord("N"), ord("p"), ord("P")):
            step = -1 if key in (ord("p"), ord("P")) else 1
            next_index = (current_image_index + step) % len(image_paths)
            next_path = image_paths[next_index]
            next_image = cv2.imread(str(next_path))
            if next_image is None:
                print(f"[WARN] Failed to load image: {next_path}")
            else:
                next_roi_rect = roi_rect
                if args.roi_each_image:
                    next_key = image_roi_key(next_path)
                    saved_roi = image_rois.get(next_key)
                    if saved_roi is None:
                        print(f"[INFO] Select LCD ROI for {next_path.name}")
                        saved_roi = select_roi(next_image)
                        if saved_roi == (0, 0, 0, 0):
                            print(f"[INFO] Staying on {image_path.name}; ROI selection for {next_path.name} was cancelled.")
                            continue
                        image_rois[next_key] = saved_roi
                    next_roi_rect = saved_roi
                    if next_key in image_params:
                        p = clone_params(image_params[next_key])
                        sync_trackbars_from_params(p)
                        print(f"[INFO] Loaded saved params for {next_path.name}.")

                current_image_index = next_index
                image_path = next_path
                image = next_image
                roi_rect = next_roi_rect
                last_saved_stages = None
                winner_stage_name = ""
                show_winner_debug(None, None, None)
                print(f"[INFO] Current image {current_image_index + 1}/{len(image_paths)}: {image_path.name}")

        elif key in (ord("s"), ord("S")):
            stages_to_save = last_saved_stages or last_stages
            if stages_to_save is not None:
                cv2.imwrite("debug_ocr_input.png", stages_to_save["ocr_input"])
                print("[INFO] Saved: debug_ocr_input.png")

        elif key in (ord("w"), ord("W")):
            if args.roi_each_image:
                image_rois[image_roi_key(image_path)] = roi_rect
                image_params[image_roi_key(image_path)] = clone_params(p)
            save_profile(args.out_profile, p, roi_rect, image_rois=image_rois, image_params=image_params)
            print(f"[INFO] Saved profile: {args.out_profile} with current ROI {roi_rect}")
            if image_rois:
                print(f"[INFO] Saved {len(image_rois)} per-image ROI(s).")
            if image_params:
                print(f"[INFO] Saved {len(image_params)} per-image parameter set(s).")

        elif key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
