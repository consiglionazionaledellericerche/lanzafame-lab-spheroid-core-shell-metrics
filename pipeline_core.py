#!/usr/bin/env python3
"""
01_extract_spheroid_metrics.py

Image analysis pipeline for heterotypic melanoma-fibroblast spheroids.
Processes paired GFP/PKH26 widefield fluorescence images (.tif) and
extracts per-spheroid morphometrics and cell-type-resolved radial
intensity profiles. (The internal channel key and the processed-data
column prefix remain "TRITC"/"r_" for continuity with the wider project;
deposited filenames use the "PKH26" token, naming the fluorophore imaged.
"TRITC" and "CY5" are accepted as legacy filename aliases.)

Provenance columns (added for the Scientific Data deposit):
    source_experiment   "{date}_{strategy_code}" label, default = input folder name
    timepoint           day label parsed from the filename (actual acquisition day)
    timepoint_nominal   protocol day the acquisition is assigned to (--nominal_map)
    acquisition_time    'acquisition-time-local' read from the MetaMorph TIFF header
    days_post_seeding   calendar days between --seeding_date and acquisition_time

Outputs:
    1) spheroid_metrics.csv  — per-spheroid summary (morphometrics + channel descriptors)
    2) radial_profiles.csv   — normalized concentric-ring intensity distributions

Reference:
    Veniali et al. (2026) Fluorescence imaging dataset of melanoma fibroblast
    co-spheroids with morphometric and radial profiling data. Scientific Data
    (Data Descriptor; submitted).

Usage:
    python 01_extract_spheroid_metrics.py --input_dir /path/to/tifs --output_dir /path/to/output

CSV format: standard comma-separated, period decimal (pandas defaults)
"""

import datetime as _dt

import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2
import tifffile
from skimage import filters, morphology, measure, segmentation


# ======================================================================
# FILENAME PARSING
# ======================================================================

# Scoped to the Control-donor cohort covered by this Data Descriptor
# (C5BO, C3PV, Fb1609). The wider project also includes trichothiodystrophy-
# and xeroderma-pigmentosum-patient-derived donor cohorts (TTD/XP),
# intentionally excluded from this dataset and reserved for a separate,
# future manuscript — do not add them back here until this script's scope
# is deliberately extended to that data.
# Filename token (upper-cased) -> canonical donor identifier.
KNOWN_SAMPLE_IDS = {
    "C5BO": "C5BO", "C3PV": "C3PV", "FB1609": "Fb1609",
}

GROUP_MAP = {
    "C5BO": "Control", "C3PV": "Control", "FB1609": "Control",
}


def extract_timepoint(name: str) -> str:
    """Extract timepoint (e.g. 'D1') from filename."""
    m = re.search(r"[dD]\s*(\d+)", str(name).strip())
    return f"D{m.group(1)}" if m else "NA"


def extract_sample(name: str) -> str:
    """Extract sample identifier from filename."""
    s = str(name).strip().upper()
    for sid, canonical in KNOWN_SAMPLE_IDS.items():
        if sid in s:
            return canonical
    tokens = [t for t in re.split(r"[_\-\s]+", s) if t]
    d_idx = None
    for i, tok in enumerate(tokens):
        if re.match(r"D\d+$", tok):
            d_idx = i
            break
    if d_idx is not None:
        for tok in tokens[d_idx + 1:]:
            if tok in {"GFP", "PKH26", "TRITC", "CY5"}:
                continue
            return tok
    return "UNKNOWN"


def assign_group(sample: str) -> str:
    """Assign experimental group from sample ID."""
    return GROUP_MAP.get(sample.upper(), "Other")


def parse_filename(path: Path) -> Tuple[str, Optional[str], str, str]:
    """Parse filename into (base_id, channel, sample, timepoint)."""
    stem = path.stem
    tokens = re.split(r"[_\s\-]+", stem)
    tokens_clean = [t for t in tokens if t]
    tokens_up = [t.upper() for t in tokens_clean]

    if "GFP" in tokens_up:
        channel = "GFP"
    elif "PKH26" in tokens_up or "TRITC" in tokens_up or "CY5" in tokens_up:
        # "PKH26" (deposit filenames), "TRITC" (original acquisition filenames,
        # matching the Rhodamine/TRITC filter cube used) and "CY5" (a legacy
        # curation alias) all map to the same internal fibroblast-channel key
        # "TRITC"; downstream column names (r_*, etc.) keep that key.
        channel = "TRITC"
    else:
        channel = None

    base_tokens = [t for t in tokens_clean if t.upper() not in ("GFP", "PKH26", "TRITC", "CY5")]
    base_id = "_".join(base_tokens)
    sample = extract_sample(base_id)
    timepoint = extract_timepoint(base_id)
    return base_id, channel, sample, timepoint


# ======================================================================
# IMAGE PROCESSING
# ======================================================================

def read_acquisition_time(path: Path) -> Optional[str]:
    """Return MetaMorph 'acquisition-time-local' (YYYYMMDD HH:MM:SS.fff) from the
    TIFF ImageDescription, or None if absent."""
    try:
        with tifffile.TiffFile(str(path)) as t:
            desc = t.pages[0].tags.get("ImageDescription")
            desc = desc.value if desc is not None else ""
    except Exception:
        return None
    m = re.search(r'<prop id="acquisition-time-local"[^>]*value="([^"]*)"', desc)
    return m.group(1) if (m and m.group(1)) else None


def days_post_seeding(acq_time: Optional[str], seeding_date: Optional[_dt.date]) -> Optional[int]:
    if not acq_time or seeding_date is None:
        return None
    try:
        acq = _dt.datetime.strptime(acq_time[:8], "%Y%m%d").date()
    except ValueError:
        return None
    return (acq - seeding_date).days


def load_gray(path: Path) -> np.ndarray:
    """Load image as float32 grayscale."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Cannot read: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def normalize_percentile(img: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    """Contrast-normalize to [0, 1] using percentile clipping."""
    lo, hi = np.percentile(img, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo), 0, 1)


def segment_spheroid(sum_img: np.ndarray, min_obj_size: int = 200) -> np.ndarray:
    """Segment spheroid mask from summed GFP+TRITC image.

    Strategy: Otsu thresholding on positive pixels, morphological
    refinement, border clearing, and largest-component selection.
    """
    positive = sum_img > 0
    if not np.any(positive):
        return np.zeros_like(sum_img, dtype=bool)

    vals = sum_img[positive]
    try:
        thr = filters.threshold_otsu(vals)
    except ValueError:
        return np.zeros_like(sum_img, dtype=bool)

    mask = sum_img >= thr
    mask = morphology.remove_small_objects(mask, min_size=min_obj_size)
    mask = morphology.binary_opening(mask, morphology.disk(3))
    mask = morphology.binary_closing(mask, morphology.disk(3))
    mask = segmentation.clear_border(mask)

    lab = measure.label(mask)
    if lab.max() == 0:
        return np.zeros_like(mask, dtype=bool)

    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return lab == counts.argmax()


def positive_mask_otsu(norm_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Otsu-threshold within the spheroid mask for a single channel."""
    vals = norm_img[mask]
    vals = vals[vals > 0]
    if vals.size > 10:
        try:
            thr = filters.threshold_otsu(vals)
            return (norm_img >= thr) & mask
        except ValueError:
            return (norm_img > 0) & mask
    return (norm_img > 0) & mask


# ======================================================================
# METRICS & RADIAL PROFILES
# ======================================================================

def compute_metrics_and_profiles(
    base_id: str,
    sample: str,
    timepoint: str,
    gfp_img: np.ndarray,
    tritc_img: np.ndarray,
    pixel_size_um: float = 1.62,
    core_r_cutoff: float = 0.5,
    n_radial_bins: int = 20,
    min_object_size: int = 200,
    mask: Optional[np.ndarray] = None,
) -> Tuple[Optional[Dict], Optional[List[Dict]]]:
    """Compute all metrics and radial profiles for one spheroid.

    Parameters
    ----------
    base_id : str
        Unique spheroid identifier.
    sample : str
        Sample/donor ID.
    timepoint : str
        Timepoint label (e.g. 'D1').
    gfp_img, tritc_img : np.ndarray
        Raw grayscale images for GFP (melanoma) and TRITC (fibroblast).
    pixel_size_um : float
        Pixel size in micrometers.
    core_r_cutoff : float
        Normalized radius cutoff for core region (legacy metric).
    n_radial_bins : int
        Number of concentric rings for radial profiling.
    min_object_size : int
        Minimum object size for segmentation.
    mask : np.ndarray of bool, optional
        Precomputed spheroid mask. If None (default, used for the deposited
        data), the mask is computed with segment_spheroid(); the web tool
        passes a mask here when the user sets a manual threshold.

    Returns
    -------
    metrics : dict or None
        Per-spheroid metrics dictionary.
    profiles : list of dict or None
        Per-bin radial profile entries.
    """
    g_norm = normalize_percentile(gfp_img)
    r_norm = normalize_percentile(tritc_img)
    sum_img = g_norm + r_norm

    if mask is None:
        mask = segment_spheroid(sum_img, min_object_size)
    if not np.any(mask):
        print(f"[WARN] No valid mask for {base_id}")
        return None, None

    # --- Radial coordinate system ---
    coords = np.column_stack(np.nonzero(mask))
    centroid = coords.mean(axis=0)
    dists = np.sqrt(((coords - centroid) ** 2).sum(axis=1))

    area_pix = int(mask.sum())
    radius_pix = float(np.sqrt(area_pix / np.pi))
    radius_um = radius_pix * pixel_size_um
    area_um2 = float(area_pix) * (pixel_size_um ** 2)

    r_norm_map = np.zeros_like(sum_img, dtype=np.float32)
    if radius_pix > 0:
        r_norm_map[mask] = dists / radius_pix
    r_norm_map = np.clip(r_norm_map, 0, 1.0)

    # --- Shape metrics ---
    perim_pix = float(measure.perimeter(mask))
    perim_um = perim_pix * pixel_size_um
    circularity = float(4.0 * np.pi * area_pix / (perim_pix ** 2)) if perim_pix > 0 else np.nan

    props = measure.regionprops(mask.astype(np.uint8))
    if props:
        rp = props[0]
        solidity = float(getattr(rp, "solidity", np.nan))
        eccentricity = float(getattr(rp, "eccentricity", np.nan))
        maj = float(getattr(rp, "major_axis_length", np.nan))
        minu = float(getattr(rp, "minor_axis_length", np.nan))
        aspect_ratio = maj / minu if np.isfinite(minu) and minu > 0 else np.nan
    else:
        solidity = eccentricity = aspect_ratio = np.nan

    # --- Channel-specific masks ---
    g_pos = positive_mask_otsu(g_norm, mask)
    r_pos = positive_mask_otsu(r_norm, mask)

    spheroid_area = float(area_pix)
    g_area_frac = float(g_pos.sum()) / spheroid_area if spheroid_area > 0 else np.nan
    r_area_frac = float(r_pos.sum()) / spheroid_area if spheroid_area > 0 else np.nan

    # --- Radial means (binary masks) ---
    g_r_mean = float(r_norm_map[g_pos].mean()) if g_pos.any() else np.nan
    r_r_mean = float(r_norm_map[r_pos].mean()) if r_pos.any() else np.nan
    delta_r = (g_r_mean - r_r_mean) if np.isfinite(g_r_mean) and np.isfinite(r_r_mean) else np.nan

    # --- Core compaction ---
    core_mask = (r_norm_map < core_r_cutoff) & mask
    total_int = float((g_norm + r_norm)[mask].sum())
    core_int = float((g_norm + r_norm)[core_mask].sum())
    core_compaction = core_int / total_int if total_int > 0 else np.nan

    # --- Extended metrics ---
    g_total = float(g_norm[mask].sum())
    r_total = float(r_norm[mask].sum())
    g_core = float(g_norm[core_mask].sum())
    r_core = float(r_norm[core_mask].sum())

    g_core_frac = g_core / g_total if g_total > 0 else np.nan
    r_core_frac = r_core / r_total if r_total > 0 else np.nan
    core_bias = (g_core_frac - r_core_frac) if np.isfinite(g_core_frac) and np.isfinite(r_core_frac) else np.nan

    # Intensity-weighted radial means
    rvals = r_norm_map[mask]
    gvals = g_norm[mask]
    rvals2 = r_norm[mask]

    g_r_wmean = float((rvals * gvals).sum() / gvals.sum()) if gvals.sum() > 0 else np.nan
    r_r_wmean = float((rvals * rvals2).sum() / rvals2.sum()) if rvals2.sum() > 0 else np.nan
    delta_r_w = (g_r_wmean - r_r_wmean) if np.isfinite(g_r_wmean) and np.isfinite(r_r_wmean) else np.nan

    # Overlap (Jaccard)
    inter = float((g_pos & r_pos).sum())
    union = float((g_pos | r_pos).sum())
    pos_jaccard = inter / union if union > 0 else np.nan

    # Core vs shell intensity per channel
    shell_mask = mask & (~core_mask)
    g_mean_core = float(g_norm[core_mask].mean()) if core_mask.any() else np.nan
    g_mean_shell = float(g_norm[shell_mask].mean()) if shell_mask.any() else np.nan
    r_mean_core = float(r_norm[core_mask].mean()) if core_mask.any() else np.nan
    r_mean_shell = float(r_norm[shell_mask].mean()) if shell_mask.any() else np.nan

    group = assign_group(sample)

    metrics = {
        "file_base": base_id,
        "sample": sample,
        "group": group,
        "timepoint": timepoint,
        "area_pix": area_pix,
        "area_um2": area_um2,
        "perim_pix": perim_pix,
        "perim_um": perim_um,
        "radius_um": radius_um,
        "circularity": circularity,
        "solidity": solidity,
        "eccentricity": eccentricity,
        "aspect_ratio": aspect_ratio,
        "core_compaction": core_compaction,
        "g_r_mean": g_r_mean,
        "r_r_mean": r_r_mean,
        "delta_r": delta_r,
        "g_area_frac": g_area_frac,
        "r_area_frac": r_area_frac,
        "g_core_frac": g_core_frac,
        "r_core_frac": r_core_frac,
        "core_bias": core_bias,
        "g_r_wmean": g_r_wmean,
        "r_r_wmean": r_r_wmean,
        "delta_r_w": delta_r_w,
        "pos_jaccard": pos_jaccard,
        "g_mean_core": g_mean_core,
        "g_mean_shell": g_mean_shell,
        "r_mean_core": r_mean_core,
        "r_mean_shell": r_mean_shell,
    }

    # --- Radial profiles ---
    edges = np.linspace(0.0, 1.0, n_radial_bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])

    profiles: List[Dict] = []
    for i in range(n_radial_bins):
        lo, hi = edges[i], edges[i + 1]
        # r_norm_map is clipped to a max of 1.0 (any pixel farther from the
        # centroid than the equivalent circular radius collapses to exactly
        # 1.0 - this happens for essentially every non-perfectly-circular
        # mask). The final bin's upper edge is therefore inclusive (<=),
        # otherwise every pixel piled up at r_norm_map == 1.0 falls outside
        # all 20 bins and is silently dropped from the radial profile - not
        # just under-counted in the last bin, but absent from the profile
        # entirely. Bins 1-19 keep the exclusive '<' (adjacent-bin boundary
        # hits there are a measure-zero edge case, not a systematic effect).
        is_last_bin = (i == n_radial_bins - 1)
        if is_last_bin:
            ring = mask & (r_norm_map >= lo) & (r_norm_map <= hi)
        else:
            ring = mask & (r_norm_map >= lo) & (r_norm_map < hi)
        ring_n = int(ring.sum())

        if ring_n == 0:
            g_pos_frac = r_pos_frac = np.nan
            g_int_mean = r_int_mean = np.nan
        else:
            g_pos_frac = float(g_pos[ring].sum()) / ring_n
            r_pos_frac = float(r_pos[ring].sum()) / ring_n
            g_int_mean = float(g_norm[ring].mean())
            r_int_mean = float(r_norm[ring].mean())

        profiles.append({
            "file_base": base_id,
            "sample": sample,
            "group": group,
            "timepoint": timepoint,
            "bin": i + 1,
            "r_mid": float(mids[i]),
            "ring_area_pix": ring_n,
            "g_pos_frac": g_pos_frac,
            "r_pos_frac": r_pos_frac,
            "g_int_mean": g_int_mean,
            "r_int_mean": r_int_mean,
        })

    return metrics, profiles


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract spheroid metrics and radial profiles from paired GFP/PKH26 .tif images."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing paired GFP/PKH26 .tif images.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for CSV files (default: same as input_dir).")
    parser.add_argument("--pixel_size", type=float, default=1.62,
                        help="Pixel size in micrometers (default: 1.62).")
    parser.add_argument("--n_bins", type=int, default=20,
                        help="Number of radial bins (default: 20).")
    parser.add_argument("--core_r_cutoff", type=float, default=0.5,
                        help="Normalized radius cutoff for core region (default: 0.5).")
    parser.add_argument("--min_object_size", type=int, default=200,
                        help="Minimum object size in pixels for segmentation (default: 200).")
    parser.add_argument("--source_experiment", type=str, default=None,
                        help="Label written to the source_experiment column "
                             "(default: name of --input_dir).")
    parser.add_argument("--seeding_date", type=str, default=None,
                        help="Seeding date (YYYY-MM-DD) used to compute days_post_seeding "
                             "from the TIFF acquisition timestamp.")
    parser.add_argument("--nominal_map", type=str, default="",
                        help="Comma-separated 'ACTUAL:NOMINAL' day-label pairs, e.g. 'D4:D3,D6:D5'. "
                             "Unlisted labels map to themselves.")
    args = parser.parse_args()

    source_experiment = args.source_experiment or Path(args.input_dir).name
    seeding_date = _dt.date.fromisoformat(args.seeding_date) if args.seeding_date else None
    nominal_map = dict(kv.split(":") for kv in args.nominal_map.split(",") if kv.strip())

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Pixel size: {args.pixel_size} um/px")

    # Pair GFP/PKH26 images (internal key for the fibroblast channel: "TRITC")
    # Pairing is case-insensitive: acquisition filenames mix cases between the
    # two channel images of the same spheroid (e.g. d5_C3PV_GFP_001.tif /
    # d5_c3PV_PKH26_001.tif). The canonical file_base is written upper-case.
    pairs = defaultdict(dict)
    for f in sorted(input_dir.glob("*.tif")):
        base_id, channel, sample, timepoint = parse_filename(f)
        if channel is None:
            continue
        key = base_id.upper()
        if channel in pairs[key]:
            print(f"[WARN] Duplicate {channel} image for {key}: {f.name} ignored "
                  f"(kept {pairs[key][channel]['path'].name})")
            continue
        pairs[key][channel] = {"path": f, "sample": sample, "timepoint": timepoint}

    rows_metrics: List[Dict] = []
    rows_profiles: List[Dict] = []

    for base_id, chans in pairs.items():
        if "GFP" not in chans or "TRITC" not in chans:
            print(f"[WARN] Skipping {base_id}: missing GFP or PKH26 (fibroblast) image")
            continue

        info_g = chans["GFP"]
        info_r = chans["TRITC"]

        if info_g["sample"] != info_r["sample"] or info_g["timepoint"] != info_r["timepoint"]:
            print(f"[WARN] Sample/timepoint mismatch in {base_id}")
            continue

        sample = info_g["sample"]
        timepoint = info_g["timepoint"]

        print(f"Processing {base_id} ({sample}, {timepoint})")
        img_g = load_gray(info_g["path"])
        img_r = load_gray(info_r["path"])

        metrics, prof = compute_metrics_and_profiles(
            base_id, sample, timepoint, img_g, img_r,
            pixel_size_um=args.pixel_size,
            core_r_cutoff=args.core_r_cutoff,
            n_radial_bins=args.n_bins,
            min_object_size=args.min_object_size,
        )
        if metrics is None:
            continue
        acq_time = read_acquisition_time(info_g["path"])
        prov = {
            "source_experiment": source_experiment,
            "timepoint_nominal": nominal_map.get(timepoint, timepoint),
            "acquisition_time": acq_time,
            "days_post_seeding": days_post_seeding(acq_time, seeding_date),
        }

        def _with_prov(row: Dict) -> Dict:
            out = {"source_experiment": prov["source_experiment"]}
            for k, v in row.items():
                out[k] = v
                if k == "timepoint":
                    out["timepoint_nominal"] = prov["timepoint_nominal"]
                    out["acquisition_time"] = prov["acquisition_time"]
                    out["days_post_seeding"] = prov["days_post_seeding"]
            return out

        rows_metrics.append(_with_prov(metrics))
        if prof is not None:
            rows_profiles.extend(_with_prov(r) for r in prof)

    if not rows_metrics:
        print("No spheroids processed. Check filenames and input directory.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "spheroid_metrics.csv"
    profile_path = output_dir / "radial_profiles.csv"

    pd.DataFrame(rows_metrics).to_csv(metrics_path, index=False)
    pd.DataFrame(rows_profiles).to_csv(profile_path, index=False)

    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {profile_path}")
    print(f"Spheroids analyzed: {len(rows_metrics)}")
    print(f"Radial profile rows: {len(rows_profiles)} ({len(rows_metrics)} x {args.n_bins})")


if __name__ == "__main__":
    main()
