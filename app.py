#!/usr/bin/env python3
"""
Spheroid Core-Shell Metrics — minimal web tool
================================================

Companion tool for the Data Descriptor:
Veniali et al. (2026) Fluorescence imaging dataset of melanoma fibroblast
co-spheroids with morphometric and radial profiling data. Scientific Data
(submitted).

This is a browser interface that uses the exact same segmentation and
metrics functions as the dataset's published image-analysis pipeline (see
pipeline_core.py), so results match the paper's own analysis exactly.

Scope (deliberate): designed for two-channel widefield fluorescence images
(not confocal z-stacks). ONE spheroid per submission (one melanoma-channel image (GFP) + one
fibroblast-channel image (PKH26), already isolated in the frame, same field of
view, same pixel dimensions). Pixel size and segmentation parameters are
set by the user via the form and must be verified via the QC overlay before
trusting the output — this tool does not attempt to auto-validate
segmentation quality on images acquired under conditions different from
the original dataset.

Run locally:
    pip install -r requirements.txt
    python app.py
    # open http://127.0.0.1:5000
"""

import base64
import html as html_mod
import io
import os
import tempfile
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, request, Response

import pipeline_core as pc

app = Flask(__name__)

PAPER_TITLE = ("Fluorescence imaging dataset of melanoma fibroblast co-spheroids "
               "with morphometric and radial profiling data")
# Links shown on the page; set via environment variables on the host
# (e.g. Render → Environment) or edit the defaults here.
LAB_URL = os.environ.get("LAB_URL", "")
CODE_URL = os.environ.get("CODE_URL", "")
INSTITUTE_URL = os.environ.get("INSTITUTE_URL", "")
INSTITUTE_NAME = os.environ.get("INSTITUTE_NAME",
                                "CNR - Istituto di Genetica Molecolare “Luigi Luca Cavalli-Sforza”")

# Logos: drop static/logo.png (lab, top of page) and static/institute_logo.png
# (footer), or .svg/.jpg, next to app.py; no file, no logo.
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def find_static(stem):
    return next((f"{stem}.{ext}" for ext in ("svg", "png", "jpg", "jpeg")
                 if os.path.isfile(os.path.join(STATIC_DIR, f"{stem}.{ext}"))), None)


LOGO_FILE = find_static("logo")
INSTITUTE_LOGO_FILE = find_static("institute_logo")

# Core/shell compartments of 02_compute_core_shell_metrics.py
CS_CORE_THRESHOLD = 0.30
CS_SHELL_THRESHOLD = 0.70
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB per upload, generous for two 16-bit TIFFs

# ----------------------------------------------------------------------
# Defaults mirror the dataset's own processing parameters
# (see 01_extract_spheroid_metrics.py --help)
# ----------------------------------------------------------------------
PIXEL_SIZE_CHOICES = [
    ("1.62", "1.62 µm/px — Olympus IX71, 10x (this dataset's imaging setup)"),
    ("custom", "Other (enter manually)"),
]
MIN_OBJECT_SIZE_CHOICES = ["50", "100", "200", "400", "800"]
CORE_CUTOFF_CHOICES = ["0.3", "0.4", "0.5", "0.6", "0.7"]
N_BINS_CHOICES = ["10", "20", "30"]

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Spheroid Core-Shell Metrics</title>
<style>
  body { font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }
  h1 { font-size: 1.4rem; }
  header { display: flex; align-items: center; gap: 1.5rem; flex-wrap: wrap; }
  header img.logo { display: block; width: 240px; max-width: 100%; height: auto; }
  /* CNR visual identity manual: clear space around the logo of at least half its height (64px / 2) */
  .institute { display: flex; align-items: center; gap: 0.8rem; margin-top: 32px; padding-bottom: 32px; font-size: 0.85rem; }
  .institute img.inst-logo { display: block; height: 64px; width: auto; max-width: 100%; object-fit: contain; }
  h2 { font-size: 1.1rem; margin-top: 2rem; }
  .note { background: #f4f4f4; border-left: 3px solid #999; padding: 0.6rem 1rem; font-size: 0.9rem; color: #444; }
  .warn { background: #fff3f0; border-left: 3px solid #c0392b; padding: 0.6rem 1rem; font-size: 0.9rem; color: #7a2318; }
  form { border: 1px solid #ddd; border-radius: 8px; padding: 1.2rem; margin-top: 1rem; }
  label { display: block; margin-top: 0.8rem; font-weight: 600; font-size: 0.92rem; }
  select, input[type=number], input[type=file] { margin-top: 0.25rem; padding: 0.35rem; font-size: 0.92rem; width: 260px; box-sizing: border-box; }
  .row { display: flex; gap: 2rem; flex-wrap: wrap; }
  .col { flex: 1; min-width: 260px; }
  button { margin-top: 1.4rem; padding: 0.6rem 1.4rem; font-size: 1rem; background: #2c6e49; color: white; border: none; border-radius: 6px; cursor: pointer; }
  button:hover { background: #245a3b; }
  table { border-collapse: collapse; margin-top: 0.6rem; font-size: 0.88rem; }
  td, th { border: 1px solid #ddd; padding: 0.35rem 0.6rem; text-align: left; }
  th { background: #f0f0f0; }
  img.qc { max-width: 100%; border: 1px solid #ccc; border-radius: 4px; margin-top: 0.5rem; }
  .error { background: #fdecea; border: 1px solid #c0392b; padding: 1rem; border-radius: 6px; color: #7a2318; white-space: pre-wrap; font-size: 0.9rem; }
  footer { margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid #eee; font-size: 0.8rem; color: #888; }
</style>
</head>
<body>
<header>__LOGO__<h1>Spheroid Core-Shell Metrics</h1></header>
<p class="note">Companion tool for the Data Descriptor "__PAPER_TITLE__" (Veniali et al., <em>Scientific Data</em>). It uses the <em>exact same</em> segmentation and metrics functions as the deposited image-analysis pipeline — no new analysis logic is introduced here. For two-channel widefield fluorescence images, one spheroid per submission.</p>
<p class="warn"><strong>Check the result before trusting it:</strong> inspect the segmentation mask in the QC image below. If it does not correctly outline the spheroid, adjust the parameters and re-run. This tool does not automatically validate segmentation quality on images acquired under conditions different from the original dataset.</p>

<form method="post" enctype="multipart/form-data">
  <div class="row">
    <div class="col">
      <label>Melanoma channel image (GFP) — TIFF</label>
      <input type="file" name="gfp_file" accept=".tif,.tiff" required>
      <label>Fibroblast channel image (PKH26) — TIFF</label>
      <input type="file" name="fib_file" accept=".tif,.tiff" required>
    </div>
    <div class="col">
      <label>Pixel size</label>
      <select name="pixel_size_choice">
        __PIXEL_SIZE_OPTIONS__
      </select>
      <input type="number" step="0.001" name="pixel_size_custom" placeholder="µm/px, if 'Other'">

      <label>Minimum object size (px, post-Otsu)</label>
      <select name="min_object_size">__MIN_OBJ_OPTIONS__</select>

      <label>Core/shell radius cutoff (normalised r)</label>
      <select name="core_r_cutoff">__CORE_CUTOFF_OPTIONS__</select>

      <label>Number of radial bins</label>
      <select name="n_bins">__N_BINS_OPTIONS__</select>

      <label>Manual segmentation threshold (optional)</label>
      <input type="number" step="0.0001" name="manual_threshold" placeholder="leave blank for automatic Otsu">
    </div>
  </div>
  <button type="submit">Analyze</button>
</form>

__RESULT_HTML__

<footer>Veniali et al., "__PAPER_TITLE__", <em>Scientific Data</em> (submitted). __LINKS__Source code and dataset: see the Code Availability section of the Data Descriptor.<br>Copyright &copy; 2026 Institute of Molecular Genetics &ldquo;Luigi Luca Cavalli-Sforza&rdquo; - Consiglio Nazionale delle Ricerche, Italy. Source code released under the MIT Licence.__INSTITUTE__</footer>

</body>
</html>
"""


def render_options(choices, selected=None):
    out = []
    for c in choices:
        if isinstance(c, tuple):
            val, label = c
        else:
            val, label = c, c
        sel = " selected" if (selected is not None and val == selected) else ""
        out.append(f'<option value="{val}"{sel}>{label}</option>')
    return "\n".join(out)


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def make_qc_overlay(g_norm, r_norm, mask, core_mask):
    """QC image: RGB composite (G=melanoma, R=fibroblast) with spheroid-mask
    contour and core/shell boundary contour overlaid, for the user to verify
    segmentation quality before trusting the numeric output."""
    rgb = np.zeros((*g_norm.shape, 3), dtype=np.float32)
    rgb[..., 1] = g_norm  # green channel = GFP/melanoma
    rgb[..., 0] = r_norm  # red channel = fibroblast

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(np.clip(rgb, 0, 1))
    ax.contour(mask.astype(float), levels=[0.5], colors="cyan", linewidths=1.5)
    ax.contour(core_mask.astype(float), levels=[0.5], colors="yellow", linewidths=1.2, linestyles="dashed")
    ax.set_title("QC: spheroid outline (cyan) and core/shell boundary (dashed yellow)", fontsize=9)
    ax.axis("off")
    return fig_to_base64(fig)


def make_radial_plot(profiles):
    r_mid = [p["r_mid"] for p in profiles]
    g_int = [p["g_int_mean"] for p in profiles]
    r_int = [p["r_int_mean"] for p in profiles]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(r_mid, g_int, color="green", marker="o", markersize=3, label="GFP (melanoma)")
    ax.plot(r_mid, r_int, color="red", marker="o", markersize=3, label="fibroblast")
    ax.set_xlabel("normalised radius (0 = center, 1 = edge)")
    ax.set_ylabel("mean intensity per bin")
    ax.legend(fontsize=8)
    ax.set_title("Radial intensity profile", fontsize=9)
    fig.tight_layout()
    return fig_to_base64(fig)


def metrics_table_html(metrics):
    order = [
        "area_um2", "radius_um", "circularity", "solidity", "eccentricity", "aspect_ratio",
        "core_compaction", "g_r_mean", "r_r_mean", "delta_r",
        "g_area_frac", "r_area_frac", "g_core_frac", "r_core_frac", "core_bias",
        "g_r_wmean", "r_r_wmean", "delta_r_w", "pos_jaccard",
        "g_mean_core", "g_mean_shell", "r_mean_core", "r_mean_shell",
        "mu_mel", "mu_fb", "delta_mu", "mel_core_frac", "fb_core_frac",
        "mel_shell_frac", "fb_shell_frac", "core_enrichment", "segmentation",
    ]
    rows = []
    for k in order:
        v = metrics.get(k)
        v_str = f"{v:.4f}" if isinstance(v, float) and np.isfinite(v) else str(v)
        rows.append(f"<tr><td>{k}</td><td>{v_str}</td></tr>")
    return "<table><tr><th>Metric</th><th>Value</th></tr>" + "".join(rows) + "</table>"


def core_shell_metrics(profiles):
    """Architecture metrics from the radial profile, same formulas as
    02_compute_core_shell_metrics.py (ring-area-weighted intensity centroids,
    core r <= 0.30, shell r >= 0.70)."""
    r_mid = np.array([p["r_mid"] for p in profiles], dtype=float)
    area = np.array([p["ring_area_pix"] for p in profiles], dtype=float)
    g_w = np.nan_to_num(np.array([p["g_int_mean"] for p in profiles], dtype=float)) * area
    r_w = np.nan_to_num(np.array([p["r_int_mean"] for p in profiles], dtype=float)) * area
    g_tot, r_tot = g_w.sum(), r_w.sum()
    if g_tot <= 0 or r_tot <= 0:
        return {k: np.nan for k in ("mu_mel", "mu_fb", "delta_mu", "mel_core_frac", "fb_core_frac",
                                    "mel_shell_frac", "fb_shell_frac", "core_enrichment")}
    core, shell = r_mid <= CS_CORE_THRESHOLD, r_mid >= CS_SHELL_THRESHOLD
    out = {
        "mu_mel": float((r_mid * g_w).sum() / g_tot),
        "mu_fb": float((r_mid * r_w).sum() / r_tot),
        "mel_core_frac": float(g_w[core].sum() / g_tot),
        "fb_core_frac": float(r_w[core].sum() / r_tot),
        "mel_shell_frac": float(g_w[shell].sum() / g_tot),
        "fb_shell_frac": float(r_w[shell].sum() / r_tot),
    }
    out["delta_mu"] = out["mu_mel"] - out["mu_fb"]
    out["core_enrichment"] = out["fb_core_frac"] - out["mel_core_frac"]
    return out


def rows_to_csv(rows):
    """Standard comma-separated, period-decimal CSV (same format as the deposited files)."""
    import csv as csv_mod
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv_mod.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


@app.route("/", methods=["GET", "POST"])
def index():
    result_html = ""
    if request.method == "POST":
        try:
            gfp_file = request.files["gfp_file"]
            fib_file = request.files["fib_file"]

            pixel_choice = request.form.get("pixel_size_choice", "1.62")
            if pixel_choice == "custom":
                pixel_size = float(request.form.get("pixel_size_custom") or 0)
                if pixel_size <= 0:
                    raise ValueError("Invalid custom pixel size.")
            else:
                pixel_size = float(pixel_choice)

            min_object_size = int(request.form.get("min_object_size", "200"))
            core_r_cutoff = float(request.form.get("core_r_cutoff", "0.5"))
            n_bins = int(request.form.get("n_bins", "20"))
            manual_threshold_raw = request.form.get("manual_threshold", "").strip()
            manual_threshold = float(manual_threshold_raw) if manual_threshold_raw else None

            with tempfile.TemporaryDirectory() as tmpdir:
                gfp_path = os.path.join(tmpdir, "gfp.tif")
                fib_path = os.path.join(tmpdir, "fib.tif")
                gfp_file.save(gfp_path)
                fib_file.save(fib_path)

                img_g = pc.load_gray(gfp_path)
                img_r = pc.load_gray(fib_path)

                if img_g.shape != img_r.shape:
                    raise ValueError(
                        f"The two images have different dimensions ({img_g.shape} vs {img_r.shape}) — "
                        "they must be the same field of view, the same spheroid."
                    )

                g_norm = pc.normalize_percentile(img_g)
                r_norm = pc.normalize_percentile(img_r)
                sum_img = g_norm + r_norm

                if manual_threshold is not None:
                    # Same morphological pipeline as segment_spheroid(), but with a
                    # user-supplied threshold instead of the automatic Otsu value —
                    # exposed for the case where QC shows Otsu picked a bad threshold.
                    from skimage import morphology, measure, segmentation as skseg
                    mask = sum_img >= manual_threshold
                    mask = morphology.remove_small_objects(mask, min_size=min_object_size)
                    mask = morphology.binary_opening(mask, morphology.disk(3))
                    mask = morphology.binary_closing(mask, morphology.disk(3))
                    mask = skseg.clear_border(mask)
                    lab = measure.label(mask)
                    if lab.max() > 0:
                        counts = np.bincount(lab.ravel())
                        counts[0] = 0
                        mask = lab == counts.argmax()
                    else:
                        mask = np.zeros_like(mask, dtype=bool)
                else:
                    mask = pc.segment_spheroid(sum_img, min_object_size)

                if not np.any(mask):
                    raise ValueError(
                        "No spheroid could be segmented with these parameters. Try lowering "
                        "'minimum object size' or setting a lower manual threshold."
                    )

                metrics, profiles = pc.compute_metrics_and_profiles(
                    base_id="web_upload",
                    sample="N/A",
                    timepoint="N/A",
                    gfp_img=img_g,
                    tritc_img=img_r,
                    pixel_size_um=pixel_size,
                    core_r_cutoff=core_r_cutoff,
                    n_radial_bins=n_bins,
                    min_object_size=min_object_size,
                    mask=mask,  # the mask shown in the QC image (Otsu or manual threshold)
                )

                if metrics is None:
                    raise ValueError("Metrics computation failed with these parameters — check the segmentation.")

                # recompute mask/core_mask exactly as compute_metrics_and_profiles does internally,
                # only for rendering the QC overlay here
                coords = np.column_stack(np.nonzero(mask))
                centroid = coords.mean(axis=0)
                dists = np.sqrt(((coords - centroid) ** 2).sum(axis=1))
                area_pix = int(mask.sum())
                radius_pix = float(np.sqrt(area_pix / np.pi))
                r_norm_map = np.zeros_like(sum_img, dtype=np.float32)
                if radius_pix > 0:
                    r_norm_map[mask] = dists / radius_pix
                r_norm_map = np.clip(r_norm_map, 0, 1.0)
                core_mask = (r_norm_map < core_r_cutoff) & mask

                qc_b64 = make_qc_overlay(g_norm, r_norm, mask, core_mask)
                radial_b64 = make_radial_plot(profiles)
                cs = core_shell_metrics(profiles)
                metrics.update(cs)
                metrics["segmentation"] = (f"manual threshold {manual_threshold}"
                                           if manual_threshold is not None else "Otsu (automatic)")
                table_html = metrics_table_html(metrics)
                csv_b64 = base64.b64encode(rows_to_csv(profiles).encode("utf-8")).decode("ascii")
                met_b64 = base64.b64encode(rows_to_csv([metrics]).encode("utf-8")).decode("ascii")

                result_html = f"""
                <h2>QC result</h2>
                <img class="qc" src="data:image/png;base64,{qc_b64}">
                <h2>Radial profile</h2>
                <img class="qc" src="data:image/png;base64,{radial_b64}">
                <h2>Per-spheroid metrics</h2>
                {table_html}
                <p><a download="spheroid_metrics.csv" href="data:text/csv;base64,{met_b64}">Download per-spheroid metrics (CSV)</a> ·
                <a download="radial_profile.csv" href="data:text/csv;base64,{csv_b64}">Download radial profile (CSV)</a></p>
                <p class="note">Core-shell metrics (mu_mel … core_enrichment) use the definitions of 02_compute_core_shell_metrics.py
                (core r ≤ 0.30, shell r ≥ 0.70) and do not depend on the core/shell cutoff selected above, which applies only to
                core_compaction, g/r_core_frac, core_bias and the mean core/shell intensities.</p>
                """
        except Exception as e:
            # Log the full traceback server-side for debugging, but show only a
            # clean message to the (potentially anonymous, public) user.
            traceback.print_exc()
            result_html = f'<div class="error">Error: {html_mod.escape(str(e))}</div>'

    links = []
    # With a logo the lab link lives on the logo; otherwise list it in the footer
    if LAB_URL and not LOGO_FILE:
        links.append(f'Laboratory: <a href="{html_mod.escape(LAB_URL)}">{html_mod.escape(LAB_URL)}</a>. ')
    if CODE_URL:
        links.append(f'Code: <a href="{html_mod.escape(CODE_URL)}">{html_mod.escape(CODE_URL)}</a>. ')
    logo = ""
    if LOGO_FILE:
        logo = f'<img class="logo" src="/static/{LOGO_FILE}" alt="Lanzafame Lab logo">'
        if LAB_URL:
            logo = f'<a href="{html_mod.escape(LAB_URL)}" title="Lanzafame Lab">{logo}</a>'
    # The institute logo already carries the institute name: with a logo the
    # link lives on the logo, otherwise the name is listed as text
    institute = ""
    if INSTITUTE_LOGO_FILE:
        institute = (f'<img class="inst-logo" src="/static/{INSTITUTE_LOGO_FILE}" '
                     f'alt="{html_mod.escape(INSTITUTE_NAME)}">')
        if INSTITUTE_URL:
            institute = f'<a href="{html_mod.escape(INSTITUTE_URL)}">{institute}</a>'
    elif INSTITUTE_URL:
        institute = f'<a href="{html_mod.escape(INSTITUTE_URL)}">{html_mod.escape(INSTITUTE_NAME)}</a>'
    if institute:
        institute = f'<div class="institute">{institute}</div>'
    html = PAGE.replace("__PAPER_TITLE__", PAPER_TITLE).replace("__LINKS__", "".join(links))
    html = html.replace("__LOGO__", logo).replace("__INSTITUTE__", institute)
    html = html.replace("__PIXEL_SIZE_OPTIONS__", render_options(PIXEL_SIZE_CHOICES, selected="1.62"))
    html = html.replace("__MIN_OBJ_OPTIONS__", render_options(MIN_OBJECT_SIZE_CHOICES, selected="200"))
    html = html.replace("__CORE_CUTOFF_OPTIONS__", render_options(CORE_CUTOFF_CHOICES, selected="0.5"))
    html = html.replace("__N_BINS_OPTIONS__", render_options(N_BINS_CHOICES, selected="20"))
    html = html.replace("__RESULT_HTML__", result_html)
    return Response(html, mimetype="text/html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
