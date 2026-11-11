---
title: Spheroid Core-Shell Metrics
emoji: 🔬
colorFrom: green
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Spheroid Core-Shell Metrics

A minimal web tool for computing core-shell morphometric and radial
intensity-profiling metrics from paired two-channel (GFP / fibroblast)
widefield spheroid images. Not intended for confocal z-stacks — see Scope
below. Companion tool for the Data Descriptor "Fluorescence imaging dataset
of melanoma fibroblast co-spheroids with morphometric and radial profiling
data" (Veniali et al., *Scientific Data*, submitted). Public instance:
<https://lanzafame-lab-spheroid-core-shell-metrics.onrender.com/>. Source code:
<https://github.com/consiglionazionaledellericerche/lanzafame-lab-spheroid-core-shell-metrics>.
Laboratory web page:
<https://www.igm.cnr.it/istituto-2/organizzazione/personale/lanzafame-manuela/>.

`app.py` provides a browser-based upload form; `pipeline_core.py` contains
the segmentation and metrics functions of
`scripts/01_extract_spheroid_metrics.py` in the dataset's published
image-analysis pipeline
(<https://github.com/consiglionazionaledellericerche/lanzafame-lab-melanoma-fibroblast-spheroids>,
version 1.1.0, <https://doi.org/10.5281/zenodo.24144822>). This tool introduces
no new analysis logic, only a browser interface, so results match the
paper's own analysis exactly. The only difference is in filename parsing:
the legacy channel tokens TRITC and CY5 are also accepted for the
fibroblast channel.

Outputs: QC overlay, radial profile plot, the per-spheroid metrics of
`spheroid_metrics_*.csv` plus the core-shell metrics of
`core_shell_metrics_*.csv` (μ_MEL, μ_FB, Δμ, core r ≤ 0.30 / shell r ≥ 0.70
fractions, core enrichment), downloadable as standard comma-separated,
period-decimal CSV files (same format as the deposited data). When a manual
threshold is entered, the metrics are computed on the manually thresholded
mask shown in the QC image.

## Scope

Designed for two-channel widefield fluorescence images (not confocal
z-stacks). One spheroid per submission: a melanoma-channel image (GFP) and a fibroblast-channel
image (PKH26; legacy TRITC/CY5 filename tokens accepted), already isolated in the frame, same field of view, same
pixel dimensions. Pixel size and segmentation parameters (minimum object
size, core/shell radius cutoff, number of radial bins, and an optional
manual segmentation threshold) are set by the user and must be verified via
the QC overlay before trusting the output. This tool does not automatically
validate segmentation quality on images acquired under conditions different
from the original dataset (different microscope, exposure, or contrast) —
that kind of automatic quality control is left for a possible future,
more general version of this tool.

## Run locally

```bash
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:5000
```

## Configuration

Optional environment variables (Render → Environment):

- `LAB_URL`: laboratory web page, linked from the lab logo (or listed in the
  footer when there is no logo).
- `INSTITUTE_URL`: institutional (CNR-IGM) page, linked in the footer;
  `INSTITUTE_NAME` overrides the link text.
- `CODE_URL`: GitHub/Zenodo link of the code, listed in the footer.

Optional logos: `static/logo.png` (lab, top of the page) and
`static/institute_logo.png` (footer); `.svg`/`.jpg` also accepted.

## Deploy

Any Docker-capable host works; the `Dockerfile` reads the `$PORT`
environment variable if set, and defaults to port 7860.

### Render.com (free tier, no credit card required)

1. Push these files to a GitHub repository.
2. On [render.com](https://render.com): New → Web Service → connect the
   repository. Render auto-detects the `Dockerfile` (Environment = Docker).
   Instance type: Free.
3. Deploy — Render assigns a public URL such as
   `https://<service-name>.onrender.com`.

Note: free Render web services spin down after 15 minutes of inactivity;
the first request after a pause takes about a minute to wake the service
back up.

### Hugging Face Spaces (Docker SDK)

Create a Space with SDK = Docker and upload these files (or `git push` to
the Space's repo). The `sdk: docker` / `app_port: 7860` fields in this
README's YAML header are read directly by Hugging Face as the Space's
configuration.

Note (as of 2026-08): Hugging Face has restricted free "CPU Basic" hardware
for Docker/Gradio Spaces on some accounts, requiring a paid plan — check
current pricing before relying on this path. Render.com above was verified
free with no credit card at the time of writing.

## Validation

The core computation (`pipeline_core.compute_metrics_and_profiles`) was
checked against the dataset's own processed CSV: running it on
`20250728_D1-D7/d1_c3pv_gfp_001.tif` / `d1_c3pv_PKH26_001.tif` through the
web form reproduces the corresponding rows of
`spheroid_metrics_SKMEL28_control_strategyC.csv` and
`core_shell_metrics_SKMEL28_control_strategyC.csv`
(`source_experiment=20250728_C`, `file_base=D1_C3PV_001`) exactly
(re-checked 2026-10-28). The full upload → segmentation → QC
overlay → metrics table → CSV download flow was tested end-to-end in a
real browser, including an error path (mismatched image dimensions).

## Requirements

Flask, NumPy, pandas, OpenCV (headless), scikit-image, matplotlib,
gunicorn — see `requirements.txt`.

## Licence and citation

Code: MIT License (see `LICENSE`), as for the dataset's processing pipeline.
The CNR and laboratory logos and any third-party content are excluded from
the licence (see `LICENSE-DATA.md`).

Copyright © 2026 Institute of Molecular Genetics "Luigi Luca Cavalli-Sforza" -
Consiglio Nazionale delle Ricerche, Italy.

If you use this tool, please cite the Data Descriptor (see `CITATION.cff`).
