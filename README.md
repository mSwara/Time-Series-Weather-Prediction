# Kolkata DBT/WBT — Deep Learning Extension

Deep-learning follow-on to a prior classical-ML internship project that predicted Dry
Bulb Temperature (DBT) and Wet Bulb Temperature (WBT) over Kolkata from ERA5
reanalysis variables. This project reuses the same 300-row dataset but asks a
genuinely different modeling question — see [`reports/`](reports/) for the full
problem statement, JD-driven skill prioritization, and the point-by-point comparison
against the internship.

## Why this isn't "the internship with an LSTM"

The internship trained two **independent single-target** regressors (best: Ridge for
DBT, LightGBM for WBT) on a flat feature table, selected by random 80:20 split.
This project instead:

- Predicts DBT and WBT **jointly** from one multi-task CNN-LSTM (shared backbone,
  two output heads), rather than two separate single-target models.
- Treats the four lag-temperature features `T(i-1)…T(i-4)` as an explicit **ordered
  sequence** fed through convolution + recurrence, instead of four unordered flat
  columns handed to a tree/linear model with no notion of temporal adjacency.
- Uses a **chronological, burst-aware** split instead of a random one (see Data
  Integrity below) — a stricter leakage standard than the internship applied.
- Adds an auxiliary **heat-stress-day classification** head (where SMOTE/ADASYN are
  legitimately applicable), rather than forcing class-imbalance techniques onto a
  continuous regression target.

## Data integrity findings that shape the design

Both established during exploratory analysis, before any modeling:

1. **The 300 rows are not a continuous daily series.** Sorting by `Date` reveals 71
   disjoint "bursts" of consecutive calendar days (median length 3–4, max 10),
   spanning 1980–2020. Long gaps (sometimes >1 year) separate bursts. This means a
   naive sliding window across the full sorted table would silently splice together
   physically unrelated days.
2. **`T(i-1)…T(i-4)` are already leakage-checked lag features**, verified to equal
   the DBT of the previous row within the same burst (0 mismatches across 299
   burst-internal transitions). They are the only genuine temporal signal available
   per row — the raw table does not carry each lag day's full meteorological vector,
   only its temperature. This constrains what a "sequence" can honestly represent
   (see `reports/phase2_design.md`).
3. Consequently: any train/val/test split, and any windowing scheme, must cut only at
   **burst boundaries**, never mid-burst — otherwise a lag feature in one split would
   be derived from a target value sitting in another split.

## Project layout

```
data/raw/            Original dataset, untouched (FINAL_DATASET_allpara.xlsx)
data/processed/       Deterministic pipeline outputs (git-ignored — regenerate from notebooks)
notebooks/            Numbered, runnable notebooks — the reproducibility trail
src/                  Reusable pipeline code imported by the notebooks
reports/              Written findings: JD analysis, design rationale, final comparison
models/               Saved best-checkpoint models (small artifacts only)
```

## Environment

TensorFlow does not currently ship wheels for Python 3.14, so this project pins a
dedicated **Python 3.10** virtual environment:

```bash
py -3.10 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Run notebooks with the `.venv` kernel.

## Reproducing the pipeline

Notebooks are numbered and meant to be run in order; each is a standalone,
re-runnable record of one pipeline stage (see individual notebook headers for what
each stage produces and consumes).
