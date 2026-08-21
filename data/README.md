# Data

This project uses **N-CMAPSS**, subset **DS01** (single degradation mode: High-Pressure
Turbine efficiency modifier), from NASA's Prognostics Data Repository.

Raw files are not committed to this repo (multi-GB HDF5). Fetch them yourself:

## Obtaining the data

1. Download the full archive from NASA's PCoE Data Set Repository:
   `https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip`
   (linked from the official index at
   https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/ —
   verify this link is still current before relying on it; NASA has moved this download before.)
2. Unzip it. Inside `data_set/` you'll find `N-CMAPSS_DS01-005.h5` through `N-CMAPSS_DS08*.h5`,
   an example notebook, and a PDF describing the simulation methodology.
3. Copy **only** `N-CMAPSS_DS01-005.h5` into `data/raw/` in this repo. The other DS0x files
   (multi-fault variants) are out of scope for v1 — see `CLAUDE.md` for the open decision on
   whether DS02+ becomes a follow-up.

## Citation

Chao, M., Kulkarni, C., Goebel, K., & Fink, O. (2021). *Aircraft Engine Run-to-Failure Dataset
under real flight conditions*, NASA Ames Prognostics Data Repository.

## Layout

- `data/raw/` — the untouched `N-CMAPSS_DS01-005.h5` (gitignored)
- `data/processed/` — feature-engineered / cycle-aggregated outputs produced by `src/` scripts
  (gitignored; regenerate from raw + code, don't hand-edit)

## DS01 structure (from the HDF5 file)

Each file stores separate train/test splits as HDF5 datasets:
`W` (flight condition/scenario-descriptor operating conditions), `X_s` (measured signals),
`X_v` (virtual/derived signals), `T` (health-parameter degradation states), `Y` (RUL target),
`A` (auxiliary: unit number, cycle, flight class, health state). Column names for each block
are stored in matching `*_var` datasets. See `docs/references/` for NASA's own example
notebook and the dataset PDF, which document this in full.
