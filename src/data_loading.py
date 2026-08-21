"""Load N-CMAPSS DS01 HDF5 data and aggregate it to one row per (unit, cycle).

The raw file stores one row per timestep within a flight (multiple thousand rows
per cycle, sampled during the flight). For degradation-trend analysis and for
cycle-based feature engineering (Phase 3), we aggregate each flight down to
summary statistics per unit-cycle.

Column groups in the file, per NASA's N-CMAPSS documentation:
  W    flight conditions (alt, Mach, TRA, T2)              -- measurable in reality
  X_s  physical sensor measurements (14 channels)            -- measurable in reality
  X_v  virtual/derived sensors (14 channels)                 -- simulation-only, NOT measurable
  T    engine health parameters (10 channels)                -- ground truth, NEVER a model input
  A    auxiliary (unit, cycle, flight class Fc, health state hs)
  Y    remaining useful life (RUL) in cycles, constant within a cycle

A real deployable model can only see W and X_s. X_v and T exist in this dataset
because it's a simulation and NASA logged the underlying physics; using them as
features would leak information no real sensor provides.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

RAW_H5_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "N-CMAPSS_DS01-005.h5"

MEASURABLE_GROUPS = ("W", "X_s")
SIMULATION_ONLY_GROUPS = ("X_v",)
GROUND_TRUTH_GROUPS = ("T",)


def _decode_names(raw: np.ndarray) -> list[str]:
    return [x.decode("utf-8") for x in raw]


def load_raw(split: str, h5_path: Path = RAW_H5_PATH) -> dict[str, np.ndarray]:
    """Load one split ("dev" or "test") as a dict of raw numpy arrays plus column names."""
    if split not in ("dev", "test"):
        raise ValueError(f"split must be 'dev' or 'test', got {split!r}")

    with h5py.File(h5_path, "r") as f:
        data = {
            "A": f[f"A_{split}"][:],
            "W": f[f"W_{split}"][:],
            "X_s": f[f"X_s_{split}"][:],
            "X_v": f[f"X_v_{split}"][:],
            "T": f[f"T_{split}"][:],
            "Y": f[f"Y_{split}"][:].flatten(),
            "A_cols": _decode_names(f["A_var"][:]),
            "W_cols": _decode_names(f["W_var"][:]),
            "X_s_cols": _decode_names(f["X_s_var"][:]),
            "X_v_cols": _decode_names(f["X_v_var"][:]),
            "T_cols": _decode_names(f["T_var"][:]),
        }
    return data


def raw_to_frame(raw: dict[str, np.ndarray]) -> pd.DataFrame:
    """Flatten one split's raw arrays into a single per-timestep DataFrame."""
    frames = [pd.DataFrame(raw["A"], columns=raw["A_cols"])]
    for group in ("W", "X_s", "X_v", "T"):
        frames.append(pd.DataFrame(raw[group], columns=raw[f"{group}_cols"]))
    frames.append(pd.DataFrame({"RUL": raw["Y"]}))
    df = pd.concat(frames, axis=1)
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    df["Fc"] = df["Fc"].astype(int)
    df["hs"] = df["hs"].astype(int)
    return df


def aggregate_to_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a per-timestep DataFrame to one row per (unit, cycle).

    W and X_s (the measurable groups) get mean/std/min/max per cycle -- std and
    the min/max range are the signal for "how much did flight conditions vary
    within this one flight," which is the whole point of using N-CMAPSS over
    the older fixed-operating-point C-MAPSS benchmark.
    X_v and T are simulation-only / ground-truth; we keep their per-cycle mean
    for exploration plots, but they are excluded from any model feature set.
    """
    w_cols = [c for c in df.columns if c in {"alt", "Mach", "TRA", "T2"}]
    xs_cols = [c for c in df.columns if c in {
        "T24", "T30", "T48", "T50", "P15", "P2", "P21", "P24", "Ps30", "P40", "P50", "Nf", "Nc", "Wf"
    }]
    xv_cols = [c for c in df.columns if c in {
        "T40", "P30", "P45", "W21", "W22", "W25", "W31", "W32", "W48", "W50", "SmFan", "SmLPC", "SmHPC", "phi"
    }]
    t_cols = [c for c in df.columns if c.endswith("_mod")]

    group = df.groupby(["unit", "cycle"], sort=True)

    agg = group[w_cols + xs_cols].agg(["mean", "std", "min", "max"])
    agg.columns = ["_".join(c) for c in agg.columns]

    other = group.agg(
        Fc=("Fc", "first"),
        hs=("hs", "last"),
        RUL=("RUL", "first"),
        n_timesteps=("RUL", "size"),
        **{f"{c}_mean": (c, "mean") for c in xv_cols + t_cols},
    )

    out = agg.join(other).reset_index()
    return out.sort_values(["unit", "cycle"]).reset_index(drop=True)


def load_cycle_level(split: str, h5_path: Path = RAW_H5_PATH) -> pd.DataFrame:
    """Convenience wrapper: raw h5 -> per-timestep frame -> per-(unit,cycle) frame."""
    raw = load_raw(split, h5_path)
    df = raw_to_frame(raw)
    return aggregate_to_cycle(df)
