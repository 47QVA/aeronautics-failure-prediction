"""Build the model-ready feature matrix from cycle-level N-CMAPSS data.

Feature set is restricted to what a real aircraft could actually provide at
inference time:
  - flight condition summary stats (W: alt, Mach, TRA, T2 -- mean/std/min/max
    per cycle)
  - physical sensor summary stats (X_s: 14 channels -- mean/std/min/max per cycle)
  - flight class (Fc), one-hot encoded
  - cycle number (how many flights this engine has done -- always known)
  - a trailing 5-cycle rolling mean of each X_s channel's per-cycle mean, per
    unit. Single-cycle sensor readings are noisy because they're confounded by
    that flight's operating conditions (see the exploration notebook's T50
    plot); smoothing over recent cycles brings out the underlying degradation
    trend without looking into the future.

Deliberately excluded: X_v (virtual/simulation-only sensors), T (ground-truth
health parameters), hs (derived from T, not a real measurement), and
n_timesteps (an artifact of how the simulation logs a flight, not a physical
signal). See src/data_loading.py for the measurable/simulation-only split.

Binary failure classification target: RUL <= FAILURE_HORIZON cycles. 20 cycles
is used as the default horizon -- roughly a fifth of a typical unit lifetime
in this dataset (75-100 cycles) -- giving a maintenance-relevant early-warning
window. Callers can pass a different horizon to `add_failure_label`.
"""
from __future__ import annotations

import pandas as pd

FAILURE_HORIZON = 20
ROLLING_WINDOW = 5

_W_BASE = ("alt", "Mach", "TRA", "T2")
_XS_BASE = ("T24", "T30", "T48", "T50", "P15", "P2", "P21", "P24", "Ps30", "P40", "P50", "Nf", "Nc", "Wf")
_STATS = ("mean", "std", "min", "max")


def feature_columns() -> list[str]:
    stat_cols = [f"{base}_{stat}" for base in (_W_BASE + _XS_BASE) for stat in _STATS]
    roll_cols = [f"{base}_mean_roll{ROLLING_WINDOW}" for base in _XS_BASE]
    return stat_cols + roll_cols + ["cycle", "Fc_1", "Fc_2", "Fc_3"]


def build_features(cycle_df: pd.DataFrame) -> pd.DataFrame:
    """Add one-hot flight-class columns and rolling-mean trend columns.

    Rolling means are computed per unit, ordered by cycle, using only past and
    current cycles (`min_periods=1`) so no future information leaks in.
    """
    df = cycle_df.sort_values(["unit", "cycle"]).copy()
    for fc in (1, 2, 3):
        df[f"Fc_{fc}"] = (df["Fc"] == fc).astype(int)

    grouped = df.groupby("unit", sort=False)
    for base in _XS_BASE:
        df[f"{base}_mean_roll{ROLLING_WINDOW}"] = grouped[f"{base}_mean"].transform(
            lambda s: s.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        )

    missing = [c for c in feature_columns() if c not in df.columns]
    if missing:
        raise ValueError(f"cycle_df is missing expected columns: {missing}")
    return df


def add_failure_label(df: pd.DataFrame, horizon: int = FAILURE_HORIZON) -> pd.DataFrame:
    df = df.copy()
    df["fails_within_horizon"] = (df["RUL"] <= horizon).astype(int)
    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y_rul, y_fails) using only the approved feature columns."""
    X = df[feature_columns()]
    y_rul = df["RUL"]
    y_fails = df["fails_within_horizon"]
    return X, y_rul, y_fails
