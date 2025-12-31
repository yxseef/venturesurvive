"""Data loading and target engineering for the venturesurvive project."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import RAW_DATA_PATH, PROCESSED_DATA_PATH, SNAPSHOT_MONTHS


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_raw_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw startups dataset from CSV."""
    csv_path = path if path is not None else RAW_DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {csv_path}")
    return pd.read_csv(csv_path)


def load_cleaned_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the cleaned/preprocessed startups dataset from CSV."""
    csv_path = path if path is not None else PROCESSED_DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cleaned data file not found at {csv_path}. "
            "Run the preprocessing step first."
        )
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------
# Date handling
# ---------------------------------------------------------------------

def _to_datetime_no_tz(s: pd.Series) -> pd.Series:
    """Parse datetimes and strip timezone if present (robust comparisons)."""
    out = pd.to_datetime(s, errors="coerce")
    try:
        if getattr(out.dt, "tz", None) is not None:
            out = out.dt.tz_localize(None)
    except Exception:
        pass
    return out


def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert known date columns to pandas datetime (tz-naive)."""
    date_cols = ["founded_at", "first_funding_at", "last_funding_at"]
    out = df.copy()

    for col in date_cols:
        if col in out.columns:
            out[col] = _to_datetime_no_tz(out[col])

    return out


# ---------------------------------------------------------------------
# Target engineering
# ---------------------------------------------------------------------

def compute_success_target(df: pd.DataFrame) -> pd.DataFrame:
    """Compute survival-based target variables.

    Adds:
    - snapshot_date : founded_at + SNAPSHOT_MONTHS (for reference/sanity checks)
    - years_alive   : proxy for company lifespan (in years), based on last/first funding date
    - survived_5y   : indicator (years_alive >= 5)
    - success       : binary target

    Definition of success:
    success = 1 if (status in {"acquired", "ipo"}) OR (years_alive >= 5)
    success = 0 otherwise

    IMPORTANT:
    - `status` is used ONLY to define the target variable.
      It must NEVER be used as a feature downstream.
    - years_alive is a proxy; it should NEVER be used as a feature when predicting from
      an early-life snapshot (e.g., first 6 months). We explicitly drop it in features.py.
    """
    out = convert_dates(df)

    # Snapshot date (reference point: prediction time = founded_at + 6 months)
    if "founded_at" in out.columns:
        out["snapshot_date"] = out["founded_at"] + pd.DateOffset(months=SNAPSHOT_MONTHS)
    else:
        out["snapshot_date"] = pd.NaT

    # Initialize columns
    out["years_alive"] = np.nan
    out["survived_5y"] = np.nan
    out["success"] = np.nan

    required_dates = {"founded_at", "first_funding_at", "last_funding_at"}
    if required_dates.issubset(out.columns):
        # Use last observed funding date as proxy for end of observation
        end_date = out["last_funding_at"].fillna(out["first_funding_at"])

        mask_valid = out["founded_at"].notna() & end_date.notna()
        out.loc[mask_valid, "years_alive"] = (
            (end_date[mask_valid] - out.loc[mask_valid, "founded_at"]).dt.days / 365.25
        )

        out.loc[mask_valid, "survived_5y"] = out.loc[mask_valid, "years_alive"] >= 5

    # Compute success
    if "status" in out.columns:
        exit_mask = out["status"].isin(["acquired", "ipo"])
        long_lived_mask = out["years_alive"].ge(5)

        # NOTE: if years_alive is NaN, long_lived_mask becomes False (conservative).
        out["success"] = (exit_mask | long_lived_mask.fillna(False)).astype(int)

    return out


__all__ = [
    "load_raw_data",
    "load_cleaned_data",
    "convert_dates",
    "compute_success_target",
]
