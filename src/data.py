"""Data loading and target engineering for the venturesurvive project."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import RAW_DATA_PATH, PROCESSED_DATA_PATH


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_raw_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw startups dataset from CSV.

    Parameters
    ----------
    path : optional Path
        Path to the raw CSV file. Defaults to RAW_DATA_PATH from config.

    Returns
    -------
    pd.DataFrame
        Raw startup dataset.
    """
    csv_path = path if path is not None else RAW_DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {csv_path}")
    return pd.read_csv(csv_path)


def load_cleaned_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the cleaned/preprocessed startups dataset from CSV.

    Parameters
    ----------
    path : optional Path
        Path to the cleaned CSV file. Defaults to PROCESSED_DATA_PATH.

    Returns
    -------
    pd.DataFrame
        Cleaned startup dataset.
    """
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

def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert known date columns to pandas datetime.

    This function is the single source of truth for date conversion
    in the project.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Copy of df with converted date columns.
    """
    date_cols = ["founded_at", "first_funding_at", "last_funding_at"]
    out = df.copy()

    for col in date_cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    return out


# ---------------------------------------------------------------------
# Target engineering
# ---------------------------------------------------------------------

def compute_success_target(df: pd.DataFrame) -> pd.DataFrame:
    """Compute survival-based target variables.

    Adds the following columns:
    - years_alive   : proxy for company lifespan (in years)
    - survived_5y   : boolean indicator (years_alive >= 5)
    - success       : binary target variable

    Definition of success:
    success = 1 if (years_alive >= 5) OR (status in {"acquired", "ipo"})
    success = 0 otherwise

    IMPORTANT
    ---------
    The column `status` is used ONLY to define the target variable.
    It must NEVER be used as a feature in downstream modeling to avoid
    information leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing at least date columns and `status`.

    Returns
    -------
    pd.DataFrame
        Copy of df with target-related columns added.
    """
    out = convert_dates(df)

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
            (end_date[mask_valid] - out.loc[mask_valid, "founded_at"])
            .dt.days
            / 365.25
        )

        out.loc[mask_valid, "survived_5y"] = out.loc[mask_valid, "years_alive"] >= 5

    if "status" in out.columns:
        exit_mask = out["status"].isin(["acquired", "ipo"])
        long_lived_mask = out["years_alive"].ge(5)
        out["success"] = (exit_mask | long_lived_mask.fillna(False)).astype(int)

    return out


# ---------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------

__all__ = [
    "load_raw_data",
    "load_cleaned_data",
    "convert_dates",
    "compute_success_target",
]
