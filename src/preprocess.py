"""Preprocessing and feature engineering for the venturesurvive project."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import RAW_DATA_PATH, PROCESSED_DATA_PATH


# ---------------------------------------------------------------------
# Filtering + target definition
# ---------------------------------------------------------------------

def filter_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """Filter raw data to keep only modeling-relevant observations.

    Steps:
    - Keep only statuses: acquired, ipo, closed
    - Convert date columns to datetime
    - Compute years_alive, survived_5y, success
    - Drop rows where success cannot be defined
    """
    if "status" not in df.columns:
        raise KeyError("Column 'status' not found in DataFrame")

    status_keep = {"acquired", "ipo", "closed"}
    out = df[df["status"].isin(status_keep)].copy()

    # Convert date columns
    date_cols = ["founded_at", "first_funding_at", "last_funding_at"]
    for col in date_cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    # Initialize target-related columns
    out["years_alive"] = np.nan
    out["survived_5y"] = np.nan

    if set(date_cols).issubset(out.columns):
        end_date = out["last_funding_at"].fillna(out["first_funding_at"])
        mask = out["founded_at"].notna() & end_date.notna()

        out.loc[mask, "years_alive"] = (
            (end_date[mask] - out.loc[mask, "founded_at"]).dt.days / 365.25
        )
        out.loc[mask, "survived_5y"] = out.loc[mask, "years_alive"] >= 5

    exit_mask = out["status"].isin({"acquired", "ipo"})
    long_lived_mask = out["years_alive"].ge(5)

    out["success"] = (exit_mask | long_lived_mask.fillna(False)).astype(int)

    before = out.shape[0]
    out = out[out["success"].notna()].copy()
    after = out.shape[0]

    if after < before:
        print(f"Dropped {before - after} rows with undefined success.")

    return out


# ---------------------------------------------------------------------
# Feature engineering (lightweight, pre-modeling)
# ---------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create preliminary numeric features for modeling."""
    out = df.copy()

    # Time-based features (days)
    if {"founded_at", "first_funding_at"}.issubset(out.columns):
        out["age_at_first_funding_days"] = (
            out["first_funding_at"] - out["founded_at"]
        ).dt.days
    else:
        out["age_at_first_funding_days"] = np.nan

    if {"first_funding_at", "last_funding_at"}.issubset(out.columns):
        out["time_between_first_last_days"] = (
            out["last_funding_at"] - out["first_funding_at"]
        ).dt.days
    else:
        out["time_between_first_last_days"] = np.nan

    max_days = 365 * 30
    out["age_at_first_funding_days"] = out["age_at_first_funding_days"].clip(
        lower=0, upper=max_days
    )
    out["time_between_first_last_days"] = out["time_between_first_last_days"].clip(
        lower=0, upper=max_days
    )

    # Funding features
    if "funding_total_usd" in out.columns:
        out["funding_total_usd_num"] = pd.to_numeric(
            out["funding_total_usd"], errors="coerce"
        )
        out["log_funding_total"] = np.log1p(out["funding_total_usd_num"])
    else:
        out["funding_total_usd_num"] = np.nan
        out["log_funding_total"] = np.nan

    # Geographic flags
    if "country_code" in out.columns:
        out["is_us"] = (out["country_code"] == "USA").astype(int)
        out["has_country"] = out["country_code"].notna().astype(int)
    else:
        out["is_us"] = 0
        out["has_country"] = 0

    return out


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

def build_clean_dataset(
    raw_path: Optional[Path] = None,
    processed_path: Optional[Path] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Build the complete cleaned dataset for modeling."""
    from .data import load_raw_data

    df_raw = load_raw_data(raw_path if raw_path is not None else RAW_DATA_PATH)
    df_filtered = filter_for_modeling(df_raw)
    df_clean = engineer_features(df_filtered)

    if save:
        target_path = processed_path if processed_path is not None else PROCESSED_DATA_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(target_path, index=False)
        print(f"Cleaned dataset saved to {target_path}")

    return df_clean


# ---------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------

__all__ = [
    "filter_for_modeling",
    "engineer_features",
    "build_clean_dataset",
]
