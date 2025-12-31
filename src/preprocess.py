"""Preprocessing utilities for the venturesurvive project.

This module prepares a cleaned dataset for modeling.

Key idea (academic rigor)
------------------------
We define prediction time as a snapshot taken SNAPSHOT_MONTHS after founded_at.
Therefore:
- We must keep founded_at (required).
- We can keep raw fields used to build snapshot features (first_funding_at, last_funding_at,
  funding totals/rounds, geo, category), but snapshot censure is handled in src/features.py.
- Target engineering is handled in src/data.py (single source of truth).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .config import RAW_DATA_PATH, PROCESSED_DATA_PATH, SNAPSHOT_MONTHS
from . import data as data_mod


# ---------------------------------------------------------------------
# Filtering + target definition
# ---------------------------------------------------------------------

def filter_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """Filter raw data to keep only modeling-relevant observations.

    Current conservative choice:
    - Keep only "resolved-ish" statuses: acquired, ipo, closed
      (avoids ambiguous labels for still operating startups)
    - Require founded_at (needed to build the 6-month snapshot)
    - Compute target columns via data.compute_success_target
    """
    if "status" not in df.columns:
        raise KeyError("Column 'status' not found in DataFrame")

    status_keep = {"acquired", "ipo", "closed"}
    out = df[df["status"].isin(status_keep)].copy()

    # Compute target using the single source of truth
    out = data_mod.compute_success_target(out)

    # We MUST have founded_at to define the snapshot date founded_at + SNAPSHOT_MONTHS
    before = out.shape[0]
    out = out[out["founded_at"].notna()].copy()
    after = out.shape[0]

    if after < before:
        print(
            f"Dropped {before - after} rows with missing founded_at "
            f"(cannot define {SNAPSHOT_MONTHS}-month snapshot)."
        )

    # Ensure success exists and is numeric (compute_success_target sets it)
    if "success" not in out.columns:
        raise KeyError("Target column 'success' was not created.")

    out["success"] = out["success"].astype(int)

    return out


# ---------------------------------------------------------------------
# Type stabilization (lightweight)
# ---------------------------------------------------------------------

def stabilize_types(df: pd.DataFrame) -> pd.DataFrame:
    """Stabilize dtypes for key columns (robustness).

    This does NOT create post-snapshot features; it only ensures numeric columns
    are numeric and dates are parseable (dates already handled in compute_success_target).
    """
    out = df.copy()

    if "funding_total_usd" in out.columns:
        out["funding_total_usd"] = pd.to_numeric(out["funding_total_usd"], errors="coerce")

    if "funding_rounds" in out.columns:
        out["funding_rounds"] = pd.to_numeric(out["funding_rounds"], errors="coerce")

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
    df_raw = data_mod.load_raw_data(raw_path if raw_path is not None else RAW_DATA_PATH)

    df_filtered = filter_for_modeling(df_raw)
    df_clean = stabilize_types(df_filtered)

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
    "stabilize_types",
    "build_clean_dataset",
]
