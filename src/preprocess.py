"""Preprocessing for the venturesurvive project (STRICT 6-MONTH SNAPSHOT)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import RAW_DATA_PATH, PROCESSED_DATA_PATH


def filter_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """Filter raw data and compute the target.

    Notes:
    - We may use last_funding_at ONLY to compute the target proxy (years_alive).
    - We apply an anti-censoring filter: keep only startups that could have been
      observed for at least 5 years within the dataset time span.
    - After target creation, we DROP last_funding_at and other leakage-prone columns
      so they cannot be used as features (Option A strict).
    """
    if "status" not in df.columns:
        raise KeyError("Column 'status' not found in DataFrame")

    status_keep = {"acquired", "ipo", "closed"}
    out = df[df["status"].isin(status_keep)].copy()

    # ------------------------------------------------------------------
    # Convert date columns (needed for target definition + anti-censoring filter)
    # ------------------------------------------------------------------
    date_cols = ["founded_at", "first_funding_at", "last_funding_at"]
    for col in date_cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    # ------------------------------------------------------------------
    # Anti-censoring filter (publishable-grade / report-friendly)
    #
    # We approximate the dataset observation end as the max timestamp available
    # in the raw data (across funding dates). Then we only keep startups founded
    # at least 5 years before that end date.
    # ------------------------------------------------------------------
    candidates = []
    for c in ["first_funding_at", "last_funding_at"]:
        if c in out.columns:
            candidates.append(out[c])
    if not candidates:
        raise KeyError(
            "Cannot compute dataset_end_date: missing both first_funding_at and last_funding_at."
        )

    dataset_end_date = pd.concat(candidates, axis=0).max()
    if pd.isna(dataset_end_date):
        raise ValueError("dataset_end_date is NaT; cannot apply anti-censoring filter.")

    threshold_date = dataset_end_date - pd.DateOffset(years=5)

    before_censor = out.shape[0]
    out = out[out["founded_at"].notna()].copy()
    out = out[out["founded_at"] <= threshold_date].copy()
    after_censor = out.shape[0]

    dropped = before_censor - after_censor
    print(
        f"✓ Anti-censoring filter: dropped {dropped:,} rows "
        f"(kept founded_at <= {threshold_date.date()} ; dataset_end_date={dataset_end_date.date()})"
    )

    # ------------------------------------------------------------------
    # Compute years_alive proxy (label engineering only)
    # ------------------------------------------------------------------
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

    # Drop rows where success cannot be defined (conservative)
    out = out[out["success"].notna()].copy()

    # ------------------------------------------------------------------
    # STRICT: remove leakage-prone columns from the cleaned dataset
    # ------------------------------------------------------------------
    strict_drop = [
        "last_funding_at",
        "years_alive",
        "survived_5y",
        "funding_total_usd",
        "funding_rounds",
    ]
    out = out.drop(columns=[c for c in strict_drop if c in out.columns])

    # status is not allowed as feature (drop it)
    out = out.drop(columns=["status"], errors="ignore")

    return out


def build_clean_dataset(
    raw_path: Optional[Path] = None,
    processed_path: Optional[Path] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Build the cleaned dataset for STRICT snapshot modeling (Option A)."""
    from .data import load_raw_data

    df_raw = load_raw_data(raw_path if raw_path is not None else RAW_DATA_PATH)
    df_clean = filter_for_modeling(df_raw)

    if save:
        target_path = processed_path if processed_path is not None else PROCESSED_DATA_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(target_path, index=False)
        print(f"Cleaned dataset saved to {target_path}")

    return df_clean


__all__ = ["filter_for_modeling", "build_clean_dataset"]
