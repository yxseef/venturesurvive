""""Preprocessing for the venturesurvive project (STRICT 6-MONTH SNAPSHOT)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import RAW_DATA_PATH, PROCESSED_DATA_PATH


def filter_for_modeling(df: pd.DataFrame, target_horizon_years: int = 5) -> pd.DataFrame:
    """Filter raw data and compute the target (STRICT 6-month snapshot).

    - Fixes future/outlier dates (critical for eligibility diagnostics)
    - Computes snapshot_date = founded_at + 6 months
    - Computes last_observed_event_date from observation columns (not founded_at)
    - Applies eligibility filter for horizon (5y or 2y)
    - Builds label 'success'
    - Drops leakage-prone columns but KEEPS snapshot_date for temporal split
    """
    if "status" not in df.columns:
        raise KeyError("Column 'status' not found in DataFrame")

    status_keep = {"acquired", "ipo", "closed"}
    out = df[df["status"].isin(status_keep)].copy()

    # -----------------------------
    # Parse date columns if present
    # -----------------------------
    date_cols = [
        "founded_at",
        "first_funding_at",
        "last_funding_at",
        "acquired_at",
        "ipo_at",
        "closed_at",
        "updated_at",
    ]
    for col in date_cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    # ----------------------------------------------------
    # Sanity cap: remove future/outlier dates
    # + handle implausibly old founded_at (1970 placeholder)
    # ----------------------------------------------------
    analysis_date = pd.Timestamp.today().normalize()
    max_allowed = analysis_date + pd.DateOffset(days=1)  # small tolerance

    # General min for "event" timestamps (keep quite permissive)
    min_allowed_general = pd.Timestamp("1970-01-01")

    # Stricter min for founded_at (avoids UNIX epoch placeholders)
    # You can set 1990-01-01 if you want to be stricter.
    min_allowed_founded = pd.Timestamp("1980-01-01")

    sanity_cols = [c for c in date_cols if c in out.columns]
    for c in sanity_cols:
        out.loc[out[c] > max_allowed, c] = pd.NaT

        # founded_at stricter
        if c == "founded_at":
            out.loc[out[c] < min_allowed_founded, c] = pd.NaT
        else:
            out.loc[out[c] < min_allowed_general, c] = pd.NaT

    # ✅ NEW: kill the common placeholder exactly (prevents snapshot_date starting at 1980-07-01)
    if "founded_at" in out.columns:
        placeholder = pd.Timestamp("1980-01-01")
        out.loc[out["founded_at"] == placeholder, "founded_at"] = pd.NaT

    # ------------------------------------------
    # snapshot_date (STRICT: 6 months after founding)
    # ------------------------------------------
    if "founded_at" not in out.columns:
        raise KeyError("Column 'founded_at' not found; cannot build snapshot_date.")

    before = out.shape[0]
    out = out[out["founded_at"].notna()].copy()
    dropped = before - out.shape[0]
    if dropped:
        print(f"✓ Dropped {dropped:,} rows with missing founded_at (needed for snapshot_date)")

    out["snapshot_date"] = out["founded_at"] + pd.DateOffset(months=6)

    # ---------------------------------------------------------
    # Eligibility diagnostic: dataset_end_date from OBSERVATIONS
    # ---------------------------------------------------------
    obs_cols = [
        c
        for c in ["first_funding_at", "last_funding_at", "acquired_at", "ipo_at", "closed_at", "updated_at"]
        if c in out.columns
    ]
    if not obs_cols:
        raise KeyError("No observation date columns available to compute dataset_end_date.")

    out["last_observed_event_date"] = out[obs_cols].max(axis=1)
    dataset_end_date = out["last_observed_event_date"].max()
    if pd.isna(dataset_end_date):
        raise ValueError("dataset_end_date is NaT; cannot compute eligibility flags.")

    out["eligible_latest_snapshot_for_5y"] = (out["snapshot_date"] + pd.DateOffset(years=5)) <= dataset_end_date
    out["eligible_latest_snapshot_for_2y"] = (out["snapshot_date"] + pd.DateOffset(years=2)) <= dataset_end_date

    print("\n================= Horizon eligibility diagnostic =================")
    print(f"DATA_END_DATE (max last_observed_event_date): {dataset_end_date.date()}")
    n = len(out)
    e5 = int(out["eligible_latest_snapshot_for_5y"].sum())
    e2 = int(out["eligible_latest_snapshot_for_2y"].sum())
    print(f"Eligible @ 5y: {e5:,} / {n:,} ({e5 / n:.1%})")
    print(f"Eligible @ 2y: {e2:,} / {n:,} ({e2 / n:.1%})")
    print("=================================================================\n")

    # -------------------------
    # Apply eligibility filter
    # -------------------------
    if target_horizon_years == 5:
        elig_col = "eligible_latest_snapshot_for_5y"
    elif target_horizon_years == 2:
        elig_col = "eligible_latest_snapshot_for_2y"
    else:
        raise ValueError("target_horizon_years must be 2 or 5")

    before = out.shape[0]
    out = out[out[elig_col]].copy()
    print(
        f"✓ Eligibility filter ({target_horizon_years}y): dropped {before - out.shape[0]:,} rows "
        f"(kept {elig_col}=True ; dataset_end_date={dataset_end_date.date()})"
    )

    # ------------------------------------------
    # Label engineering (same logic as before, but horizon-aware)
    # ------------------------------------------
    out["years_alive"] = np.nan
    out["survived_5y"] = pd.Series(pd.NA, index=out.index, dtype="boolean")

    needed = {"founded_at", "first_funding_at", "last_funding_at"}
    if needed.issubset(out.columns):
        end_date = out["last_funding_at"].fillna(out["first_funding_at"])
        mask = out["founded_at"].notna() & end_date.notna()

        out.loc[mask, "years_alive"] = (
            (end_date[mask] - out.loc[mask, "founded_at"]).dt.days / 365.25
        )
        out.loc[mask, "survived_5y"] = (out.loc[mask, "years_alive"] >= target_horizon_years).astype("boolean")

    exit_mask = out["status"].isin({"acquired", "ipo"})
    long_lived_mask = out["years_alive"].ge(target_horizon_years)
    out["success"] = (exit_mask | long_lived_mask.fillna(False)).astype(int)

    # ------------------------------------------
    # STRICT: drop leakage-prone columns
    # (BUT KEEP snapshot_date for temporal split)
    # ------------------------------------------
    strict_drop = [
        "last_funding_at",
        "years_alive",
        "survived_5y",
        "funding_total_usd",
        "funding_rounds",
        "last_observed_event_date",
        "eligible_latest_snapshot_for_5y",
        "eligible_latest_snapshot_for_2y",
    ]
    out = out.drop(columns=[c for c in strict_drop if c in out.columns])

    # status not allowed as feature
    out = out.drop(columns=["status"], errors="ignore")

    return out


def build_clean_dataset(
    raw_path: Optional[Path] = None,
    processed_path: Optional[Path] = None,
    save: bool = True,
    target_horizon_years: int = 5,
) -> pd.DataFrame:
    """Build the cleaned dataset for STRICT snapshot modeling."""
    from .data import load_raw_data

    df_raw = load_raw_data(raw_path if raw_path is not None else RAW_DATA_PATH)

    # Choose horizon here: 5y (default) or 2y
    df_clean = filter_for_modeling(df_raw, target_horizon_years=target_horizon_years)

    if save:
        target_path = processed_path if processed_path is not None else PROCESSED_DATA_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(target_path, index=False)
        print(f"Cleaned dataset saved to {target_path}")

    return df_clean


__all__ = ["filter_for_modeling", "build_clean_dataset"]
