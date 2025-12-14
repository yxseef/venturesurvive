"""Train/test splitting utilities for venturesurvive."""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    date_col: str,
    cutoff_date: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe into train/test using a temporal cutoff date.

    - Train: rows with date < cutoff
    - Test:  rows with date >= cutoff
    Rows with missing dates are dropped (conservative choice).
    """
    out = df.copy()

    # Ensure datetime for the date column
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    # Make cutoff timezone-compatible with the series
    cutoff = pd.to_datetime(cutoff_date, errors="raise")

    # If the series is tz-aware, localize cutoff to same tz
    if getattr(out[date_col].dt, "tz", None) is not None:
        cutoff = cutoff.tz_localize(out[date_col].dt.tz)
    
    # Convert both to tz-naive for comparison (simpler and more robust)
    if hasattr(out[date_col].dt, "tz") and out[date_col].dt.tz is not None:
        out[date_col] = out[date_col].dt.tz_localize(None)
    if hasattr(cutoff, "tz") and cutoff.tz is not None:
        cutoff = cutoff.tz_localize(None)

    # Build masks
    mask_valid = out[date_col].notna()
    mask_train = mask_valid & (out[date_col] < cutoff)
    mask_test = mask_valid & (out[date_col] >= cutoff)

    train_df = out.loc[mask_train].reset_index(drop=True)
    test_df = out.loc[mask_test].reset_index(drop=True)

    return train_df, test_df


__all__ = ["temporal_split"]
