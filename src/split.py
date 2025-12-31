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

    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    cutoff = pd.to_datetime(cutoff_date, errors="raise")

    # Make tz-naive comparisons robust
    try:
        if getattr(out[date_col].dt, "tz", None) is not None:
            out[date_col] = out[date_col].dt.tz_localize(None)
    except Exception:
        pass

    try:
        if getattr(cutoff, "tz", None) is not None:
            cutoff = cutoff.tz_localize(None)
    except Exception:
        pass

    mask_valid = out[date_col].notna()
    mask_train = mask_valid & (out[date_col] < cutoff)
    mask_test = mask_valid & (out[date_col] >= cutoff)

    train_df = out.loc[mask_train].reset_index(drop=True)
    test_df = out.loc[mask_test].reset_index(drop=True)

    return train_df, test_df


def temporal_split_snapshot(
    df: pd.DataFrame,
    *,
    snapshot_col: str = "snapshot_date",
    cutoff_date: str = "2013-01-01",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal split aligned with the prediction time (snapshot).

    With a 6-month snapshot approach, the 'time of prediction' for each startup is
    snapshot_date = founded_at + 6 months. This function splits the dataset using
    that snapshot date:

    - Train: snapshot_date < cutoff
    - Test : snapshot_date >= cutoff

    Rows with missing snapshot dates are dropped (conservative choice).
    """
    return temporal_split(df, date_col=snapshot_col, cutoff_date=cutoff_date)


__all__ = ["temporal_split", "temporal_split_snapshot"]
