"""Utilities to split the dataset into train and test sets."""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    cutoff_date: str = "2013-01-01",
    date_col: str = "first_funding_at",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal train/test split based on a cutoff date.

    Rows with date_col < cutoff_date go to train,
    rows with date_col >= cutoff_date go to test.
    """
    if date_col not in df.columns:
        raise KeyError(f"Column '{date_col}' not found in DataFrame")

    out = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(out[date_col]):
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    cutoff = pd.to_datetime(cutoff_date)

    mask_train = out[date_col].notna() & (out[date_col] < cutoff)
    mask_test = out[date_col].notna() & (out[date_col] >= cutoff)

    train = out.loc[mask_train].copy()
    test = out.loc[mask_test].copy()

    return train, test


__all__ = ["temporal_split"]
