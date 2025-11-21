"""Utilities to split the dataset into train and test sets."""

from __future__ import annotations

from datetime import datetime
from typing import Tuple

import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    cutoff_date: str = "2013-01-01",
    date_col: str = "first_funding_at",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal train/test split based on a cutoff date.

    Rows with date_col < cutoff go to train, >= cutoff go to test.
    The date column is expected to be datetime; it is converted if needed.
    """
    out = df.copy()

    if date_col not in out.columns:
        raise KeyError(f"Column {date_col!r} not found in DataFrame")

    if not pd.api.types.is_datetime64_any_dtype(out[date_col]):
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    cutoff = pd.to_datetime(cutoff_date)
    mask_train = out[date_col] < cutoff
    train = out.loc[mask_train].copy()
    test = out.loc[~mask_train].copy()
    return train, test
