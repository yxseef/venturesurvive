"""Data loading and basic preprocessing utilities for the venturesurvive project."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


# Project root is the parent of the src/ directory where this file lives
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH_DEFAULT = PROJECT_ROOT / "data" / "startups_raw.csv"


def load_raw(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the raw startups dataset from CSV.

    Parameters
    ----------
    path: optional custom path to the CSV. Defaults to DATA_PATH_DEFAULT.
    """
    csv_path = Path(path) if path is not None else DATA_PATH_DEFAULT
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {csv_path!s}")
    df = pd.read_csv(csv_path)
    return df


def filter_status(df: pd.DataFrame, statuses: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Filter rows to keep only selected status values.

    Default is {"acquired", "ipo", "closed"}.
    """
    if "status" not in df.columns:
        raise KeyError("Column 'status' not found in DataFrame")

    keep = set(statuses) if statuses is not None else {"acquired", "ipo", "closed"}
    mask = df["status"].isin(keep)
    return df.loc[mask].copy()


def create_label(
    df: pd.DataFrame,
    positive: Optional[Iterable[str]] = None,
    negative: Optional[Iterable[str]] = None,
    label_col: str = "success",
) -> pd.DataFrame:
    """Create a binary label column from status.

    By default:
    - positive: {"acquired", "ipo"}
    - negative: {"closed"}
    """
    if "status" not in df.columns:
        raise KeyError("Column 'status' not found in DataFrame")

    positive = set(positive) if positive is not None else {"acquired", "ipo"}
    negative = set(negative) if negative is not None else {"closed"}

    mapping = {s: 1 for s in positive}
    mapping.update({s: 0 for s in negative})

    out = df.copy()
    out[label_col] = out["status"].map(mapping)
    if out[label_col].isna().any():
        # We deliberately keep rows with NaN label so caller can decide what to do.
        pass
    return out


def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert known date columns to pandas datetime (in-place copy)."""
    date_cols = ["founded_at", "first_funding_at", "last_funding_at"]
    out = df.copy()
    for col in date_cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def clean_funding(df: pd.DataFrame, column: str = "funding_total_usd") -> pd.DataFrame:
    """Clean the total funding column.

    - Convert to numeric (non-parsable values become NaN).
    - Replace negative values by NaN (defensive).
    """
    out = df.copy()
    if column not in out.columns:
        return out

    funding = pd.to_numeric(out[column], errors="coerce")
    funding = funding.where(funding >= 0)
    out[column] = funding
    return out
