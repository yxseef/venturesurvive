"""Feature engineering utilities for the venturesurvive project.

Snapshot definition
-------------------
We define the prediction time as a snapshot taken SNAPSHOT_MONTHS after a startup's
foundation date (founded_at). Features must be derivable from information available
up to founded_at + SNAPSHOT_MONTHS.

Because the dataset is Crunchbase-like and often contains lifetime aggregates
(e.g., total funding, total rounds), we apply conservative censoring rules:
- We NEVER use last_funding_at or any feature directly derived from it.
- Lifetime totals are only used when we can be confident they are fully observed
  within the snapshot window (e.g., last_funding_at <= snapshot_date).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import SNAPSHOT_MONTHS


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _to_datetime_no_tz(s: pd.Series) -> pd.Series:
    """Parse datetimes and strip timezone to keep comparisons robust."""
    out = pd.to_datetime(s, errors="coerce")
    # strip tz if present
    try:
        if getattr(out.dt, "tz", None) is not None:
            out = out.dt.tz_localize(None)
    except Exception:
        pass
    return out


def _extract_first_category(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str) or not value:
        return None
    return value.split("|")[0].strip() or None


# ---------------------------------------------------------------------
# Snapshot (6-month) features
# ---------------------------------------------------------------------

def make_snapshot_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features restricted to the first SNAPSHOT_MONTHS months of life."""
    out = df.copy()

    # Ensure datetime columns exist and are tz-naive
    if "founded_at" in out.columns:
        out["founded_at"] = _to_datetime_no_tz(out["founded_at"])
    if "first_funding_at" in out.columns:
        out["first_funding_at"] = _to_datetime_no_tz(out["first_funding_at"])
    if "last_funding_at" in out.columns:
        out["last_funding_at"] = _to_datetime_no_tz(out["last_funding_at"])

    # Snapshot date: founded_at + SNAPSHOT_MONTHS
    if "founded_at" in out.columns:
        out["snapshot_date"] = out["founded_at"] + pd.DateOffset(months=SNAPSHOT_MONTHS)
    else:
        out["snapshot_date"] = pd.NaT

    # Max plausible snapshot duration in days (approx) for clipping
    max_snapshot_days = int(round(SNAPSHOT_MONTHS * 30.4375))  # ~183 for 6 months

    # Funding timing within snapshot
    if {"founded_at", "first_funding_at"}.issubset(out.columns):
        has_first_funding_by_snap = (
            out["founded_at"].notna()
            & out["first_funding_at"].notna()
            & out["snapshot_date"].notna()
            & (out["first_funding_at"] <= out["snapshot_date"])
        )

        out["has_first_funding_6m"] = has_first_funding_by_snap.astype(int)

        out["age_at_first_funding_days_6m"] = np.where(
            has_first_funding_by_snap,
            (out["first_funding_at"] - out["founded_at"]).dt.days,
            np.nan,
        )

        # clip to [0, max_snapshot_days] to reduce outliers / date noise
        out["age_at_first_funding_days_6m"] = pd.to_numeric(
            out["age_at_first_funding_days_6m"], errors="coerce"
        ).clip(lower=0, upper=max_snapshot_days)

    else:
        out["has_first_funding_6m"] = 0
        out["age_at_first_funding_days_6m"] = np.nan

    # Cohort features from founded_at (allowed at t=0)
    if "founded_at" in out.columns:
        out["founded_year"] = out["founded_at"].dt.year
        out["founded_month"] = out["founded_at"].dt.month
        out["founded_quarter"] = out["founded_at"].dt.quarter
        out["founded_year_missing"] = out["founded_at"].isna().astype(int)
    else:
        out["founded_year"] = np.nan
        out["founded_month"] = np.nan
        out["founded_quarter"] = np.nan
        out["founded_year_missing"] = 1

    # Conservative funding aggregates: only usable if fully observed within snapshot
    # (i.e., last_funding_at <= snapshot_date). Otherwise we set to NaN to avoid leakage.
    if "funding_total_usd" in out.columns:
        out["funding_total_usd_num"] = pd.to_numeric(out["funding_total_usd"], errors="coerce")
    else:
        out["funding_total_usd_num"] = np.nan

    if "funding_rounds" in out.columns:
        out["funding_rounds_num"] = pd.to_numeric(out["funding_rounds"], errors="coerce")
    else:
        out["funding_rounds_num"] = np.nan

    fully_observed_by_snap = (
        out.get("last_funding_at", pd.Series([pd.NaT] * len(out))).notna()
        & out["snapshot_date"].notna()
        & (out.get("last_funding_at") <= out["snapshot_date"])
    )

    out["funding_fully_observed_6m"] = fully_observed_by_snap.astype(int)

    out["funding_total_usd_6m"] = np.where(
        fully_observed_by_snap,
        out["funding_total_usd_num"],
        np.nan,
    )
    out["log_funding_total_6m"] = np.log1p(out["funding_total_usd_6m"])

    out["funding_rounds_6m"] = np.where(
        fully_observed_by_snap,
        out["funding_rounds_num"],
        np.nan,
    )

    # Funding per round (only when both are known within snapshot)
    rounds_safe = pd.Series(out["funding_rounds_6m"]).replace(0, np.nan)
    out["funding_per_round_6m"] = out["funding_total_usd_6m"] / rounds_safe
    out["log_funding_per_round_6m"] = np.log1p(out["funding_per_round_6m"])

    return out


# ---------------------------------------------------------------------
# Geographic features
# ---------------------------------------------------------------------

def make_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple geographic indicator features (available at t=0)."""
    out = df.copy()

    country = out.get("country_code")

    if country is not None:
        out["is_us"] = country.eq("USA").astype(int)
        out["is_uk"] = country.eq("GBR").astype(int)
        out["is_eu"] = country.isin(
            {
                "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA",
                "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD",
                "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE",
            }
        ).astype(int)
    else:
        out["is_us"] = 0
        out["is_uk"] = 0
        out["is_eu"] = 0

    for col in ["country_code", "state_code", "region", "city"]:
        if col in out.columns:
            out[f"{col}_missing"] = out[col].isna().astype(int)

    return out


# ---------------------------------------------------------------------
# Category features
# ---------------------------------------------------------------------

def make_category_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract a main category from `category_list` (available at t=0)."""
    out = df.copy()

    if "category_list" in out.columns:
        out["category_main"] = out["category_list"].map(_extract_first_category)
    else:
        out["category_main"] = np.nan

    return out


# ---------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------

def assemble_features(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble a feature-ready DataFrame under the 6-month snapshot constraint."""
    out = df.copy()

    # Snapshot features (restricted to first SNAPSHOT_MONTHS months)
    out = make_snapshot_features(out)

    # Geo + category features
    out = make_geo_features(out)
    out = make_category_features(out)

    # IMPORTANT: remove leakage-prone / post-outcome columns from features
    # Keep date columns for splitting (train.py will drop them after split),
    # but do not keep engineered lifetime proxies.
    leakage_cols = [
        "years_alive",
        "survived_5y",
        # raw lifetime aggregates (we use censored versions instead)
        "funding_total_usd",
        "funding_rounds",
        # do not use last_funding_at as a feature
        # (we keep it only if present for potential label construction upstream,
        #  but it's safer to drop it here from the modeling frame)
        "last_funding_at",
        # category_list kept only for deriving category_main
        "category_list",
        # identifiers
        "permalink",
        "name",
        "homepage_url",
    ]
    out = out.drop(columns=[c for c in leakage_cols if c in out.columns])

    return out


# ---------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------

__all__ = [
    "make_snapshot_features",
    "make_geo_features",
    "make_category_features",
    "assemble_features",
]
