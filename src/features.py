"""Feature engineering utilities for the venturesurvive project."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Time-based features
# ---------------------------------------------------------------------

def make_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create basic time-based features (in days)."""
    out = df.copy()

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

    return out


# ---------------------------------------------------------------------
# Geographic features
# ---------------------------------------------------------------------

def make_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple geographic indicator features."""
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

def _extract_first_category(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str) or not value:
        return None
    return value.split("|")[0].strip() or None


def make_category_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract a main category from `category_list`."""
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
    """Assemble a feature-ready DataFrame from a cleaned input DataFrame."""
    out = df.copy()

    # Time features
    out = make_time_features(out)

    # Geographic features
    out = make_geo_features(out)

    # Category features
    out = make_category_features(out)

    # Funding features (numeric-safe)
    if "funding_total_usd" in out.columns:
        funding = pd.to_numeric(out["funding_total_usd"], errors="coerce")
        out["funding_total_usd_num"] = funding
        out["log_funding_total"] = np.log1p(funding)

    if {"funding_total_usd_num", "funding_rounds"}.issubset(out.columns):
        rounds_safe = out["funding_rounds"].replace(0, np.nan)
        out["funding_per_round"] = out["funding_total_usd_num"] / rounds_safe
        out["log_funding_per_round"] = np.log1p(out["funding_per_round"])

        total_thr = out["funding_total_usd_num"].quantile(0.75)
        out["high_total_funding"] = (out["funding_total_usd_num"] >= total_thr).astype(int)

        round_thr = out["funding_per_round"].quantile(0.75)
        out["high_funding_per_round"] = (
            out["funding_per_round"] >= round_thr
        ).astype(int)

        out["has_multiple_rounds"] = (out["funding_rounds"] >= 2).astype(int)

    # Additional temporal ratios (years)
    if {"founded_at", "first_funding_at"}.issubset(out.columns):
        mask = out["founded_at"].notna() & out["first_funding_at"].notna()
        out["time_to_first_funding_years"] = np.where(
            mask,
            (out["first_funding_at"] - out["founded_at"]).dt.days / 365.25,
            np.nan,
        )

    if {"founded_at", "last_funding_at"}.issubset(out.columns):
        mask = out["founded_at"].notna() & out["last_funding_at"].notna()
        out["company_age_at_last_funding_years"] = np.where(
            mask,
            (out["last_funding_at"] - out["founded_at"]).dt.days / 365.25,
            np.nan,
        )

    if {"first_funding_at", "last_funding_at", "funding_rounds"}.issubset(out.columns):
        denom = np.where(out["funding_rounds"] >= 2, out["funding_rounds"] - 1, np.nan)
        mask = (
            out["first_funding_at"].notna()
            & out["last_funding_at"].notna()
            & (out["funding_rounds"] >= 2)
        )
        out["avg_round_interval_years"] = np.where(
            mask,
            (out["last_funding_at"] - out["first_funding_at"]).dt.days / 365.25 / denom,
            np.nan,
        )

    # Drop identifiers / leakage-prone columns
    drop_cols = ["permalink", "name", "homepage_url", "category_list"]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])

    return out


# ---------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------

__all__ = [
    "make_time_features",
    "make_geo_features",
    "make_category_features",
    "assemble_features",
]
