"""Feature engineering utilities for the venturesurvive project."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def make_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create basic time-based features.

    Requires datetime columns:
    - founded_at
    - first_funding_at
    - last_funding_at
    """
    out = df.copy()

    if {"founded_at", "first_funding_at", "last_funding_at"}.issubset(out.columns):
        out["age_at_first_funding"] = (
            out["first_funding_at"] - out["founded_at"]
        ).dt.days
        out["time_between_first_last"] = (
            out["last_funding_at"] - out["first_funding_at"]
        ).dt.days
    else:
        # Columns may be missing if dates were not converted upstream
        out["age_at_first_funding"] = np.nan
        out["time_between_first_last"] = np.nan

    return out


def make_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple geographic features from country / region / city.

    This keeps original columns and adds a few boolean flags.
    """
    out = df.copy()

    country = out.get("country_code")
    out["is_us"] = country.eq("USA") if country is not None else False
    out["is_uk"] = country.eq("GBR") if country is not None else False
    out["is_eu"] = country.isin({
        "AUT",
        "BEL",
        "BGR",
        "HRV",
        "CYP",
        "CZE",
        "DNK",
        "EST",
        "FIN",
        "FRA",
        "DEU",
        "GRC",
        "HUN",
        "IRL",
        "ITA",
        "LVA",
        "LTU",
        "LUX",
        "MLT",
        "NLD",
        "POL",
        "PRT",
        "ROU",
        "SVK",
        "SVN",
        "ESP",
        "SWE",
    }) if country is not None else False

    # Missingness flags can help the model
    for col in ["country_code", "state_code", "region", "city"]:
        if col in out.columns:
            out[f"{col}_missing"] = out[col].isna() | (out[col] == "")

    return out


def _extract_first_category(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str) or not value:
        return None
    # Categories are often separated by '|'
    return value.split("|")[0].strip() or None


def make_category_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract a main category from `category_list` column.

    Adds a new column `category_main`.
    """
    out = df.copy()
    if "category_list" in out.columns:
        out["category_main"] = out["category_list"].map(_extract_first_category)
    else:
        out["category_main"] = np.nan
    return out

def assemble_features(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble a feature-ready DataFrame from a cleaned input DataFrame.

    This function applies the different feature builders and returns
    a new DataFrame with:
    - numeric funding column
    - time features
    - geographic flags
    - main category
    It also drops a few obviously non-informative identifier columns.
    """
    out = df.copy()

    # 1. Time-based features
    out = make_time_features(out)

    # 2. Geographic features
    out = make_geo_features(out)

    # 3. Category-based features
    out = make_category_features(out)

    # 4. Numeric log funding
    if "funding_total_usd" in out.columns:
        funding = pd.to_numeric(out["funding_total_usd"], errors="coerce")
        out["log_funding_total"] = np.log1p(funding)

    # 5. OPTION B : Nouveau Feature Engineering (très académique)

    # 5.1 Funding per round
    if {"funding_total_usd", "funding_rounds"}.issubset(out.columns):
        rounds_safe = out["funding_rounds"].replace(0, np.nan)
        out["funding_per_round"] = out["funding_total_usd"] / rounds_safe
        out["log_funding_per_round"] = np.log1p(out["funding_per_round"])

        # Thresholds (75th percentile)
        total_thr = out["funding_total_usd"].quantile(0.75)
        out["high_total_funding"] = (out["funding_total_usd"] >= total_thr).astype(int)

        round_thr = out["funding_per_round"].quantile(0.75)
        out["high_funding_per_round"] = (
            out["funding_per_round"] >= round_thr
        ).astype(int)

        out["has_multiple_rounds"] = (out["funding_rounds"] >= 2).astype(int)

    # 5.2 Time until first funding
    if {"founded_at", "first_funding_at"}.issubset(out.columns):
        mask = out["founded_at"].notna() & out["first_funding_at"].notna()
        out["time_to_first_funding_years"] = np.where(
            mask,
            (out["first_funding_at"] - out["founded_at"]).dt.days / 365.25,
            np.nan,
        )

    # 5.3 Company age at last funding
    if {"founded_at", "last_funding_at"}.issubset(out.columns):
        mask = out["founded_at"].notna() & out["last_funding_at"].notna()
        out["company_age_at_last_funding_years"] = np.where(
            mask,
            (out["last_funding_at"] - out["founded_at"]).dt.days / 365.25,
            np.nan,
        )

    # 5.4 Average interval between funding rounds
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

    # 6. Drop non-informative columns
    drop_cols = [
        "permalink",
        "name",
        "homepage_url",
        "category_list",
    ]
    keep_cols = [c for c in out.columns if c not in drop_cols]
    out = out[keep_cols]

    return out
