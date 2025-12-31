"""Feature engineering utilities for the venturesurvive project (STRICT 6-MONTH SNAPSHOT)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


SNAPSHOT_MONTHS = 6


# ---------------------------------------------------------------------
# Snapshot + time features (STRICT)
# ---------------------------------------------------------------------

def make_snapshot_features(df: pd.DataFrame, snapshot_months: int = SNAPSHOT_MONTHS) -> pd.DataFrame:
    """Create STRICT snapshot-safe features (t <= founded_at + snapshot_months).

    Key idea:
    - We may use event timestamps to derive *whether an event happened by snapshot*.
      (e.g., "funded within 6 months"), which is equivalent to knowledge available at snapshot time.
    - We must NOT use any information that depends on future outcomes beyond snapshot
      (e.g., last_funding_at, lifetime funding totals/rounds).
    """
    out = df.copy()

    # Ensure datetime columns (tz-naive)
    for col in ["founded_at", "first_funding_at"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
            if getattr(out[col].dt, "tz", None) is not None:
                out[col] = out[col].dt.tz_localize(None)

    # Snapshot date = founded_at + 6 months (calendar months)
    out["snapshot_date"] = pd.NaT
    mask_founded = out["founded_at"].notna()
    out.loc[mask_founded, "snapshot_date"] = out.loc[mask_founded, "founded_at"] + pd.DateOffset(months=snapshot_months)

    # Horizon in days (varies slightly with calendar months)
    out["snapshot_horizon_days"] = np.where(
        out["snapshot_date"].notna() & out["founded_at"].notna(),
        (out["snapshot_date"] - out["founded_at"]).dt.days,
        np.nan,
    )

    # Funding within snapshot (STRICT)
    has_first = out["first_funding_at"].notna() & out["snapshot_date"].notna()
    funded_within = has_first & (out["first_funding_at"] <= out["snapshot_date"])
    out["funded_within_6m"] = funded_within.astype(int)
    out["first_funding_missing"] = out["first_funding_at"].isna().astype(int)

    # Age at first funding (censored at snapshot)
    # - If funded within snapshot: use true delay
    # - Else: set to horizon_days (we know "no funding by snapshot", so delay >= horizon; we encode as horizon)
    out["age_at_first_funding_days"] = np.nan
    mask_ok = out["founded_at"].notna() & out["snapshot_date"].notna()

    # funded within snapshot => exact
    out.loc[mask_ok & funded_within, "age_at_first_funding_days"] = (
        (out.loc[mask_ok & funded_within, "first_funding_at"] - out.loc[mask_ok & funded_within, "founded_at"]).dt.days
    )

    # not funded within snapshot => censored to horizon
    out.loc[mask_ok & (~funded_within), "age_at_first_funding_days"] = out.loc[mask_ok & (~funded_within), "snapshot_horizon_days"]

    # Basic sanity clipping
    out["age_at_first_funding_days"] = out["age_at_first_funding_days"].clip(lower=0)

    return out


# ---------------------------------------------------------------------
# Geographic features (safe at t=0)
# ---------------------------------------------------------------------

def make_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple geographic indicator features (t=0 safe)."""
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
# Category features (safe at t=0)
# ---------------------------------------------------------------------

def _extract_first_category(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str) or not value:
        return None
    return value.split("|")[0].strip() or None


def make_category_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract a main category from `category_list` (t=0 safe)."""
    out = df.copy()

    if "category_list" in out.columns:
        out["category_main"] = out["category_list"].map(_extract_first_category)
    else:
        out["category_main"] = np.nan

    return out


# ---------------------------------------------------------------------
# Feature assembly (STRICT)
# ---------------------------------------------------------------------

def assemble_features(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble a STRICT snapshot-safe feature DataFrame (Option A).

    Forbidden sources:
    - last_funding_at
    - lifetime funding aggregates (funding_total_usd, funding_rounds, etc.)
    """
    out = df.copy()

    # Snapshot-safe time features (only founded_at + first_funding_at used)
    out = make_snapshot_features(out)

    # Geographic features
    out = make_geo_features(out)

    # Category features
    out = make_category_features(out)

    # ------------------------------------------------------------------
    # Drop identifiers + forbidden/leakage-prone columns
    # ------------------------------------------------------------------
    drop_cols = [
        # identifiers
        "permalink", "name", "homepage_url",
        # raw category_list (we keep category_main)
        "category_list",
        # forbidden / post-snapshot / lifetime aggregates
        "last_funding_at",
        "funding_total_usd",
        "funding_rounds",
        "funding_total_usd_num",
        "log_funding_total",
        "funding_per_round",
        "log_funding_per_round",
        "high_total_funding",
        "high_funding_per_round",
        "has_multiple_rounds",
        # other leaky artifacts if present
        "time_between_first_last_days",
        "time_between_first_last_years",
        "company_age_at_last_funding_years",
        "avg_round_interval_years",
        "years_alive",
        "survived_5y",
    ]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])

    return out


__all__ = [
    "make_snapshot_features",
    "make_geo_features",
    "make_category_features",
    "assemble_features",
]
