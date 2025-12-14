"""Exploratory data analysis utilities for the venturesurvive project."""

from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _finalize_plot(show: bool = True, save_path: Optional[str] = None) -> None:
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------
# Core summaries
# ---------------------------------------------------------------------

def summarize_dataset(df: pd.DataFrame) -> None:
    """Print dataset shape, columns, info, and basic statistics."""
    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumn names:\n{list(df.columns)}")
    print("\n" + "=" * 60)
    print("Dataset Information:")
    print("=" * 60)
    df.info()
    print("\n" + "=" * 60)
    print("Basic Statistics (numeric columns):")
    print("=" * 60)
    print(df.describe())


# ---------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------

def plot_status_distribution(
    df: pd.DataFrame,
    status_col: str = "status",
    *,
    show: bool = True,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Plot and return the distribution of startup statuses."""
    if status_col not in df.columns:
        return pd.DataFrame()

    status_counts = df[status_col].value_counts()
    status_pct = df[status_col].value_counts(normalize=True) * 100

    status_df = pd.DataFrame(
        {"Count": status_counts, "Percentage": status_pct.round(2)}
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    status_counts.plot(kind="bar", ax=ax, edgecolor="black")
    ax.set_title("Distribution of Startup Status (Raw Data)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Status")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    _finalize_plot(show=show, save_path=save_path)

    return status_df


def plot_years_alive_distribution(
    df: pd.DataFrame,
    years_col: str = "years_alive",
    *,
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """Plot the distribution of years_alive (restricted to [0, 40] years)."""
    if years_col not in df.columns:
        return

    years_alive = df[years_col].dropna()
    years_alive = years_alive[(years_alive >= 0) & (years_alive <= 40)]

    plt.figure(figsize=(10, 6))
    sns.histplot(years_alive, bins=50)
    plt.title("Distribution of Years Alive (Cleaned)", fontsize=14, fontweight="bold")
    plt.xlabel("Years alive")
    plt.ylabel("Count")
    plt.tight_layout()
    _finalize_plot(show=show, save_path=save_path)


def plot_success_distribution(
    df: pd.DataFrame,
    success_col: str = "success",
    *,
    show: bool = True,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Plot and return the distribution of the binary success target."""
    if success_col not in df.columns:
        return pd.DataFrame()

    success_counts = df[success_col].value_counts(dropna=False).sort_index()
    success_pct = (
        df[success_col].value_counts(normalize=True, dropna=False).sort_index() * 100
    )

    success_df = pd.DataFrame(
        {"Count": success_counts, "Percentage": success_pct.round(2)}
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    success_counts.plot(kind="bar", ax=ax, edgecolor="black")
    ax.set_title("Distribution of Success Target", fontsize=14, fontweight="bold")
    ax.set_xlabel("Success (0 = no, 1 = yes)")
    ax.set_ylabel("Count")
    ax.set_xticklabels(["0", "1"], rotation=0)
    plt.tight_layout()
    _finalize_plot(show=show, save_path=save_path)

    return success_df


# ---------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------

def plot_funding_vs_success(
    df: pd.DataFrame,
    *,
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """Boxplot of total funding vs success (log scale)."""
    required = {"success", "funding_total_usd"}
    if not required.issubset(df.columns):
        return

    df_plot = df.copy()
    df_plot["funding_total_usd_num"] = pd.to_numeric(
        df_plot["funding_total_usd"], errors="coerce"
    )

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_plot.dropna(subset=["funding_total_usd_num", "success"]),
        x="success",
        y="funding_total_usd_num",
    )
    plt.yscale("log")
    plt.title("Funding vs. Success (log scale)", fontsize=14, fontweight="bold")
    plt.xlabel("Success (0 = no, 1 = yes)")
    plt.ylabel("Total funding (USD, log scale)")
    plt.tight_layout()
    _finalize_plot(show=show, save_path=save_path)


def plot_region_success_rate(
    df: pd.DataFrame,
    *,
    min_startups: int = 50,
    top_n: int = 15,
    show: bool = True,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Plot success rate by region (top N by sample size)."""
    required = {"success", "region"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    region_stats = (
        df.dropna(subset=["region", "success"])
        .groupby("region")
        .agg(success_rate=("success", "mean"), n_startups=("success", "size"))
    )
    region_stats = region_stats[region_stats["n_startups"] >= min_startups]

    top_regions = (
        region_stats.sort_values("n_startups", ascending=False)
        .head(top_n)
        .sort_values("success_rate", ascending=False)
    )

    plt.figure(figsize=(12, 6))
    top_regions["success_rate"].plot(kind="bar", edgecolor="black")
    plt.title(
        f"Success Rate by Region (Top {top_n}, min {min_startups})",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Region")
    plt.ylabel("Mean success rate")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _finalize_plot(show=show, save_path=save_path)

    return top_regions


# ---------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------

def analyze_missing_values(
    df: pd.DataFrame,
    *,
    show: bool = True,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Analyze and plot missing values."""
    missing_counts = df.isna().sum()
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)

    missing_df = (
        pd.DataFrame(
            {"Missing_Count": missing_counts, "Missing_Percentage": missing_pct}
        )
        .sort_values("Missing_Percentage", ascending=False)
        .query("Missing_Count > 0")
    )

    if not missing_df.empty:
        plt.figure(figsize=(12, 6))
        missing_df["Missing_Percentage"].plot(kind="barh", edgecolor="black")
        plt.title("Missing Values by Feature (% of total)", fontsize=14, fontweight="bold")
        plt.xlabel("Percentage Missing (%)")
        plt.ylabel("Feature")
        plt.tight_layout()
        _finalize_plot(show=show, save_path=save_path)

    return missing_df


def plot_numeric_distributions(
    df: pd.DataFrame,
    numeric_cols: Iterable[str],
    *,
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """Plot histograms for numeric features."""
    cols = [c for c in numeric_cols if c in df.columns]
    if not cols:
        return

    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, cols):
        df[col].dropna().hist(bins=50, ax=ax, edgecolor="black")
        ax.set_title(f"{col}", fontsize=11, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

    for ax in axes[len(cols):]:
        fig.delaxes(ax)

    plt.tight_layout()
    _finalize_plot(show=show, save_path=save_path)


# ---------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------

__all__ = [
    "summarize_dataset",
    "plot_status_distribution",
    "plot_years_alive_distribution",
    "plot_success_distribution",
    "plot_funding_vs_success",
    "plot_region_success_rate",
    "analyze_missing_values",
    "plot_numeric_distributions",
]
