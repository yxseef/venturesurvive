"""Audit utilities for VentureSurvive (snapshot 6-month constraint).

Purpose (academic):
- Quantify cohort / label shift over time (success rate vs snapshot year)
- Compare train vs test distributions under snapshot-aligned temporal split
- Export a table + a plot to results/ for inclusion in the report
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from . import data as data_mod
from . import features as features_mod
from . import split as split_mod
from .config import RESULTS_DIR


def audit_cohort_shift(
    *,
    cutoff_date: str = "2013-01-01",
    min_n_per_year: int = 50,
    save: bool = True,
    show: bool = True,
) -> pd.DataFrame:
    """Compute cohort shift diagnostics by snapshot year.

    Returns a DataFrame with:
    - n_train, success_rate_train
    - n_test,  success_rate_test
    - n_all,   success_rate_all
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df_clean = data_mod.load_cleaned_data()
    df_feat = features_mod.assemble_features(df_clean)

    if "success" not in df_feat.columns:
        raise KeyError("Missing 'success' in assembled features.")
    if "snapshot_date" not in df_feat.columns:
        raise KeyError("Missing 'snapshot_date' in assembled features.")

    df_feat = df_feat.copy()
    df_feat["snapshot_date"] = pd.to_datetime(df_feat["snapshot_date"], errors="coerce")
    df_feat = df_feat.dropna(subset=["snapshot_date", "success"])

    # Build split frame with only what we need
    df_all = df_feat[["snapshot_date", "success"]].copy()

    train_df, test_df = split_mod.temporal_split_snapshot(
        df_all, snapshot_col="snapshot_date", cutoff_date=cutoff_date
    )

    # Add year
    df_all["snapshot_year"] = df_all["snapshot_date"].dt.year
    train_df["snapshot_year"] = pd.to_datetime(train_df["snapshot_date"]).dt.year
    test_df["snapshot_year"] = pd.to_datetime(test_df["snapshot_date"]).dt.year

    def _year_stats(d: pd.DataFrame, prefix: str) -> pd.DataFrame:
        g = d.groupby("snapshot_year")["success"].agg(["size", "mean"]).rename(
            columns={"size": f"n_{prefix}", "mean": f"success_rate_{prefix}"}
        )
        return g

    stats_all = _year_stats(df_all, "all")
    stats_train = _year_stats(train_df, "train")
    stats_test = _year_stats(test_df, "test")

    out = stats_all.join(stats_train, how="left").join(stats_test, how="left")

    # Filter tiny years (optional but helps avoid noisy interpretation)
    out = out[out["n_all"] >= min_n_per_year].sort_index()

    if save:
        csv_path = RESULTS_DIR / "cohort_shift_by_year.csv"
        out.reset_index().to_csv(csv_path, index=False)
        print(f"✓ Saved cohort shift table to {csv_path}")

    # Plot (train vs test success rate over time)
    fig, ax = plt.subplots(figsize=(10, 6))

    if "success_rate_train" in out.columns:
        ax.plot(out.index, out["success_rate_train"], marker="o", label="Train success rate")
    if "success_rate_test" in out.columns:
        ax.plot(out.index, out["success_rate_test"], marker="o", label="Test success rate")

    ax.set_title("Cohort shift diagnostic: success rate by snapshot year")
    ax.set_xlabel("Snapshot year (founded_at + 6 months)")
    ax.set_ylabel("Mean success rate")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save:
        png_path = RESULTS_DIR / "cohort_shift_by_year.png"
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved cohort shift plot to {png_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff-date", type=str, default="2013-01-01")
    parser.add_argument("--min-n-per-year", type=int, default=50)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    audit_cohort_shift(
        cutoff_date=args.cutoff_date,
        min_n_per_year=args.min_n_per_year,
        save=not args.no_save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
