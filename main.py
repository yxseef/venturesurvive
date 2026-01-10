"""
Main entry point for the VentureSurvive project.

This script runs the full machine learning pipeline:
- optionally rebuilds the cleaned dataset
- chooses a cutoff date for the temporal split
- trains several models
- prints and saves evaluation results

Run with:
    python main.py
"""

import pandas as pd

from src.config import PROCESSED_DATA_PATH
from src.train import run_modeling_pipeline

# Set to False once preprocessing is stable
REBUILD_CLEAN_DATASET = True

# Automatically choose a cutoff date (to avoid empty train/test splits)
AUTO_CUTOFF = True
CUTOFF_QUANTILE = 0.80  # roughly 80% train / 20% test


def auto_cutoff_date(processed_csv_path, quantile=0.80):
    """
    Choose a cutoff date based on a quantile of snapshot_date.
    This helps ensure that both train and test sets are non-empty.
    """
    df = pd.read_csv(processed_csv_path)

    if "snapshot_date" not in df.columns:
        raise ValueError("snapshot_date not found in processed dataset")

    dates = pd.to_datetime(df["snapshot_date"], errors="coerce").dropna().sort_values()
    if dates.empty:
        raise ValueError("No valid snapshot_date values found")

    cutoff = pd.to_datetime(dates.quantile(quantile)).normalize()

    # small safety checks
    if cutoff <= dates.iloc[0]:
        cutoff = dates.iloc[int(0.2 * len(dates))]
    if cutoff >= dates.iloc[-1]:
        cutoff = dates.iloc[int(0.8 * len(dates))]

    return cutoff.date().isoformat()


def main():
    print("=" * 60)
    print("🚀 VentureSurvive — Startup Success Prediction")
    print("Snapshot: first 6 months of startup life")
    print("=" * 60)

    # Optional: rebuild cleaned dataset
    if REBUILD_CLEAN_DATASET:
        from src.preprocess import build_clean_dataset

        print("\n🔧 Rebuilding cleaned dataset...")
        build_clean_dataset(save=True)

    # Choose cutoff date
    if AUTO_CUTOFF:
        cutoff_date = auto_cutoff_date(
            str(PROCESSED_DATA_PATH), quantile=CUTOFF_QUANTILE
        )
        print(f"✓ Using automatic cutoff_date: {cutoff_date}")
    else:
        cutoff_date = "2013-01-01"
        print(f"✓ Using fixed cutoff_date: {cutoff_date}")

    # Run full modeling pipeline
    results = run_modeling_pipeline(
        cutoff_date=cutoff_date,
        tune_rf=True,
        n_iter_rf=15,
        cv_splits=3,
        save_models=True,
    )

    # Print results nicely
    print("\n" + "=" * 60)
    print("📊 Model Performance Summary")
    print("=" * 60)

    for model_name, metrics in results.items():
        print(f"\n🔹 {model_name}")
        for metric, value in metrics.items():
            try:
                print(f"  {metric:<12s}: {value:.4f}")
            except Exception:
                print(f"  {metric:<12s}: {value}")

    print("\n" + "=" * 60)
    print("✅ Pipeline finished successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
