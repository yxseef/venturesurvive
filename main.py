"""
Main entry point for the VentureSurvive project.

This script runs the full end-to-end machine learning pipeline:
- (optionally) rebuilds the cleaned dataset
- loads cleaned data
- assembles snapshot-safe features (first 6 months of life)
- performs a temporal train/test split aligned with snapshot_date
- trains multiple models
- evaluates their performance
- saves trained models to disk

Usage
-----
python main.py
"""

from __future__ import annotations

import pandas as pd

from src.config import PROCESSED_DATA_PATH
from src.train import run_modeling_pipeline
from src.utils import setup_logging

# Optional: rebuild the cleaned dataset (useful after preprocessing changes)
REBUILD_CLEAN_DATASET = True  # set to False once everything is stable

# Auto cutoff to prevent empty train/test splits after eligibility filtering
AUTO_CUTOFF = True
CUTOFF_QUANTILE = 0.80  # 80/20 split


def _auto_cutoff_date_from_processed(processed_csv_path: str, quantile: float = 0.80) -> str:
    """Pick a cutoff_date from snapshot_date quantile to avoid empty train/test splits."""
    df = pd.read_csv(processed_csv_path)

    if "snapshot_date" not in df.columns:
        raise ValueError(
            "snapshot_date is missing in the processed dataset. "
            "Keep it in preprocess.py (do not drop it)."
        )

    s = pd.to_datetime(df["snapshot_date"], errors="coerce").dropna().sort_values()
    if s.empty:
        raise ValueError("snapshot_date has no valid values; cannot compute cutoff_date.")

    cutoff_ts = pd.to_datetime(s.quantile(quantile)).normalize()

    # Safety: ensure cutoff is strictly inside [min, max] so both sets are non-empty
    s_min = s.iloc[0].normalize()
    s_max = s.iloc[-1].normalize()

    if cutoff_ts <= s_min:
        cutoff_ts = s.iloc[max(1, int(0.20 * len(s)))].normalize()
    if cutoff_ts >= s_max:
        cutoff_ts = s.iloc[max(1, int(0.80 * len(s)) - 1)].normalize()

    return cutoff_ts.date().isoformat()


def main() -> None:
    """Run the VentureSurvive modeling pipeline."""
    setup_logging()

    print("=" * 70)
    print("🚀 VentureSurvive — Startup Success Prediction")
    print("   Snapshot definition: first 6 months of startup life")
    print("=" * 70)

    if REBUILD_CLEAN_DATASET:
        from src.preprocess import build_clean_dataset

        print("\n🔧 Rebuilding cleaned dataset...")
        build_clean_dataset(save=True)  # Rebuild with 2-year horizon --> (save=True, target_horizon_years=2)

    # Determine cutoff_date
    if AUTO_CUTOFF:
        cutoff_date = _auto_cutoff_date_from_processed(str(PROCESSED_DATA_PATH), quantile=CUTOFF_QUANTILE)
        print(f"✓ Using auto cutoff_date ({int(CUTOFF_QUANTILE * 100)}/{int((1 - CUTOFF_QUANTILE) * 100)}): {cutoff_date}")
    else:
        cutoff_date = "2013-01-01"

    results = run_modeling_pipeline(
        cutoff_date=cutoff_date,
        tune_rf=True,
        n_iter_rf=15,
        cv_splits=3,
        save_models=True,
    )

    print("\n" + "=" * 70)
    print("📊 Model Performance Summary")
    print("=" * 70)

    for model_name, metrics in results.items():
        print(f"\n🔹 {model_name}")
        for metric, value in metrics.items():
            try:
                print(f"  {metric:<10s}: {value:.4f}")
            except Exception:
                print(f"  {metric:<10s}: {value}")

    print("\n" + "=" * 70)
    print("✅ Pipeline finished successfully")
    print("   Trained models are saved in the `models/` directory.")
    print("=" * 70)


if __name__ == "__main__":
    main()
