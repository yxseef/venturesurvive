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

from src.train import run_modeling_pipeline
from src.utils import setup_logging

# Optional: rebuild the cleaned dataset (useful after preprocessing changes)
REBUILD_CLEAN_DATASET = False


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
        build_clean_dataset(save=True)

    results = run_modeling_pipeline(
        cutoff_date="2013-01-01",
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
            # avoid formatting issues if NaN
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
