"""
Main entry point for the VentureSurvive project.

This script runs the full end-to-end machine learning pipeline:
- loads cleaned data
- assembles features
- performs a temporal train/test split
- trains multiple models
- evaluates their performance
- saves trained models to disk

Usage
-----
python main.py
"""

from __future__ import annotations

from pprint import pprint

from src.train import run_modeling_pipeline
from src.utils import setup_logging


def main() -> None:
    """Run the VentureSurvive modeling pipeline."""
    setup_logging()

    print("=" * 70)
    print("🚀 VentureSurvive — Startup Success Prediction")
    print("=" * 70)

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
            print(f"  {metric:<10s}: {value:.4f}")

    print("\n" + "=" * 70)
    print("✅ Pipeline finished successfully")
    print("   Trained models are saved in the `models/` directory.")
    print("=" * 70)


if __name__ == "__main__":
    main()
