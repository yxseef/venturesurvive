"""Configuration for the venturesurvive project."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "startups_raw.csv"

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DATA_PATH = PROCESSED_DIR / "startups_clean.csv"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------
# Modeling setup
# ---------------------------------------------------------------------

RANDOM_STATE = 42

# We define the prediction time as a "snapshot" taken after the first N months
# of a startup's life. All features must be derivable using information available
# up to founded_at + SNAPSHOT_MONTHS.
SNAPSHOT_MONTHS = 6

# ---------------------------------------------------------------------
# Ensure required directories exist (side-effect on import is acceptable)
# ---------------------------------------------------------------------

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_PATH",
    "PROCESSED_DIR",
    "PROCESSED_DATA_PATH",
    "MODELS_DIR",
    "RESULTS_DIR",
    "RANDOM_STATE",
    "SNAPSHOT_MONTHS",
]
