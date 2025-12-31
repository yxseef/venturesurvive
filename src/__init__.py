"""VentureSurvive package for startup success prediction.

Note:
We intentionally avoid importing submodules at package import time.
This prevents optional dependencies (e.g., seaborn in eda) from breaking
the whole package and keeps imports fast and robust.
"""

__all__ = [
    "config",
    "data",
    "preprocess",
    "features",
    "split",
    "models",
    "evaluate",
    "train",
    "utils",
    # "eda" is intentionally not exported by default (optional dependency seaborn)
]
