"""General utility functions for the venturesurvive project."""

from __future__ import annotations

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Configure basic logging for the project."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,  # ensure reconfiguration in notebooks / repeated runs
    )
__all__ = ["setup_logging"]
