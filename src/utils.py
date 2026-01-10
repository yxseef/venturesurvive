"""General utility functions for the venturesurvive project."""

from __future__ import annotations

import logging


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

