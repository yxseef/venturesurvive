"""Model training utilities for the venturesurvive project."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_logistic(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train a basic logistic regression classifier.

    Uses reasonable defaults suitable for a first baseline.
    """
    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Train a basic random forest classifier for comparison."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model
