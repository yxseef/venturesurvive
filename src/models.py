"""Model and pipeline utilities for the venturesurvive project."""

from __future__ import annotations

from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """Create a ColumnTransformer for preprocessing."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        # ✅ IMPORTANT: do NOT passthrough unknown columns (prevents accidental leakage)
        remainder="drop",
    )


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

def make_logistic_regression_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    *,
    random_state: int = 42,
) -> Pipeline:
    """Logistic Regression pipeline."""
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    clf = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )


def make_random_forest_baseline_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    *,
    random_state: int = 42,
) -> Pipeline:
    """Baseline Random Forest pipeline."""
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )


def make_random_forest_tuned_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    *,
    random_state: int = 42,
) -> Pipeline:
    """Random Forest pipeline with tuned hyperparameters."""
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    clf = RandomForestClassifier(
        n_estimators=225,
        max_depth=9,
        max_features=0.44555916400773216,
        min_samples_split=30,
        min_samples_leaf=4,
        class_weight=None,
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )


def make_lightgbm_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    *,
    random_state: int = 42,
) -> Pipeline:
    """LightGBM pipeline."""
    if LGBMClassifier is None:
        raise RuntimeError("lightgbm is not installed; cannot build LightGBM pipeline.")

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    clf = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )


__all__ = [
    "build_preprocessor",
    "make_logistic_regression_pipeline",
    "make_random_forest_baseline_pipeline",
    "make_random_forest_tuned_pipeline",
    "make_lightgbm_pipeline",
]
