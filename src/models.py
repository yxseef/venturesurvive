"""
Model and pipeline utilities for the venturesurvive project.

This file contains:
- a simple preprocessing pipeline (numeric + categorical)
- a few model pipelines (LogReg, RandomForest, LightGBM if installed)
"""

from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# LightGBM is optional (project should still run without it)
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """
    Preprocessing for the models:
    - numeric: median imputation + standard scaling
    - categorical: most_frequent imputation + one-hot encoding

    Note: remainder="drop" to avoid accidentally passing columns we didn't plan to use.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def make_logistic_regression_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    random_state: int = 42,
) -> Pipeline:
    """Logistic Regression + preprocessing."""
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
    random_state: int = 42,
) -> Pipeline:
    """Baseline Random Forest + preprocessing."""
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    clf = RandomForestClassifier(
        n_estimators=300,
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
    random_state: int = 42,
) -> Pipeline:
    """
    Random Forest with fixed "best" hyperparameters (example).
    If you don’t use this pipeline directly in train.py, you can remove it.
    """
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    clf = RandomForestClassifier(
        n_estimators=225,
        max_depth=9,
        max_features=0.45,          # rounded (easier to explain than a long float)
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
    random_state: int = 42,
) -> Pipeline:
    """LightGBM + preprocessing (only if lightgbm is installed)."""
    if LGBMClassifier is None:
        raise RuntimeError("lightgbm is not installed")

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    clf = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
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
