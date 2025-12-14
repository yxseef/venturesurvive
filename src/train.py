"""Training and experiment orchestration for the venturesurvive project."""

from __future__ import annotations

from typing import Dict

import joblib
import numpy as np
from scipy.stats import randint, uniform
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import RandomizedSearchCV

from . import data as data_mod
from . import evaluate as eval_mod
from . import features as features_mod
from . import models as models_mod
from . import split as split_mod
from .config import MODELS_DIR, RANDOM_STATE


def run_modeling_pipeline(
    cutoff_date: str = "2013-01-01",
    random_state: int = RANDOM_STATE,
    tune_rf: bool = True,
    n_iter_rf: int = 15,
    cv_splits: int = 3,
    save_models: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Run the complete modeling pipeline from cleaned data to trained models."""
    np.random.seed(random_state)

    # ------------------------------------------------------------------
    # 1. Load cleaned dataset
    # ------------------------------------------------------------------
    df_clean = data_mod.load_cleaned_data()
    print(f"✓ Dataset loaded: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")

    # ------------------------------------------------------------------
    # 2. Feature assembly
    # ------------------------------------------------------------------
    df_features = features_mod.assemble_features(df_clean)

    if "success" not in df_features.columns:
        raise KeyError("Target column 'success' not found in dataset")

    if "status" in df_features.columns:
        df_features = df_features.drop(columns=["status"])

    y = df_features["success"].astype(int)
    X = df_features.drop(columns=["success"])

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print(
        f"✓ Features assembled: {len(numeric_cols)} numeric, "
        f"{len(categorical_cols)} categorical"
    )

    # ------------------------------------------------------------------
    # 3. Temporal train / test split
    # ------------------------------------------------------------------
    df_all = X.copy()
    df_all["success"] = y.values

    train_df, test_df = split_mod.temporal_split(
        df_all, cutoff_date=cutoff_date, date_col="first_funding_at"
    )

    date_cols = ["founded_at", "first_funding_at", "last_funding_at"]
    drop_cols = [c for c in date_cols if c in train_df.columns]

    if drop_cols:
        train_df = train_df.drop(columns=drop_cols)
        test_df = test_df.drop(columns=drop_cols)

    y_train = train_df["success"].astype(int)
    y_test = test_df["success"].astype(int)
    X_train = train_df.drop(columns=["success"])
    X_test = test_df.drop(columns=["success"])

    print(f"✓ Temporal split: {X_train.shape[0]:,} train, {X_test.shape[0]:,} test")

    results: Dict[str, Dict[str, float]] = {}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 4. Logistic Regression
    # ------------------------------------------------------------------
    print("\n🔹 Training Logistic Regression...")
    log_reg_pipe = models_mod.make_logistic_regression_pipeline(
        numeric_features=numeric_cols,
        categorical_features=categorical_cols,
        random_state=random_state,
    )
    log_reg_pipe.fit(X_train, y_train)

    y_prob_lr = log_reg_pipe.predict_proba(X_test)[:, 1]
    metrics_lr = eval_mod.evaluate_classification(log_reg_pipe, X_test, y_test)
    metrics_lr["brier"] = float(brier_score_loss(y_test, y_prob_lr))
    results["Logistic Regression"] = metrics_lr

    if save_models:
        joblib.dump(log_reg_pipe, MODELS_DIR / "log_reg_baseline.joblib")

    # ------------------------------------------------------------------
    # 5. Random Forest (baseline)
    # ------------------------------------------------------------------
    print("🔹 Training Random Forest (baseline)...")
    rf_baseline = models_mod.make_random_forest_baseline_pipeline(
        numeric_features=numeric_cols,
        categorical_features=categorical_cols,
        random_state=random_state,
    )
    rf_baseline.fit(X_train, y_train)

    y_prob_rf = rf_baseline.predict_proba(X_test)[:, 1]
    metrics_rf = eval_mod.evaluate_classification(rf_baseline, X_test, y_test)
    metrics_rf["brier"] = float(brier_score_loss(y_test, y_prob_rf))
    results["Random Forest (baseline)"] = metrics_rf

    if save_models:
        joblib.dump(rf_baseline, MODELS_DIR / "random_forest_baseline.joblib")

    # ------------------------------------------------------------------
    # 6. Random Forest tuning (optional)
    # ------------------------------------------------------------------
    if tune_rf:
        print("🔹 Tuning Random Forest hyperparameters...")
        param_distributions = {
            "classifier__n_estimators": randint(80, 250),
            "classifier__max_depth": randint(4, 18),
            "classifier__min_samples_split": randint(2, 40),
            "classifier__min_samples_leaf": randint(1, 25),
            "classifier__max_features": uniform(0.3, 0.7),
            "classifier__class_weight": ["balanced", None],
        }

        tuner = RandomizedSearchCV(
            estimator=rf_baseline,
            param_distributions=param_distributions,
            n_iter=n_iter_rf,
            scoring="roc_auc",
            cv=cv_splits,
            n_jobs=-1,
            random_state=random_state,
            verbose=1,
        )
        tuner.fit(X_train, y_train)

        best_rf = tuner.best_estimator_
        y_prob_rf_tuned = best_rf.predict_proba(X_test)[:, 1]

        metrics_rf_tuned = eval_mod.evaluate_classification(best_rf, X_test, y_test)
        metrics_rf_tuned["brier"] = float(brier_score_loss(y_test, y_prob_rf_tuned))
        results["Random Forest (tuned)"] = metrics_rf_tuned

        if save_models:
            joblib.dump(best_rf, MODELS_DIR / "random_forest_tuned.joblib")

    # ------------------------------------------------------------------
    # 7. LightGBM
    # ------------------------------------------------------------------
    print("🔹 Training LightGBM...")
    try:
        lgbm_pipe = models_mod.make_lightgbm_pipeline(
            numeric_features=numeric_cols,
            categorical_features=categorical_cols,
            random_state=random_state,
        )
        lgbm_pipe.fit(X_train, y_train)

        y_prob_lgbm = lgbm_pipe.predict_proba(X_test)[:, 1]
        metrics_lgbm = eval_mod.evaluate_classification(lgbm_pipe, X_test, y_test)
        metrics_lgbm["brier"] = float(brier_score_loss(y_test, y_prob_lgbm))
        results["LightGBM"] = metrics_lgbm

        if save_models:
            joblib.dump(lgbm_pipe, MODELS_DIR / "lightgbm_baseline.joblib")

    except RuntimeError:
        print("⚠️  LightGBM not installed, skipping.")

    print("\n✓ Modeling pipeline complete!")
    return results


__all__ = ["run_modeling_pipeline"]
