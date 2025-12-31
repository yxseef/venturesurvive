"""Training and experiment orchestration for the venturesurvive project."""

from __future__ import annotations

from typing import Dict

import joblib
import numpy as np
import warnings
from scipy.stats import randint, uniform
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

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
    """Run the complete modeling pipeline from cleaned data to trained models.

    Snapshot definition:
    - Features are computed under a 6-month snapshot constraint in src/features.py.
    - Temporal split is aligned with snapshot_date (founded_at + 6 months).
    """
    np.random.seed(random_state)

    # ------------------------------------------------------------------
    # 1. Load cleaned dataset
    # ------------------------------------------------------------------
    df_clean = data_mod.load_cleaned_data()
    print(f"✓ Dataset loaded: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")

    # ------------------------------------------------------------------
    # 2. Feature assembly (6-month snapshot-safe)
    # ------------------------------------------------------------------
    df_features = features_mod.assemble_features(df_clean)

    if "success" not in df_features.columns:
        raise KeyError("Target column 'success' not found in dataset")

    # Never use status as feature
    if "status" in df_features.columns:
        df_features = df_features.drop(columns=["status"])

    y = df_features["success"].astype(int)
    X = df_features.drop(columns=["success"])

    # We need snapshot_date for temporal split
    if "snapshot_date" not in X.columns:
        raise KeyError(
            "Column 'snapshot_date' not found after feature assembly. "
            "It is required for snapshot-aligned temporal split."
        )

    print("✓ Features assembled under snapshot constraint")

    # ------------------------------------------------------------------
    # 3. Temporal train / test split aligned with snapshot_date
    # ------------------------------------------------------------------
    df_all = X.copy()
    df_all["success"] = y.values

    train_df, test_df = split_mod.temporal_split_snapshot(
        df_all, snapshot_col="snapshot_date", cutoff_date=cutoff_date
    )

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Temporal split produced an empty train or test set. "
            "Adjust cutoff_date or check snapshot_date availability."
        )

    # Sort train/test chronologically (important for TimeSeriesSplit + clean reporting)
    train_df = train_df.sort_values("snapshot_date").reset_index(drop=True)
    test_df = test_df.sort_values("snapshot_date").reset_index(drop=True)

    # ✅ RIGOR CHECK: ensure chronological order is preserved (prevents silent CV leakage)
    if "snapshot_date" in train_df.columns:
        assert train_df["snapshot_date"].is_monotonic_increasing, (
            "train_df is not sorted by snapshot_date; TimeSeriesSplit would be invalid."
        )

    # Drop date columns from modeling matrices (keep only engineered/categorical features)
    date_cols = ["founded_at", "first_funding_at", "snapshot_date"]
    drop_cols = [c for c in date_cols if c in train_df.columns]

    if drop_cols:
        train_df = train_df.drop(columns=drop_cols)
        test_df = test_df.drop(columns=drop_cols)

    y_train = train_df["success"].astype(int)
    y_test = test_df["success"].astype(int)
    X_train = train_df.drop(columns=["success"])
    X_test = test_df.drop(columns=["success"])

    # ------------------------------------------------------------------
    # ✅ STRICT leakage guard (Option A)
    # ------------------------------------------------------------------
    forbidden_exact = {
        "last_funding_at",
        "years_alive",
        "survived_5y",
        "funding_total_usd",
        "funding_rounds",
        "funding_total_usd_num",
        "log_funding_total",
        "funding_per_round",
        "log_funding_per_round",
        "high_total_funding",
        "high_funding_per_round",
        "has_multiple_rounds",
        "time_between_first_last_days",
        "company_age_at_last_funding_years",
        "avg_round_interval_years",
    }
    bad_cols = sorted([c for c in X_train.columns if c in forbidden_exact])
    if bad_cols:
        raise ValueError(f"Leakage columns detected (Option A strict): {bad_cols}")

    # ------------------------------------------------------------------
    # Quick class balance checks (report-friendly)
    # ------------------------------------------------------------------
    pos_rate_train = float(y_train.mean()) if len(y_train) else float("nan")
    pos_rate_test = float(y_test.mean()) if len(y_test) else float("nan")
    print(
        f"✓ Class balance: success=1 rate "
        f"(train={pos_rate_train:.3f}, test={pos_rate_test:.3f})"
    )

    baseline_acc = float((y_test == 0).mean()) if len(y_test) else float("nan")
    print(f"✓ Baseline (always 0) accuracy on test: {baseline_acc:.4f}")

    # Recompute feature lists AFTER split & dropping dates (most robust)
    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    print(
        f"✓ Temporal split (snapshot-aligned): "
        f"{X_train.shape[0]:,} train, {X_test.shape[0]:,} test"
    )
    print(f"✓ Modeling columns: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")

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
    # 6. Random Forest tuning (optional) with TimeSeriesSplit
    # ------------------------------------------------------------------
    if tune_rf:
        print("🔹 Tuning Random Forest hyperparameters (TimeSeriesSplit)...")

        param_distributions = {
            "classifier__n_estimators": randint(80, 250),
            "classifier__max_depth": randint(4, 18),
            "classifier__min_samples_split": randint(2, 40),
            "classifier__min_samples_leaf": randint(1, 25),
            "classifier__max_features": uniform(0.3, 0.7),
            "classifier__class_weight": ["balanced", None],
        }

        tscv = TimeSeriesSplit(n_splits=cv_splits)

        tuner = RandomizedSearchCV(
            estimator=rf_baseline,
            param_distributions=param_distributions,
            n_iter=n_iter_rf,
            scoring="roc_auc",
            cv=tscv,
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

        # Silence harmless sklearn warning about feature names with LGBM + transformed matrices
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*X does not have valid feature names, but LGBMClassifier was fitted with feature names.*",
                category=UserWarning,
                module=r"sklearn\.utils\.validation",
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
