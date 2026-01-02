""""Training and experiment orchestration for the venturesurvive project."""

from __future__ import annotations

from typing import Dict

import joblib
import numpy as np
import pandas as pd
import warnings
from scipy.stats import randint, uniform
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from . import data as data_mod
from . import evaluate as eval_mod
from . import features as features_mod
from . import models as models_mod
from . import split as split_mod
from .config import MODELS_DIR, RESULTS_DIR, RANDOM_STATE


def _baseline_metrics(y_true: np.ndarray, *, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Compute baseline metrics without a model object."""
    from sklearn import metrics

    out: Dict[str, float] = {
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(metrics.balanced_accuracy_score(y_true, y_pred)),
        "precision": float(metrics.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(metrics.recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(metrics.f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(metrics.matthews_corrcoef(y_true, y_pred)),
    }

    # ROC-AUC: constant scores can raise; use 0.5 as baseline if undefined
    try:
        out["roc_auc"] = float(metrics.roc_auc_score(y_true, y_score))
    except Exception:
        out["roc_auc"] = 0.5

    # PR-AUC (Average Precision): for constant scores, AP equals prevalence
    try:
        out["pr_auc"] = float(metrics.average_precision_score(y_true, y_score))
    except Exception:
        out["pr_auc"] = float(np.mean(y_true)) if len(y_true) else float("nan")

    # Brier score (probability calibration)
    try:
        out["brier"] = float(brier_score_loss(y_true, y_score))
    except Exception:
        out["brier"] = float("nan")

    return out


def _print_residual_shift_diagnostic(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    snapshot_col: str = "snapshot_date",
    y_col: str = "success",
    show_last_n_years: int = 12,
    save_csv: bool = True,
) -> None:
    """Print + (optionally) export a report-friendly 'residual shift' diagnostic.

    What it prints:
    - Train/test snapshot_date ranges
    - Train/test success rates
    - Success rate by snapshot year (train vs test), plus counts

    What it saves (if save_csv=True):
    - RESULTS_DIR/residual_shift_by_year.csv (full table across all years)
    """
    if snapshot_col not in train_df.columns or snapshot_col not in test_df.columns:
        print("⚠️  Residual shift diagnostic skipped: missing snapshot_date in train/test.")
        return
    if y_col not in train_df.columns or y_col not in test_df.columns:
        print("⚠️  Residual shift diagnostic skipped: missing target column in train/test.")
        return

    tr_dates = pd.to_datetime(train_df[snapshot_col], errors="coerce")
    te_dates = pd.to_datetime(test_df[snapshot_col], errors="coerce")

    tr_min = tr_dates.min()
    tr_max = tr_dates.max()
    te_min = te_dates.min()
    te_max = te_dates.max()

    tr_y = pd.to_numeric(train_df[y_col], errors="coerce")
    te_y = pd.to_numeric(test_df[y_col], errors="coerce")

    print("\n================= Residual shift diagnostic =================")
    print(
        f"Train {snapshot_col} range: "
        f"{tr_min.date() if pd.notna(tr_min) else 'NaT'} -> {tr_max.date() if pd.notna(tr_max) else 'NaT'} "
        f"(n={len(train_df):,})"
    )
    print(
        f"Test  {snapshot_col} range: "
        f"{te_min.date() if pd.notna(te_min) else 'NaT'} -> {te_max.date() if pd.notna(te_max) else 'NaT'} "
        f"(n={len(test_df):,})"
    )
    print(
        f"Success rate: train={float(np.nanmean(tr_y)):.3f} | test={float(np.nanmean(te_y)):.3f}"
    )

    tr_tbl = (
        pd.DataFrame({"year": tr_dates.dt.year, "y": tr_y})
        .dropna(subset=["year", "y"])
        .groupby("year", as_index=False)
        .agg(n_train=("y", "size"), rate_train=("y", "mean"))
        .sort_values("year")
    )
    te_tbl = (
        pd.DataFrame({"year": te_dates.dt.year, "y": te_y})
        .dropna(subset=["year", "y"])
        .groupby("year", as_index=False)
        .agg(n_test=("y", "size"), rate_test=("y", "mean"))
        .sort_values("year")
    )

    merged = tr_tbl.merge(te_tbl, on="year", how="outer").sort_values("year")
    merged["delta_rate_test_minus_train"] = merged["rate_test"] - merged["rate_train"]

    if save_csv:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "residual_shift_by_year.csv"
        merged.to_csv(out_path, index=False)
        print(f"✓ Saved residual shift table to {out_path}")

    # show last N years only
    merged_show = merged.tail(show_last_n_years).copy() if len(merged) > show_last_n_years else merged.copy()

    if not merged_show.empty:
        def fmt(x):
            return f"{x:.3f}" if pd.notna(x) else ""

        merged_show_print = merged_show.copy()
        merged_show_print["rate_train"] = merged_show_print["rate_train"].map(fmt)
        merged_show_print["rate_test"] = merged_show_print["rate_test"].map(fmt)
        merged_show_print["delta_rate_test_minus_train"] = merged_show_print["delta_rate_test_minus_train"].map(fmt)

        print("\nSuccess rate by snapshot year (train vs test):")
        print(merged_show_print.to_string(index=False))

    print("==============================================================\n")


def _psi_for_numeric(train_s: pd.Series, test_s: pd.Series, n_bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index (PSI) for one numeric feature.

    Bins are defined by TRAIN quantiles (common in drift monitoring).
    """
    train_s = pd.to_numeric(train_s, errors="coerce")
    test_s = pd.to_numeric(test_s, errors="coerce")
    train_s = train_s.replace([np.inf, -np.inf], np.nan).dropna()
    test_s = test_s.replace([np.inf, -np.inf], np.nan).dropna()

    if train_s.empty or test_s.empty:
        return float("nan")

    # If constant / near-constant, PSI not meaningful
    if train_s.nunique() <= 1:
        return 0.0

    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(train_s.quantile(qs).values)

    # Need at least 3 edges to form bins
    if len(edges) < 3:
        return 0.0

    # Ensure edges are strictly increasing
    edges[0] = -np.inf
    edges[-1] = np.inf

    train_bins = pd.cut(train_s, bins=edges, include_lowest=True)
    test_bins = pd.cut(test_s, bins=edges, include_lowest=True)

    train_dist = train_bins.value_counts(normalize=True).sort_index()
    test_dist = test_bins.value_counts(normalize=True).sort_index()

    # Align indices and apply epsilon smoothing
    idx = train_dist.index.union(test_dist.index)
    train_p = train_dist.reindex(idx, fill_value=0.0).values
    test_p = test_dist.reindex(idx, fill_value=0.0).values

    train_p = np.clip(train_p, eps, 1.0)
    test_p = np.clip(test_p, eps, 1.0)

    psi = np.sum((test_p - train_p) * np.log(test_p / train_p))
    return float(psi)


def _compute_numeric_psi_table(X_train: pd.DataFrame, X_test: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Compute PSI for all numeric columns and return a sorted table."""
    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    rows = []
    for c in numeric_cols:
        psi = _psi_for_numeric(X_train[c], X_test[c], n_bins=n_bins)
        rows.append({"feature": c, "psi": psi})
    df_psi = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    return df_psi


def _try_save_diagnostic_plots(psi_table: pd.DataFrame) -> None:
    """Optional: save 2 plots into RESULTS_DIR if matplotlib is available.
    - residual_shift_by_year.png
    - psi_top_numeric.png

    This function NEVER breaks the pipeline (fails gracefully).
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("⚠️  matplotlib not installed; skipping diagnostic plots.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Plot 1: residual shift by year (train vs test) from saved CSV
    try:
        residual_path = RESULTS_DIR / "residual_shift_by_year.csv"
        if residual_path.exists():
            dfp = pd.read_csv(residual_path).sort_values("year")
            dfp = dfp[dfp["year"].notna()]
            x = dfp["year"].astype(int)

            plt.figure()
            if "rate_train" in dfp.columns:
                plt.plot(x, dfp["rate_train"], marker="o", label="train")
            if "rate_test" in dfp.columns:
                plt.plot(x, dfp["rate_test"], marker="o", label="test")

            plt.xlabel("snapshot year")
            plt.ylabel("success rate")
            plt.title("Residual label/cohort shift: success rate by year")
            plt.legend()
            out_path = RESULTS_DIR / "residual_shift_by_year.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"✓ Saved plot to {out_path}")
    except Exception as e:
        print(f"⚠️  Could not save residual shift plot: {e}")

    # Plot 2: top PSI numeric
    try:
        if psi_table is not None and not psi_table.empty:
            top = psi_table.dropna(subset=["psi"]).head(10).copy()
            top = top.iloc[::-1]  # nicer order for barh

            plt.figure()
            plt.barh(top["feature"], top["psi"])
            plt.xlabel("PSI")
            plt.title("Top numeric covariate shift (PSI)")
            out_path = RESULTS_DIR / "psi_top_numeric.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"✓ Saved plot to {out_path}")
    except Exception as e:
        print(f"⚠️  Could not save PSI plot: {e}")


def run_modeling_pipeline(
    cutoff_date: str = "2013-01-01",
    random_state: int = RANDOM_STATE,
    tune_rf: bool = True,
    n_iter_rf: int = 15,
    cv_splits: int = 3,
    save_models: bool = True,
    save_metrics: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Run the complete modeling pipeline from cleaned data to trained models.

    Snapshot definition:
    - Features are computed under a 6-month snapshot constraint in src/features.py.
    - Temporal split is aligned with snapshot_date (founded_at + 6 months).
    """
    np.random.seed(random_state)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

    # ✅ RIGOR CHECK: ensure chronological order is preserved
    assert train_df["snapshot_date"].is_monotonic_increasing, (
        "train_df is not sorted by snapshot_date; TimeSeriesSplit would be invalid."
    )

    # ------------------------------------------------------------------
    # ✅ Residual shift diagnostic (label/cohort shift)
    # Runs BEFORE dropping snapshot_date (used for splitting only).
    # ------------------------------------------------------------------
    _print_residual_shift_diagnostic(
        train_df,
        test_df,
        snapshot_col="snapshot_date",
        y_col="success",
        show_last_n_years=12,
        save_csv=True,
    )

    # ------------------------------------------------------------------
    # ✅ CRITICAL: remove date columns BEFORE building X_train/X_test
    # snapshot_date is used ONLY for splitting, never as a feature.
    # ------------------------------------------------------------------
    date_cols = ["founded_at", "first_funding_at", "snapshot_date"]
    drop_cols = [c for c in date_cols if c in train_df.columns]
    if drop_cols:
        train_df = train_df.drop(columns=drop_cols)
        test_df = test_df.drop(columns=drop_cols)

    y_train = train_df["success"].astype(int).to_numpy()
    y_test = test_df["success"].astype(int).to_numpy()
    X_train = train_df.drop(columns=["success"])
    X_test = test_df.drop(columns=["success"])

    # ✅ Anti-leak checks (must never fail)
    assert "snapshot_date" not in X_train.columns, "LEAK: snapshot_date is still in X_train!"
    assert "snapshot_date" not in X_test.columns, "LEAK: snapshot_date is still in X_test!"
    assert "success" not in X_train.columns and "success" not in X_test.columns, "Target leaked into X!"

    # ------------------------------------------------------------------
    # ✅ Covariate shift diagnostic (PSI on numeric features)
    # ------------------------------------------------------------------
    psi_table = _compute_numeric_psi_table(X_train, X_test, n_bins=10)
    psi_path = RESULTS_DIR / "covariate_shift_psi_numeric.csv"
    psi_table.to_csv(psi_path, index=False)
    print(f"✓ Saved covariate shift PSI table to {psi_path}")

    # Print top drifted numeric features (report-friendly)
    top_k = 10
    if not psi_table.empty:
        print("\nTop numeric covariate shift (PSI):")
        print(psi_table.head(top_k).to_string(index=False))

    # Optional: save plots (never breaks)
    _try_save_diagnostic_plots(psi_table)

    # ------------------------------------------------------------------
    # Quick class balance checks (report-friendly)
    # ------------------------------------------------------------------
    pos_rate_train = float(y_train.mean()) if len(y_train) else float("nan")
    pos_rate_test = float(y_test.mean()) if len(y_test) else float("nan")
    print(f"✓ Class balance: success=1 rate (train={pos_rate_train:.3f}, test={pos_rate_test:.3f})")

    # ------------------------------------------------------------------
    # Baselines (IMPORTANT with label shift)
    # ------------------------------------------------------------------
    # Baseline A: always predict 0
    y_pred_zero = np.zeros_like(y_test)
    y_score_zero = np.zeros_like(y_test, dtype=float)
    baseline_zero = _baseline_metrics(y_test, y_pred=y_pred_zero, y_score=y_score_zero)
    print(f"✓ Baseline (always 0) accuracy on test: {baseline_zero['accuracy']:.4f}")

    # Baseline B: constant probability = train prevalence (calibration baseline)
    y_score_prev = np.full_like(y_test, fill_value=pos_rate_train, dtype=float)
    y_pred_prev = (y_score_prev >= 0.5).astype(int)
    baseline_prev = _baseline_metrics(y_test, y_pred=y_pred_prev, y_score=y_score_prev)
    print(f"✓ Baseline PR-AUC on test (≈ prevalence): {baseline_prev['pr_auc']:.4f}")

    # Recompute feature lists AFTER split & dropping dates
    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"✓ Temporal split (snapshot-aligned): {X_train.shape[0]:,} train, {X_test.shape[0]:,} test")
    print(f"✓ Modeling columns: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")

    results: Dict[str, Dict[str, float]] = {}
    results["Baseline (always 0)"] = baseline_zero
    results["Baseline (train prevalence prob)"] = baseline_prev

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

    # ------------------------------------------------------------------
    # Export metrics table (for report)
    # ------------------------------------------------------------------
    if save_metrics:
        df_out = pd.DataFrame(results).T
        csv_path = RESULTS_DIR / "metrics_summary.csv"
        df_out.to_csv(csv_path, index=True)
        print(f"✓ Saved metrics table to {csv_path}")

    print("\n✓ Modeling pipeline complete!")
    return results


__all__ = ["run_modeling_pipeline"]
