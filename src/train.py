"""
Training and experiment orchestration for the venturesurvive project.

This module runs the full pipeline:
- load cleaned data
- build snapshot-safe features (6-month rule)
- split train/test using snapshot_date
- train a few models and compare them
- save metrics and (optionally) models
"""

import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from . import data as data_mod
from . import evaluate as eval_mod
from . import features as features_mod
from . import models as models_mod
from . import split as split_mod
from .config import MODELS_DIR, RESULTS_DIR, RANDOM_STATE


def baseline_metrics(y_true, y_pred, y_score):
    """
    Simple metric computation without needing the model object.

    Note: ROC-AUC / PR-AUC can fail if predictions are constant.
    In that case, we return a reasonable default baseline.
    """
    from sklearn import metrics

    out = {}
    out["accuracy"] = float(metrics.accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(metrics.balanced_accuracy_score(y_true, y_pred))
    out["precision"] = float(metrics.precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(metrics.recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(metrics.f1_score(y_true, y_pred, zero_division=0))
    out["mcc"] = float(metrics.matthews_corrcoef(y_true, y_pred))

    try:
        out["roc_auc"] = float(metrics.roc_auc_score(y_true, y_score))
    except Exception:
        out["roc_auc"] = 0.5

    try:
        out["pr_auc"] = float(metrics.average_precision_score(y_true, y_score))
    except Exception:
        out["pr_auc"] = float(np.mean(y_true)) if len(y_true) else float("nan")

    try:
        out["brier"] = float(brier_score_loss(y_true, y_score))
    except Exception:
        out["brier"] = float("nan")

    return out


def print_residual_shift(train_df, test_df, snapshot_col="snapshot_date", y_col="success", save_csv=True):
    """
    Small, report-friendly diagnostic:
    shows whether success rate changes a lot across years (train vs test).
    """
    if snapshot_col not in train_df.columns or snapshot_col not in test_df.columns:
        print("⚠️ Residual shift diagnostic skipped: missing snapshot_date.")
        return
    if y_col not in train_df.columns or y_col not in test_df.columns:
        print("⚠️ Residual shift diagnostic skipped: missing target column.")
        return

    tr_dates = pd.to_datetime(train_df[snapshot_col], errors="coerce")
    te_dates = pd.to_datetime(test_df[snapshot_col], errors="coerce")
    tr_y = pd.to_numeric(train_df[y_col], errors="coerce")
    te_y = pd.to_numeric(test_df[y_col], errors="coerce")

    print("\n================= Residual shift diagnostic =================")
    print(
        f"Train range: {tr_dates.min().date() if pd.notna(tr_dates.min()) else 'NaT'}"
        f" -> {tr_dates.max().date() if pd.notna(tr_dates.max()) else 'NaT'}"
        f" (n={len(train_df):,})"
    )
    print(
        f"Test  range: {te_dates.min().date() if pd.notna(te_dates.min()) else 'NaT'}"
        f" -> {te_dates.max().date() if pd.notna(te_dates.max()) else 'NaT'}"
        f" (n={len(test_df):,})"
    )
    print(f"Success rate: train={float(np.nanmean(tr_y)):.3f} | test={float(np.nanmean(te_y)):.3f}")

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

    # print last ~12 years if there are many
    merged_show = merged.tail(12) if len(merged) > 12 else merged

    def fmt(x):
        return f"{x:.3f}" if pd.notna(x) else ""

    if not merged_show.empty:
        show = merged_show.copy()
        show["rate_train"] = show["rate_train"].map(fmt)
        show["rate_test"] = show["rate_test"].map(fmt)
        show["delta_rate_test_minus_train"] = show["delta_rate_test_minus_train"].map(fmt)

        print("\nSuccess rate by snapshot year (train vs test):")
        print(show.to_string(index=False))

    print("==============================================================\n")


def psi_numeric(train_s, test_s, n_bins=10, eps=1e-6):
    """
    Simple PSI (Population Stability Index) for numeric series.
    Bins come from TRAIN quantiles.
    """
    train_s = pd.to_numeric(train_s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    test_s = pd.to_numeric(test_s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

    if train_s.empty or test_s.empty:
        return float("nan")
    if train_s.nunique() <= 1:
        return 0.0

    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(train_s.quantile(qs).values)

    if len(edges) < 3:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    train_bins = pd.cut(train_s, bins=edges, include_lowest=True)
    test_bins = pd.cut(test_s, bins=edges, include_lowest=True)

    train_dist = train_bins.value_counts(normalize=True).sort_index()
    test_dist = test_bins.value_counts(normalize=True).sort_index()

    idx = train_dist.index.union(test_dist.index)
    train_p = train_dist.reindex(idx, fill_value=0.0).values
    test_p = test_dist.reindex(idx, fill_value=0.0).values

    train_p = np.clip(train_p, eps, 1.0)
    test_p = np.clip(test_p, eps, 1.0)

    return float(np.sum((test_p - train_p) * np.log(test_p / train_p)))


def compute_numeric_psi_table(X_train, X_test, n_bins=10):
    """PSI for all numeric columns; returns a sorted table."""
    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    rows = []
    for c in numeric_cols:
        rows.append({"feature": c, "psi": psi_numeric(X_train[c], X_test[c], n_bins=n_bins)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def try_save_diagnostic_plots(psi_table):
    """
    Optional: saves a couple of simple plots if matplotlib is installed.
    This should never crash the pipeline.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("⚠️ matplotlib not installed; skipping plots.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # residual shift plot (from CSV if available)
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
            plt.title("Success rate by year (train vs test)")
            plt.legend()
            out_path = RESULTS_DIR / "residual_shift_by_year.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"✓ Saved plot to {out_path}")
    except Exception as e:
        print(f"⚠️ Could not save residual shift plot: {e}")

    # top PSI plot
    try:
        if psi_table is not None and not psi_table.empty:
            top = psi_table.dropna(subset=["psi"]).head(10).copy()
            top = top.iloc[::-1]

            plt.figure()
            plt.barh(top["feature"], top["psi"])
            plt.xlabel("PSI")
            plt.title("Top numeric drift (PSI)")
            out_path = RESULTS_DIR / "psi_top_numeric.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"✓ Saved plot to {out_path}")
    except Exception as e:
        print(f"⚠️ Could not save PSI plot: {e}")


def run_modeling_pipeline(
    cutoff_date="2013-01-01",
    random_state=RANDOM_STATE,
    tune_rf=True,
    n_iter_rf=15,
    cv_splits=3,
    save_models=True,
    save_metrics=True,
):
    """
    Main function called by main.py.
    Trains baseline models + optional tuning and exports results.
    """
    np.random.seed(random_state)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) load cleaned data
    df_clean = data_mod.load_cleaned_data()
    print(f"✓ Dataset loaded: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")

    # 2) build features (should respect 6-month snapshot constraint inside features.py)
    df_features = features_mod.assemble_features(df_clean)

    if "success" not in df_features.columns:
        raise KeyError("Target column 'success' not found in dataset")

    # never use status as feature
    if "status" in df_features.columns:
        df_features = df_features.drop(columns=["status"])

    y = df_features["success"].astype(int)
    X = df_features.drop(columns=["success"])

    if "snapshot_date" not in X.columns:
        raise KeyError("Missing 'snapshot_date' after feature assembly (needed for temporal split).")

    print("✓ Features assembled")

    # 3) temporal split using snapshot_date
    df_all = X.copy()
    df_all["success"] = y.values

    train_df, test_df = split_mod.temporal_split_snapshot(
        df_all, snapshot_col="snapshot_date", cutoff_date=cutoff_date
    )

    if train_df.empty or test_df.empty:
        raise ValueError("Temporal split produced an empty train or test set. Try another cutoff_date.")

    # keep chronological order (useful for TimeSeriesSplit)
    train_df = train_df.sort_values("snapshot_date").reset_index(drop=True)
    test_df = test_df.sort_values("snapshot_date").reset_index(drop=True)

    # quick diagnostic before dropping dates
    print_residual_shift(train_df, test_df, snapshot_col="snapshot_date", y_col="success", save_csv=True)

    # IMPORTANT: snapshot_date is used only to split. We drop it before training.
    date_cols = ["founded_at", "first_funding_at", "snapshot_date"]
    drop_cols = [c for c in date_cols if c in train_df.columns]
    if drop_cols:
        train_df = train_df.drop(columns=drop_cols)
        test_df = test_df.drop(columns=drop_cols)

    y_train = train_df["success"].astype(int).to_numpy()
    y_test = test_df["success"].astype(int).to_numpy()
    X_train = train_df.drop(columns=["success"])
    X_test = test_df.drop(columns=["success"])

    # very simple anti-leak check
    if "snapshot_date" in X_train.columns or "snapshot_date" in X_test.columns:
        raise ValueError("Leak detected: snapshot_date is still in features.")

    # covariate shift: PSI on numeric features
    psi_table = compute_numeric_psi_table(X_train, X_test, n_bins=10)
    psi_path = RESULTS_DIR / "covariate_shift_psi_numeric.csv"
    psi_table.to_csv(psi_path, index=False)
    print(f"✓ Saved covariate shift PSI table to {psi_path}")

    if not psi_table.empty:
        print("\nTop numeric drift (PSI):")
        print(psi_table.head(10).to_string(index=False))

    try_save_diagnostic_plots(psi_table)

    # class balance info
    pos_rate_train = float(y_train.mean()) if len(y_train) else float("nan")
    pos_rate_test = float(y_test.mean()) if len(y_test) else float("nan")
    print(f"✓ Class balance: train={pos_rate_train:.3f}, test={pos_rate_test:.3f}")

    # Baseline A: always 0
    results = {}
    y_pred_zero = np.zeros_like(y_test)
    y_score_zero = np.zeros_like(y_test, dtype=float)
    results["Baseline (always 0)"] = baseline_metrics(y_test, y_pred_zero, y_score_zero)

    # Baseline B: constant prob = train prevalence
    y_score_prev = np.full_like(y_test, fill_value=pos_rate_train, dtype=float)
    y_pred_prev = (y_score_prev >= 0.5).astype(int)
    results["Baseline (train prevalence prob)"] = baseline_metrics(y_test, y_pred_prev, y_score_prev)

    print(f"✓ Baseline (always 0) accuracy: {results['Baseline (always 0)']['accuracy']:.4f}")
    print(f"✓ Baseline PR-AUC (≈ prevalence): {results['Baseline (train prevalence prob)']['pr_auc']:.4f}")

    # feature types for preprocessing
    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"✓ Split: {X_train.shape[0]:,} train, {X_test.shape[0]:,} test")
    print(f"✓ Columns: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")

    # 4) Logistic Regression
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

    # 5) Random Forest (baseline)
    print("🔹 Training Random Forest (baseline)...")
    rf_pipe = models_mod.make_random_forest_baseline_pipeline(
        numeric_features=numeric_cols,
        categorical_features=categorical_cols,
        random_state=random_state,
    )
    rf_pipe.fit(X_train, y_train)

    y_prob_rf = rf_pipe.predict_proba(X_test)[:, 1]
    metrics_rf = eval_mod.evaluate_classification(rf_pipe, X_test, y_test)
    metrics_rf["brier"] = float(brier_score_loss(y_test, y_prob_rf))
    results["Random Forest (baseline)"] = metrics_rf

    if save_models:
        joblib.dump(rf_pipe, MODELS_DIR / "random_forest_baseline.joblib")

    # 6) Random Forest tuning (optional)
    if tune_rf:
        print("🔹 Tuning Random Forest (RandomizedSearchCV + TimeSeriesSplit)...")

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
            estimator=rf_pipe,
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

    # 7) LightGBM (optional)
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
        print("⚠️ LightGBM not installed, skipping.")

    # 8) export metrics
    if save_metrics:
        df_out = pd.DataFrame(results).T
        csv_path = RESULTS_DIR / "metrics_summary.csv"
        df_out.to_csv(csv_path, index=True)
        print(f"✓ Saved metrics table to {csv_path}")

    print("\n✓ Modeling pipeline complete!")
    return results
