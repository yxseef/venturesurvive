"""
Visualization script for VentureSurvive model evaluation.

This script loads trained models and generates visualizations:
- ROC curves for all models
- Precision-Recall curves
- Confusion matrices
- Model performance comparison
- Feature importance analysis

Snapshot definition:
- Features are computed using only the first 6 months of startup life (see src/features.py).
- Temporal split is aligned with snapshot_date (founded_at + 6 months).
"""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics

from src import data as data_mod
from src import evaluate as eval_mod
from src import features as features_mod
from src import split as split_mod
from src.config import MODELS_DIR, RESULTS_DIR, RANDOM_STATE

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10


def load_models() -> dict:
    """Load all trained models from disk."""
    models = {}

    model_files = {
        "Logistic Regression": "log_reg_baseline.joblib",
        "Random Forest (baseline)": "random_forest_baseline.joblib",
        "Random Forest (tuned)": "random_forest_tuned.joblib",
        "LightGBM": "lightgbm_baseline.joblib",
    }

    for name, filename in model_files.items():
        path = MODELS_DIR / filename
        if path.exists():
            models[name] = joblib.load(path)
            print(f"✓ Loaded {name}")
        else:
            print(f"⚠️  Model not found: {filename}")

    return models


def prepare_data(cutoff_date: str = "2013-01-01"):
    """Load and prepare test data (snapshot-aligned)."""
    print("\n" + "=" * 70)
    print("📊 Preparing data for visualization (snapshot-aligned)")
    print("=" * 70)

    df_clean = data_mod.load_cleaned_data()
    print(f"✓ Dataset loaded: {df_clean.shape[0]:,} rows")

    # Assemble snapshot-safe features
    df_features = features_mod.assemble_features(df_clean)

    if "status" in df_features.columns:
        df_features = df_features.drop(columns=["status"])

    y = df_features["success"].astype(int)
    X = df_features.drop(columns=["success"])

    if "snapshot_date" not in X.columns:
        raise KeyError("snapshot_date missing; required for snapshot-aligned split.")

    df_all = X.copy()
    df_all["success"] = y.values

    train_df, test_df = split_mod.temporal_split_snapshot(
        df_all, snapshot_col="snapshot_date", cutoff_date=cutoff_date
    )

    if test_df.empty:
        raise ValueError("Test set is empty after temporal split. Adjust cutoff_date.")

    # Sort chronologically
    test_df = test_df.sort_values("snapshot_date").reset_index(drop=True)

    # Drop date cols from modeling matrix
    date_cols = ["founded_at", "first_funding_at", "snapshot_date"]
    drop_cols = [c for c in date_cols if c in test_df.columns]
    if drop_cols:
        test_df = test_df.drop(columns=drop_cols)

    y_test = test_df["success"].astype(int)
    X_test = test_df.drop(columns=["success"])

    print(f"✓ Test set: {X_test.shape[0]:,} samples, {X_test.shape[1]} features")
    return X_test, y_test


def plot_roc_comparison(models_dict: dict, X_test, y_test):
    """Plot ROC curves for all models on the same figure."""
    print("\n📈 Generating ROC curves comparison...")

    fig, ax = plt.subplots(figsize=(10, 8))

    for name, model in models_dict.items():
        if not hasattr(model, "predict_proba") and not hasattr(model, "decision_function"):
            print(f"  ⚠️  {name} has no score method; skipping ROC.")
            continue

        # Use evaluate helper to get scores
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        fpr, tpr, _ = metrics.roc_curve(y_test, y_score)
        auc = metrics.roc_auc_score(y_test, y_score)

        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - Model Comparison (Snapshot 6m)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = RESULTS_DIR / "roc_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved to {save_path}")
    plt.show()


def plot_pr_comparison(models_dict: dict, X_test, y_test):
    """Plot Precision-Recall curves for all models."""
    print("\n📈 Generating Precision-Recall curves comparison...")

    fig, ax = plt.subplots(figsize=(10, 8))

    for name, model in models_dict.items():
        if not hasattr(model, "predict_proba") and not hasattr(model, "decision_function"):
            print(f"  ⚠️  {name} has no score method; skipping PR.")
            continue

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        precision, recall, _ = metrics.precision_recall_curve(y_test, y_score)
        ap = metrics.average_precision_score(y_test, y_score)

        ax.plot(recall, precision, label=f"{name} (AP = {ap:.3f})", linewidth=2)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves - Model Comparison (Snapshot 6m)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = RESULTS_DIR / "pr_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved to {save_path}")
    plt.show()


def plot_confusion_matrices(models_dict: dict, X_test, y_test):
    """Plot confusion matrices for all models (normalized)."""
    print("\n📊 Generating confusion matrices...")

    n_models = len(models_dict)
    ncols = 2
    nrows = math.ceil(n_models / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = axes.flatten()

    for idx, (name, model) in enumerate(models_dict.items()):
        ax = axes[idx]
        y_pred = model.predict(X_test)
        cm = metrics.confusion_matrix(y_test, y_pred, normalize="true")

        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(name, fontsize=12, fontweight="bold")

    # Turn off any unused axes
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    save_path = RESULTS_DIR / "confusion_matrices.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved to {save_path}")
    plt.show()


def plot_metrics_comparison(models_dict: dict, X_test, y_test):
    """Create a bar chart comparing key metrics across models."""
    print("\n📊 Generating metrics comparison...")

    metrics_data = []
    for name, model in models_dict.items():
        m = eval_mod.evaluate_classification(model, X_test, y_test)
        m["model"] = name
        metrics_data.append(m)

    df_metrics = pd.DataFrame(metrics_data)
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    # Dynamic layout
    n = len(metric_names)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, metric in enumerate(metric_names):
        ax = axes[i]
        df_metrics.plot(x="model", y=metric, kind="bar", ax=ax, legend=False, edgecolor="black")
        ax.set_title(metric.upper().replace("_", " "), fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis="y")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    save_path = RESULTS_DIR / "metrics_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved to {save_path}")
    plt.show()

    print("\n" + "=" * 70)
    print("📊 Metrics Summary Table")
    print("=" * 70)
    print(df_metrics.to_string(index=False))


def plot_feature_importance_analysis(models_dict: dict, X_test, y_test):
    """Compute and plot permutation feature importance for a chosen model."""
    print("\n🔍 Computing feature importance (permutation)...")

    # Choose a model to explain: prefer tuned RF, else Logistic
    preferred_order = ["Random Forest (tuned)", "Random Forest (baseline)", "Logistic Regression"]
    chosen_name = next((n for n in preferred_order if n in models_dict), None)

    if chosen_name is None:
        print("  ⚠️  No suitable model found for feature importance.")
        return

    model = models_dict[chosen_name]

    importance_df = eval_mod.compute_feature_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
    )

    top_n = 20
    top = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(range(top_n), top["importance_mean"], edgecolor="black")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top["feature"])
    ax.set_xlabel("Permutation Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances - {chosen_name}", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    save_path = RESULTS_DIR / "feature_importance.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved to {save_path}")
    plt.show()

    csv_path = RESULTS_DIR / "feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved importance table to {csv_path}")


def main():
    """Generate all visualizations."""
    print("=" * 70)
    print("📊 VentureSurvive - Model Evaluation Visualizations")
    print("   Snapshot definition: first 6 months of startup life")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n🔧 Loading trained models...")
    models_dict = load_models()

    if not models_dict:
        print("\n❌ No models found. Please run main.py first to train models.")
        return

    # Use the same cutoff logic as main.py
    from src.config import PROCESSED_DATA_PATH
    import pandas as pd
    
    # Auto-cutoff based on snapshot_date quantile (same as main.py)
    df_cutoff = pd.read_csv(PROCESSED_DATA_PATH)
    s = pd.to_datetime(df_cutoff["snapshot_date"], errors="coerce").dropna().sort_values()
    cutoff_ts = pd.to_datetime(s.quantile(0.80)).normalize()
    cutoff_date = cutoff_ts.date().isoformat()
    
    print(f"  ✓ Using auto cutoff_date (80/20): {cutoff_date}")
    
    X_test, y_test = prepare_data(cutoff_date=cutoff_date)

    plot_roc_comparison(models_dict, X_test, y_test)
    plot_pr_comparison(models_dict, X_test, y_test)
    plot_confusion_matrices(models_dict, X_test, y_test)
    plot_metrics_comparison(models_dict, X_test, y_test)
    plot_feature_importance_analysis(models_dict, X_test, y_test)

    print("\n" + "=" * 70)
    print("✅ All visualizations generated successfully!")
    print(f"   Results saved in: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
