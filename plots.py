"""
Visualization script for VentureSurvive model evaluation.

This script loads trained models and generates comprehensive visualizations:
- ROC curves for all models
- Precision-Recall curves
- Confusion matrices
- Feature importance analysis
- Model performance comparison

Usage
-----
python plots.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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


def prepare_data():
    """Load and prepare test data."""
    print("\n" + "=" * 70)
    print("📊 Preparing data for visualization")
    print("=" * 70)
    
    # Load cleaned data
    df_clean = data_mod.load_cleaned_data()
    print(f"✓ Dataset loaded: {df_clean.shape[0]:,} rows")
    
    # Assemble features
    df_features = features_mod.assemble_features(df_clean)
    
    # Remove status column to avoid leakage
    if "status" in df_features.columns:
        df_features = df_features.drop(columns=["status"])
    
    # Separate target
    y = df_features["success"].astype(int)
    X = df_features.drop(columns=["success"])
    
    # Temporal split
    df_all = X.copy()
    df_all["success"] = y.values
    
    train_df, test_df = split_mod.temporal_split(
        df_all, 
        date_col="first_funding_at",
        cutoff_date="2013-01-01"
    )
    
    # Drop date columns
    date_cols = ["founded_at", "first_funding_at", "last_funding_at"]
    drop_cols = [c for c in date_cols if c in train_df.columns]
    
    if drop_cols:
        test_df = test_df.drop(columns=drop_cols)
    
    y_test = test_df["success"].astype(int)
    X_test = test_df.drop(columns=["success"])
    
    print(f"✓ Test set: {X_test.shape[0]:,} samples, {X_test.shape[1]} features")
    
    return X_test, y_test


def plot_roc_comparison(models: dict, X_test, y_test):
    """Plot ROC curves for all models on the same figure."""
    print("\n📈 Generating ROC curves comparison...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    for (name, model), color in zip(models.items(), colors):
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = eval_mod.metrics.roc_curve(y_test, y_score)
        auc = eval_mod.metrics.roc_auc_score(y_test, y_score)
        
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", 
                linewidth=2, color=color)
    
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = RESULTS_DIR / "roc_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved to {save_path}")
    plt.show()


def plot_pr_comparison(models: dict, X_test, y_test):
    """Plot Precision-Recall curves for all models."""
    print("\n📈 Generating Precision-Recall curves comparison...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    for (name, model), color in zip(models.items(), colors):
        y_score = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = eval_mod.metrics.precision_recall_curve(y_test, y_score)
        ap = eval_mod.metrics.average_precision_score(y_test, y_score)
        
        ax.plot(recall, precision, label=f"{name} (AP = {ap:.3f})", 
                linewidth=2, color=color)
    
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves - Model Comparison", 
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = RESULTS_DIR / "pr_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved to {save_path}")
    plt.show()


def plot_confusion_matrices(models: dict, X_test, y_test):
    """Plot confusion matrices for all models."""
    print("\n📊 Generating confusion matrices...")
    
    n_models = len(models)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (name, model) in enumerate(models.items()):
        ax = axes[idx]
        
        # Compute confusion matrix directly
        y_pred = model.predict(X_test)
        cm = eval_mod.metrics.confusion_matrix(y_test, y_pred, normalize="true")
        
        # Plot using ConfusionMatrixDisplay
        disp = eval_mod.metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, colorbar=True, cmap="Blues")
        ax.set_title(name, fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    save_path = RESULTS_DIR / "confusion_matrices.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved to {save_path}")
    plt.show()


def plot_metrics_comparison(models: dict, X_test, y_test):
    """Create a bar chart comparing all metrics across models."""
    print("\n📊 Generating metrics comparison...")
    
    # Compute metrics for all models
    metrics_data = []
    for name, model in models.items():
        metrics_dict = eval_mod.evaluate_classification(model, X_test, y_test)
        metrics_dict["model"] = name
        metrics_data.append(metrics_dict)
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    
    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        df_metrics.plot(
            x="model", y=metric, kind="bar", ax=ax, 
            legend=False, color="#2ca02c", edgecolor="black"
        )
        ax.set_title(metric.upper().replace("_", " "), fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis="y")
    
    # Hide the last subplot
    axes[-1].axis("off")
    
    plt.tight_layout()
    save_path = RESULTS_DIR / "metrics_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved to {save_path}")
    plt.show()
    
    # Print metrics table
    print("\n" + "=" * 70)
    print("📊 Metrics Summary Table")
    print("=" * 70)
    print(df_metrics.to_string(index=False))


def plot_feature_importance_analysis(models: dict, X_test, y_test):
    """Compute and plot feature importance for the best model."""
    print("\n🔍 Computing feature importance (this may take a moment)...")
    
    # Use the best model (Logistic Regression based on ROC AUC)
    best_model_name = "Logistic Regression"
    best_model = models.get(best_model_name)
    
    if best_model is None:
        print("  ⚠️  Best model not found, skipping feature importance")
        return
    
    # Compute permutation importance
    importance_df = eval_mod.compute_feature_importance(
        best_model, 
        X_test, 
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE
    )
    
    # Plot top 20 features
    top_n = 20
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(range(top_n), top_features["importance_mean"], 
            color="#1f77b4", edgecolor="black")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features["feature"])
    ax.set_xlabel("Permutation Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances - {best_model_name}", 
                 fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    save_path = RESULTS_DIR / "feature_importance.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved to {save_path}")
    plt.show()
    
    # Save importance table
    csv_path = RESULTS_DIR / "feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved importance table to {csv_path}")


def main():
    """Generate all visualizations."""
    print("=" * 70)
    print("📊 VentureSurvive - Model Evaluation Visualizations")
    print("=" * 70)
    
    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print("\n🔧 Loading trained models...")
    models = load_models()
    
    if not models:
        print("\n❌ No models found. Please run main.py first to train models.")
        return
    
    # Prepare data
    X_test, y_test = prepare_data()
    
    # Generate all plots
    plot_roc_comparison(models, X_test, y_test)
    plot_pr_comparison(models, X_test, y_test)
    plot_confusion_matrices(models, X_test, y_test)
    plot_metrics_comparison(models, X_test, y_test)
    plot_feature_importance_analysis(models, X_test, y_test)
    
    print("\n" + "=" * 70)
    print("✅ All visualizations generated successfully!")
    print(f"   Results saved in: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()