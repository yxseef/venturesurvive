"""Evaluation utilities for classification models."""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _finalize_plot(show: bool = True, save_path: Optional[str] = None) -> None:
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def _get_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def evaluate_classification(model, X_test, y_test) -> Dict[str, float]:
    """Compute standard classification metrics."""
    y_pred = model.predict(X_test)
    y_score = _get_score(model, X_test)

    results = {
        "accuracy": float(metrics.accuracy_score(y_test, y_pred)),
        "precision": float(metrics.precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(metrics.recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(metrics.f1_score(y_test, y_pred, zero_division=0)),
    }

    if y_score is not None:
        try:
            results["roc_auc"] = float(metrics.roc_auc_score(y_test, y_score))
        except Exception:
            results["roc_auc"] = float("nan")

    return results


# ---------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------

def plot_roc(
    model,
    X_test,
    y_test,
    *,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Plot ROC curve for a binary classifier."""
    if ax is None:
        _, ax = plt.subplots()

    y_score = _get_score(model, X_test)
    if y_score is None:
        raise ValueError("Model does not provide a suitable score for ROC curve.")

    fpr, tpr, _ = metrics.roc_curve(y_test, y_score)
    auc = metrics.roc_auc_score(y_test, y_score)

    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    _finalize_plot(show=show, save_path=save_path)
    return ax


def plot_pr(
    model,
    X_test,
    y_test,
    *,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Plot precision-recall curve."""
    if ax is None:
        _, ax = plt.subplots()

    y_score = _get_score(model, X_test)
    if y_score is None:
        raise ValueError("Model does not provide a suitable score for PR curve.")

    precision, recall, _ = metrics.precision_recall_curve(y_test, y_score)
    ap = metrics.average_precision_score(y_test, y_score)

    ax.plot(recall, precision, label=f"PR (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    _finalize_plot(show=show, save_path=save_path)
    return ax


def plot_confusion(
    model,
    X_test,
    y_test,
    *,
    normalize: bool = False,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Plot confusion matrix."""
    if ax is None:
        _, ax = plt.subplots()

    y_pred = model.predict(X_test)
    cm = metrics.confusion_matrix(
        y_test, y_pred, normalize="true" if normalize else None
    )
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False)

    ax.set_title("Confusion Matrix (normalized)" if normalize else "Confusion Matrix")
    _finalize_plot(show=show, save_path=save_path)
    return ax


# ---------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------

def compute_feature_importance(
    model,
    X_test: pd.DataFrame,
    y_test,
    *,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation feature importance at the raw feature level."""
    from sklearn.inspection import permutation_importance

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importance_df = (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    return importance_df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    *,
    top_n: int = 15,
    title: str = "Feature Importances",
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """Plot permutation feature importance."""
    top = importance_df.head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"], top["importance_mean"])
    plt.xlabel("Permutation Importance")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    _finalize_plot(show=show, save_path=save_path)


# ---------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------

__all__ = [
    "evaluate_classification",
    "plot_roc",
    "plot_pr",
    "plot_confusion",
    "compute_feature_importance",
    "plot_feature_importance",
]
