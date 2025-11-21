"""Evaluation utilities for classification models."""

from __future__ import annotations

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def evaluate_classification(model, X_test, y_test) -> Dict[str, float]:
    """Compute standard classification metrics.

    Returns accuracy, precision, recall, f1, and ROC AUC when possible.
    """
    y_pred = model.predict(X_test)

    # Try to obtain scores for ROC/PR
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = None

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


def plot_roc(model, X_test, y_test, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot ROC curve for a binary classifier."""
    if ax is None:
        fig, ax = plt.subplots()

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        raise ValueError("Model does not provide a suitable score for ROC.")

    fpr, tpr, _ = metrics.roc_curve(y_test, y_score)
    auc = metrics.roc_auc_score(y_test, y_score)

    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    return ax


def plot_pr(model, X_test, y_test, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot precision-recall curve."""
    if ax is None:
        fig, ax = plt.subplots()

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        raise ValueError("Model does not provide a suitable score for PR curve.")

    precision, recall, _ = metrics.precision_recall_curve(y_test, y_score)
    ap = metrics.average_precision_score(y_test, y_score)

    ax.plot(recall, precision, label=f"PR (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    return ax


def plot_confusion(model, X_test, y_test, normalize: bool = False, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot confusion matrix."""
    if ax is None:
        fig, ax = plt.subplots()

    y_pred = model.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred, normalize="true" if normalize else None)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False)
    title = "Confusion Matrix (normalized)" if normalize else "Confusion Matrix"
    ax.set_title(title)
    return ax
