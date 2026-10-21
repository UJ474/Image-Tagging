"""Consistent metrics, model parameter counts, and prediction timing."""

from numbers import Integral
import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.utils.validation import check_is_fitted


def compute_metrics(y_true, y_pred):
    """Return JSON-ready metrics using the fixed Fashion-MNIST label order.

    Macro F1 averages all ten classes, including classes missing from a
    sample. Undefined precision/recall values contribute zero.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("Expected nonempty, equally sized one-dimensional labels.")
    labels = np.arange(10)
    if not np.isin(y_true, labels).all() or not np.isin(y_pred, labels).all():
        raise ValueError("Fashion-MNIST labels must be integers from 0 to 9.")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0,
        )),
        "per_class_recall": recall_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0,
        ).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def count_parameters(model):
    """Count fitted linear weights/intercepts or total random-forest nodes.

    For a Pipeline, inspect its final estimator. Preprocessing parameters
    are excluded, matching the team's result schema. k-NN is non-parametric:
    report its stored feature array's nbytes / 1024**2 in MiB instead.
    """
    if hasattr(model, "steps"):
        model = model.steps[-1][1]
    check_is_fitted(model)
    if hasattr(model, "coef_"):
        return int(model.coef_.size + np.asarray(model.intercept_).size)
    if hasattr(model, "estimators_") and all(
        hasattr(tree, "tree_") for tree in model.estimators_
    ):
        return int(sum(tree.tree_.node_count for tree in model.estimators_))
    raise TypeError("Expected a fitted linear classifier or random forest.")


def measure_inference_latency(model, X_sample, n_iterations=5):
    """Return median batch prediction time per image in milliseconds.

    Warm up on up to ten images. Time predict(), including any preprocessing
    in a passed Pipeline; feature extraction outside the model is excluded.
    Use identical batch sizes, iteration counts, and hardware for comparisons.
    """
    if len(X_sample) == 0:
        raise ValueError("X_sample must contain at least one image.")
    if isinstance(n_iterations, bool) or not isinstance(n_iterations, Integral) or n_iterations < 1:
        raise ValueError("n_iterations must be a positive integer.")
    model.predict(X_sample[:min(10, len(X_sample))])
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        model.predict(X_sample)
        times.append((time.perf_counter() - start) / len(X_sample))
    return float(np.median(times) * 1000.0)
