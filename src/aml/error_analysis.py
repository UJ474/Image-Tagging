"""Plots and sample-selection helpers for Fashion-MNIST error analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _normalise_rows(matrix):
    """Normalize each row safely, retaining all-zero rows as zeros."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion matrix must be square.")
    totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals != 0)


def _save_figure(fig, save_path):
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", save_path=None):
    """Plot a row-normalized confusion matrix and optionally save the figure."""
    cm = np.asarray(cm)
    if len(class_names) != cm.shape[0]:
        raise ValueError("class_names length must match the confusion matrix.")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(_normalise_rows(cm), annot=True, fmt=".1%", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set(title=title, xlabel="Predicted category", ylabel="True category")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_garment_confusion_submatrix(cm, class_names, subset_indices=(0, 2, 4, 6),
                                     title="Upper-body garment confusion", save_path=None):
    """Zoom into T-shirt, Pullover, Coat, and Shirt predictions.

    Rows are normalized against the full confusion matrix, not merely the
    selected columns. This preserves the percentage of each true class that
    escaped the upper-body subset.
    """
    cm = np.asarray(cm)
    indices = np.asarray(subset_indices, dtype=int)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or np.any(indices < 0) or np.any(indices >= cm.shape[0]):
        raise ValueError("subset_indices must be valid confusion-matrix indices.")
    if len(class_names) != cm.shape[0]:
        raise ValueError("class_names length must match the confusion matrix.")
    subset = _normalise_rows(cm)[np.ix_(indices, indices)]
    names = [class_names[index] for index in indices]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(subset, annot=True, fmt=".1%", cmap="Reds", vmin=0,
                xticklabels=names, yticklabels=names, ax=ax)
    ax.set(title=title, xlabel="Predicted category", ylabel="True category")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def extract_misclassifications(y_true, y_pred, X_images, true_class, pred_class, max_samples=5):
    """Return up to ``max_samples`` images and indices for one error direction."""
    y_true, y_pred, X_images = np.asarray(y_true), np.asarray(y_pred), np.asarray(X_images)
    if y_true.ndim != 1 or y_pred.shape != y_true.shape or len(X_images) != len(y_true):
        raise ValueError("labels must be matching 1D arrays and align with X_images.")
    if max_samples < 1:
        raise ValueError("max_samples must be positive.")
    indices = np.flatnonzero((y_true == true_class) & (y_pred == pred_class))[:max_samples]
    return X_images[indices], indices


def plot_misclassification_gallery(images, indices, true_name, predicted_name, save_path=None):
    """Render selected 28x28 garment errors with source indices as captions."""
    images, indices = np.asarray(images), np.asarray(indices)
    if images.ndim != 3 or images.shape[1:] != (28, 28) or len(images) != len(indices):
        raise ValueError("images must have shape (N, 28, 28) and align with indices.")
    columns = min(5, max(1, len(images)))
    fig, axes = plt.subplots(1, columns, figsize=(2.2 * columns, 2.8), squeeze=False)
    for axis, image, index in zip(axes.flat, images, indices):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"test #{index}", fontsize=9)
        axis.axis("off")
    for axis in axes.flat[len(images):]:
        axis.axis("off")
    fig.suptitle(f"True: {true_name} → predicted: {predicted_name}", y=1.02, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig
