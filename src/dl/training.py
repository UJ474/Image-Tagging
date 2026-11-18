"""Training engine, evaluation loop, and early stopping for Phase 2 CNN models.

Owner: Kartik Mehra (@KartikMehra22)
Phase: Phase 2 — Deep Learning

This module is the shared training infrastructure used by all Phase 2 experiments.
It provides:
  - EarlyStopping  : Tracks validation Macro F1 and signals when to halt.
  - train_epoch    : One pass through the training DataLoader with backpropagation.
  - evaluate_model : Full-dataset prediction and metric computation (accuracy,
                     Macro F1, per-class recall) using the Phase 1 compute_metrics().
  - fit_model      : End-to-end training loop with LR scheduling, early stopping,
                     and best-weight checkpointing. Returns a compact history dict.

Usage example (from a notebook)
---------------------------------
    from dl.training import fit_model
    from dl.data import get_dataloaders
    from dl.models.cnn import SimpleCNN

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)
    model = SimpleCNN()
    history = fit_model(model, train_loader, val_loader, epochs=30, lr=1e-3)
"""

from __future__ import annotations

import copy
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from aml.evaluation import compute_metrics


# ---------------------------------------------------------------------------
# 1. Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when validation Macro F1 does not improve.

    Parameters
    ----------
    patience : int
        Number of consecutive epochs without improvement before stopping.
        Teacher's requirement: patience=5.
    mode : str
        'max' — stop when the monitored metric stops increasing (default for F1).
        'min' — stop when it stops decreasing (useful for loss).
    min_delta : float
        Minimum absolute change to count as an improvement. Prevents
        trivially small floating-point changes from resetting the counter.

    Attributes
    ----------
    best_score : float or None
        Best observed metric value so far.
    best_weights : dict or None
        Deep copy of the model state_dict at the best epoch. Restored
        automatically via restore_best_weights().
    counter : int
        Consecutive epochs without a qualifying improvement.
    early_stop : bool
        Set to True when the patience limit is reached.
    best_epoch : int
        Epoch index (0-based) where the best score was recorded.

    Example
    -------
        es = EarlyStopping(patience=5, mode='max')
        for epoch in range(30):
            val_f1 = evaluate(model, val_loader)
            es(val_f1, model)
            if es.early_stop:
                break
        es.restore_best_weights(model)
    """

    def __init__(self, patience: int = 5, mode: str = "max", min_delta: float = 1e-4):
        if patience < 1:
            raise ValueError("patience must be at least 1.")
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'.")

        self.patience = patience
        self.mode = mode
        self.min_delta = abs(min_delta)

        self.best_score: Optional[float] = None
        self.best_weights: Optional[dict] = None
        self.counter: int = 0
        self.early_stop: bool = False
        self.best_epoch: int = 0
        self._epoch: int = 0

    def _is_improvement(self, score: float) -> bool:
        """Return True if score qualifies as a meaningful improvement."""
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def __call__(self, score: float, model: nn.Module) -> None:
        """Check the current score and update internal state.

        Parameters
        ----------
        score : float
            Validation metric for this epoch (e.g., Macro F1 from 0 to 1).
        model : nn.Module
            The model being trained. Its weights are copied on improvement.
        """
        if self._is_improvement(score):
            self.best_score = score
            self.best_epoch = self._epoch
            # Deep-copy so later weight updates do not corrupt the saved state.
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        self._epoch += 1

    def restore_best_weights(self, model: nn.Module) -> None:
        """Load the best checkpoint back into the model in-place.

        Call this after the training loop exits (whether by early stopping or
        exhausting all epochs) to ensure the model is at its peak validation state.
        """
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


# ---------------------------------------------------------------------------
# 2. Single-Epoch Training Step
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch: forward pass, backpropagation, weight update.

    Parameters
    ----------
    model : nn.Module
        The CNN model (SimpleCNN or DeepCNN) in training mode.
    loader : DataLoader
        Training DataLoader (shuffled, with augmentation).
    criterion : nn.Module
        Loss function. Use CrossEntropyLoss for 10-class classification.
    optimizer : torch.optim.Optimizer
        Adam (or any other) optimizer containing the model parameters.
    device : torch.device
        'cpu' or 'cuda'. Both model and batch tensors must be on the same device.

    Returns
    -------
    avg_loss : float
        Mean cross-entropy loss over all mini-batches in the epoch.
    accuracy : float
        Fraction of training samples correctly classified in this epoch.
    """
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)              # Forward pass
        loss = criterion(logits, labels)    # Cross-entropy loss
        loss.backward()                     # Backpropagation
        optimizer.step()                    # Weight update (Adam step)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# 3. Evaluation (Validation & Test)
# ---------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Compute predictions and return Phase-1-compatible metrics.

    Uses the shared ``compute_metrics()`` from ``src/aml/evaluation.py`` so
    Phase 1 and Phase 2 results are directly comparable (same accuracy, Macro
    F1, per-class recall, and confusion-matrix keys).

    Parameters
    ----------
    model : nn.Module
        Fitted CNN model (must be in eval mode; this function sets it).
    loader : DataLoader
        Validation or test DataLoader (no augmentation, no shuffle).
    device : torch.device
        Device to run inference on.

    Returns
    -------
    metrics : dict
        Dictionary with keys:
        - 'accuracy'       : float in [0, 1]
        - 'macro_f1'       : float in [0, 1]
        - 'per_class_recall': list of 10 floats
        - 'confusion_matrix': 10×10 list of ints
    """
    model.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().tolist())

    return compute_metrics(
        np.array(all_labels, dtype=np.int64),
        np.array(all_preds, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
