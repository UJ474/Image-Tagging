"""Feature extraction module for Fashion-MNIST (Raw Pixels and HOG Descriptors)."""

import os
from pathlib import Path
import numpy as np


def extract_pixels(images):
    """Flatten (N, 28, 28) or single (28, 28) images into (N, 784) float32 vectors.

    Parameters
    ----------
    images : np.ndarray
        Array of shape (N, 28, 28) or (28, 28) with values in [0.0, 1.0].

    Returns
    -------
    np.ndarray
        2D feature matrix of shape (N, 784) and dtype float32.
    """
    images = np.asarray(images, dtype=np.float32)
    if images.ndim == 2:
        images = images[np.newaxis, ...]
    elif images.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array of images, got shape {images.shape}.")

    if images.shape[1:] != (28, 28):
        raise ValueError(f"Expected image dimensions (28, 28), got {images.shape[1:]}.")

    return images.reshape(images.shape[0], -1).astype(np.float32)
