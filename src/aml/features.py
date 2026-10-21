"""Feature extraction module for Fashion-MNIST (Raw Pixels and HOG Descriptors)."""

import os
from pathlib import Path
import numpy as np
from skimage.feature import hog


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


def extract_single_hog(
    image,
    orientations=9,
    pixels_per_cell=(7, 7),
    cells_per_block=(2, 2),
):
    """Compute HOG descriptor for a single (28, 28) image.

    Parameters
    ----------
    image : np.ndarray
        Single image of shape (28, 28) with float32 values in [0.0, 1.0].
    orientations : int, default=9
        Number of gradient orientation bins.
    pixels_per_cell : tuple of (int, int), default=(7, 7)
        Size (in pixels) of each cell.
    cells_per_block : tuple of (int, int), default=(2, 2)
        Number of cells in each block.

    Returns
    -------
    np.ndarray
        1D feature vector of shape (324,) and dtype float32.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.shape != (28, 28):
        raise ValueError(f"Expected image of shape (28, 28), got {image.shape}.")

    descriptor = hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        transform_sqrt=False,
        feature_vector=True,
    )
    return descriptor.astype(np.float32)


def extract_hog_batch(
    images,
    orientations=9,
    pixels_per_cell=(7, 7),
    cells_per_block=(2, 2),
):
    """Compute HOG descriptors for a batch of (N, 28, 28) images.

    Parameters
    ----------
    images : np.ndarray
        Array of shape (N, 28, 28) or single (28, 28) image.
    orientations : int, default=9
        Number of gradient orientation bins.
    pixels_per_cell : tuple of (int, int), default=(7, 7)
        Size (in pixels) of each cell.
    cells_per_block : tuple of (int, int), default=(2, 2)
        Number of cells in each block.

    Returns
    -------
    np.ndarray
        2D feature matrix of shape (N, 324) and dtype float32.
    """
    images = np.asarray(images, dtype=np.float32)
    if images.ndim == 2:
        images = images[np.newaxis, ...]
    elif images.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array of images, got shape {images.shape}.")

    if images.shape[1:] != (28, 28):
        raise ValueError(f"Expected image dimensions (28, 28), got {images.shape[1:]}.")

    n_samples = images.shape[0]
    descriptors = [
        extract_single_hog(
            images[i],
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
        )
        for i in range(n_samples)
    ]
    return np.asarray(descriptors, dtype=np.float32)
