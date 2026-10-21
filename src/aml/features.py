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


def get_features(images, feature_type="pixels", cache_path=None):
    """Extract or load cached features.

    Parameters
    ----------
    images : np.ndarray or None
        Array of shape (N, 28, 28) or (28, 28). Can be None if cache_path exists.
    feature_type : str, default="pixels"
        Type of feature to extract: "pixels" (784-d) or "hog" (324-d).
    cache_path : str or Path or None, optional
        Optional path to a .npz cache file.

    Returns
    -------
    np.ndarray
        Feature matrix of shape (N, 784) for pixels or (N, 324) for hog.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as cached:
                return cached["features"]

    if images is None:
        raise ValueError(f"Cache file {cache_path} does not exist and images is None.")

    if feature_type == "pixels":
        features = extract_pixels(images)
    elif feature_type == "hog":
        features = extract_hog_batch(images)
    else:
        raise ValueError(f"Unknown feature_type: '{feature_type}'. Expected 'pixels' or 'hog'.")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, features=features)

    return features


def precompute_and_cache_all(data_dir="data/raw", cache_dir="data/cache"):
    """Precompute and cache features for train, val, test, and full training splits.

    Parameters
    ----------
    data_dir : str or Path, default="data/raw"
        Directory containing raw Fashion-MNIST files.
    cache_dir : str or Path, default="data/cache"
        Directory where precomputed .npz archives will be saved.
    """
    from aml.data import get_train_val_test_splits

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data splits...")
    splits = get_train_val_test_splits(data_dir=data_dir)

    split_names = ("train", "val", "test", "train_full")
    for name in split_names:
        imgs = splits[f"X_{name}"]
        print(f"Extracting pixels for {name} ({len(imgs)} images)...")
        get_features(
            imgs,
            feature_type="pixels",
            cache_path=cache_dir / f"{name}_pixels.npz",
        )

        print(f"Extracting HOG for {name} ({len(imgs)} images)...")
        get_features(
            imgs,
            feature_type="hog",
            cache_path=cache_dir / f"{name}_hog.npz",
        )

    print(f"All features successfully cached in {cache_dir}.")


if __name__ == "__main__":
    precompute_and_cache_all()
