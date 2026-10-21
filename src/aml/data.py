"""Load Fashion-MNIST, normalize pixels, and reproduce the team's data splits."""

import gzip
import hashlib
from pathlib import Path
import struct
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from zipfile import BadZipFile

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
_CACHE_KEYS = ("X_train_raw", "y_train_raw", "X_test", "y_test")
_DATA_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
# Checksums published by https://github.com/zalandoresearch/fashion-mnist
_FILES = (
    ("train-images-idx3-ubyte.gz", (60_000, 28, 28), "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
    ("train-labels-idx1-ubyte.gz", (60_000,), "25c81989df183df01b3e8a0aad5dffbe"),
    ("t10k-images-idx3-ubyte.gz", (10_000, 28, 28), "bef4ecab320f06d8554ea6380940ec79"),
    ("t10k-labels-idx1-ubyte.gz", (10_000,), "bb300cfdad3c16e7a12a480ee83cd310"),
)


def _download_idx(filename, shape, checksum):
    """Download an official uint8 IDX file and verify its checksum and header."""
    with urlopen(_DATA_URL + filename, timeout=30) as response:
        payload = response.read()
    if hashlib.md5(payload, usedforsecurity=False).hexdigest() != checksum:
        raise ValueError(f"Checksum mismatch for {filename}; retry the download.")
    raw = gzip.decompress(payload)
    header_size = 4 * (len(shape) + 1)
    header = struct.unpack(f">{len(shape) + 1}I", raw[:header_size])
    if header != (0x0800 + len(shape), *shape):
        raise ValueError(f"Unexpected IDX header for {filename}.")
    return np.frombuffer(raw, dtype=np.uint8, offset=header_size).reshape(shape)


def _validate_official_data(X_train_raw, y_train_raw, X_test, y_test):
    """Reject malformed caches instead of silently training on the wrong data."""
    for X, y, size in ((X_train_raw, y_train_raw, 60_000), (X_test, y_test, 10_000)):
        if X.shape != (size, 28, 28) or y.shape != (size,):
            raise ValueError(f"Expected {size} images of shape (28, 28) and labels.")
        if X.dtype != np.float32 or y.dtype != np.int64:
            raise ValueError("Expected float32 images and int64 labels.")
        if not np.isfinite(X).all() or X.min() < 0 or X.max() > 1:
            raise ValueError("Image values must be finite and normalized to [0, 1].")
        if not np.isin(y, np.arange(10)).all():
            raise ValueError("Fashion-MNIST labels must be integers from 0 to 9.")


def load_fashion_mnist(data_dir="data/raw"):
    """Return official 60,000 training and 10,000 test images and labels.

    Images have shape (N, 28, 28), dtype float32, and values in [0, 1].
    Labels have shape (N,) and dtype int64. The first call downloads the
    checksum-verified IDX files from Zalando's official GitHub repository;
    later calls use fashion_mnist_raw.npz offline.
    data_dir is relative to the current working directory unless absolute.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_file = data_dir / "fashion_mnist_raw.npz"

    if cache_file.exists():
        try:
            with np.load(cache_file, allow_pickle=False) as cached:
                arrays = tuple(cached[key] for key in _CACHE_KEYS)
            _validate_official_data(*arrays)
        except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
            raise ValueError(
                f"Invalid Fashion-MNIST cache: {cache_file}. "
                "Delete it and retry to download a fresh copy."
            ) from exc
        return arrays

    train_images, train_labels, test_images, test_labels = (
        _download_idx(*spec) for spec in _FILES
    )
    X_train_raw = train_images.astype(np.float32)
    X_test = test_images.astype(np.float32)
    X_train_raw /= np.float32(255.0)
    X_test /= np.float32(255.0)
    arrays = (X_train_raw, train_labels.astype(np.int64), X_test, test_labels.astype(np.int64))
    _validate_official_data(*arrays)

    # Replace the cache only after the complete archive has been written.
    with NamedTemporaryFile(dir=data_dir, suffix=".npz", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        np.savez_compressed(temporary_path, **dict(zip(_CACHE_KEYS, arrays)))
        temporary_path.replace(cache_file)
    finally:
        temporary_path.unlink(missing_ok=True)
    return arrays


def get_train_val_test_splits(data_dir="data/raw", val_size=12_000, seed=42):
    """Return stratified 48k/12k development splits plus the official 10k test.

    X_train_full/y_train_full retain the entire official training set for
    refitting after model selection. The held-out test set is never split.
    val_size can also be a fraction, as accepted by StratifiedShuffleSplit.
    """
    X_train_raw, y_train_raw, X_test, y_test = load_fashion_mnist(data_dir)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(X_train_raw, y_train_raw))
    return {
        "X_train": X_train_raw[train_idx],
        "y_train": y_train_raw[train_idx],
        "X_val": X_train_raw[val_idx],
        "y_val": y_train_raw[val_idx],
        "X_train_full": X_train_raw,
        "y_train_full": y_train_raw,
        "X_test": X_test,
        "y_test": y_test,
        "class_names": CLASS_NAMES.copy(),
    }
