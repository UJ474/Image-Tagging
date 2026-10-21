"""Unit tests for feature extraction and caching: aml.features."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from aml.features import (
    extract_hog_batch,
    extract_pixels,
    extract_single_hog,
    get_features,
    precompute_and_cache_all,
)


class FeaturesTests(unittest.TestCase):
    def test_extract_pixels_shapes_and_values(self):
        sample = np.linspace(0.0, 1.0, 784, dtype=np.float32).reshape(28, 28)
        single = extract_pixels(sample)
        self.assertEqual(single.shape, (1, 784))
        self.assertEqual(single.dtype, np.float32)
        np.testing.assert_allclose(single[0], sample.ravel())

        batch = np.broadcast_to(sample, (4, 28, 28))
        batched = extract_pixels(batch)
        self.assertEqual(batched.shape, (4, 784))
        self.assertEqual(batched.dtype, np.float32)

    def test_extract_pixels_invalid_inputs(self):
        with self.assertRaises(ValueError):
            extract_pixels(np.zeros(784))
        with self.assertRaises(ValueError):
            extract_pixels(np.zeros((2, 28, 28, 1)))
        with self.assertRaises(ValueError):
            extract_pixels(np.zeros((2, 32, 32)))

    def test_extract_hog_single_and_batch(self):
        sample = np.zeros((28, 28), dtype=np.float32)
        sample[10:18, 10:18] = 1.0  # Add a sharp edge box

        single = extract_single_hog(sample)
        self.assertEqual(single.shape, (324,))
        self.assertEqual(single.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(single)))
        self.assertGreater(np.max(single), 0.0)

        batch = np.stack([sample, np.zeros((28, 28), dtype=np.float32)])
        batched = extract_hog_batch(batch)
        self.assertEqual(batched.shape, (2, 324))
        self.assertEqual(batched.dtype, np.float32)
        np.testing.assert_allclose(batched[0], single)

    def test_extract_hog_invalid_inputs(self):
        with self.assertRaises(ValueError):
            extract_single_hog(np.zeros((32, 32)))
        with self.assertRaises(ValueError):
            extract_hog_batch(np.zeros((2, 14, 14)))

    def test_get_features_caching_and_reload(self):
        images = np.zeros((3, 28, 28), dtype=np.float32)
        with TemporaryDirectory() as tmp_dir:
            cache_file = Path(tmp_dir) / "sub" / "test_cache.npz"

            # 1. Compute and save to cache
            features = get_features(images, feature_type="pixels", cache_path=cache_file)
            self.assertTrue(cache_file.exists())
            self.assertEqual(features.shape, (3, 784))

            # 2. Reload directly from cache with images=None
            reloaded = get_features(None, feature_type="pixels", cache_path=cache_file)
            np.testing.assert_array_equal(features, reloaded)

            # 3. Test HOG caching
            hog_cache = Path(tmp_dir) / "test_hog.npz"
            hog_features = get_features(images, feature_type="hog", cache_path=hog_cache)
            self.assertEqual(hog_features.shape, (3, 324))
            reloaded_hog = get_features(None, feature_type="hog", cache_path=hog_cache)
            np.testing.assert_array_equal(hog_features, reloaded_hog)

    def test_get_features_validation(self):
        images = np.zeros((2, 28, 28), dtype=np.float32)
        with self.assertRaises(ValueError):
            get_features(images, feature_type="invalid_type")

        with self.assertRaises(ValueError):
            get_features(None, feature_type="pixels", cache_path="non_existent.npz")

    def test_precompute_and_cache_all(self):
        mock_splits = {
            "X_train": np.zeros((2, 28, 28), dtype=np.float32),
            "X_val": np.zeros((1, 28, 28), dtype=np.float32),
            "X_test": np.zeros((1, 28, 28), dtype=np.float32),
            "X_train_full": np.zeros((3, 28, 28), dtype=np.float32),
        }
        with TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir) / "cache"
            with patch("aml.data.get_train_val_test_splits", return_value=mock_splits):
                precompute_and_cache_all(data_dir=tmp_dir, cache_dir=cache_dir)

            expected_files = [
                "train_pixels.npz", "train_hog.npz",
                "val_pixels.npz", "val_hog.npz",
                "test_pixels.npz", "test_hog.npz",
                "train_full_pixels.npz", "train_full_hog.npz",
            ]
            for filename in expected_files:
                self.assertTrue((cache_dir / filename).exists(), f"Missing cache file: {filename}")


if __name__ == "__main__":
    unittest.main()
