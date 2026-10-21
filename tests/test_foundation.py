"""Offline checks: python -m unittest discover -s tests -v."""

import gzip
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import struct
import unittest
from unittest.mock import Mock, patch

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aml.data import CLASS_NAMES, _download_idx, get_train_val_test_splits, load_fashion_mnist
from aml.evaluation import compute_metrics, count_parameters, measure_inference_latency


class DataTests(unittest.TestCase):
    def test_download_normalization_official_boundary_and_offline_cache(self):
        # All 70k rows are represented without allocating a large source array.
        row = (np.arange(784) % 256).astype(np.uint8)
        images = np.broadcast_to(row.reshape(1, 28, 28), (70_000, 28, 28))
        labels = np.arange(70_000) % 10
        with TemporaryDirectory() as directory:
            with patch("aml.data._download_idx", side_effect=[
                images[:60_000], labels[:60_000], images[60_000:], labels[60_000:],
            ]) as fetch:
                downloaded = load_fashion_mnist(directory)
                self.assertEqual(fetch.call_count, 4)
            with patch("aml.data._download_idx", side_effect=AssertionError("Network used")):
                cached = load_fashion_mnist(directory)
            for actual, expected in zip(cached, downloaded):
                np.testing.assert_array_equal(actual, expected)
            X_train, y_train, X_test, y_test = cached
            self.assertEqual(X_train.shape, (60_000, 28, 28))
            self.assertEqual(X_test.shape, (10_000, 28, 28))
            self.assertEqual(X_train.dtype, np.float32)
            self.assertEqual(y_train.dtype, np.int64)
            self.assertEqual(X_train.min(), 0)
            self.assertEqual(X_train.max(), 1)
            np.testing.assert_allclose(X_train[0].ravel(), row / 255.0, rtol=1e-7)
            np.testing.assert_array_equal(y_train, labels[:60_000])
            np.testing.assert_array_equal(y_test, labels[60_000:])

    def test_idx_checksum_and_header_validation(self):
        payload = gzip.compress(struct.pack(">II", 2049, 3) + bytes([2, 0, 9]))
        checksum = hashlib.md5(payload, usedforsecurity=False).hexdigest()
        with patch("aml.data.urlopen") as request:
            request.return_value.__enter__.return_value.read.return_value = payload
            np.testing.assert_array_equal(_download_idx("labels.gz", (3,), checksum), [2, 0, 9])
            self.assertEqual(request.call_args.kwargs["timeout"], 30)
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                _download_idx("labels.gz", (3,), "incorrect")
            with self.assertRaisesRegex(ValueError, "Unexpected IDX header"):
                _download_idx("labels.gz", (4,), checksum)

    def test_invalid_cache_is_actionable_and_does_not_download(self):
        with TemporaryDirectory() as directory:
            np.savez(Path(directory) / "fashion_mnist_raw.npz", unexpected=[0])
            with patch("aml.data._download_idx") as fetch:
                with self.assertRaisesRegex(ValueError, "Delete it and retry"):
                    load_fashion_mnist(directory)
                fetch.assert_not_called()

    def test_splits_are_stratified_repeatable_disjoint_and_preserve_test(self):
        # Encode each row's identity so overlap and coverage can be checked.
        X = np.arange(60_000, dtype=np.float32).reshape(-1, 1, 1)
        y = np.arange(60_000, dtype=np.int64) % 10
        X_test, y_test = np.zeros((10_000, 1, 1)), np.arange(10_000) % 10
        with patch("aml.data.load_fashion_mnist", return_value=(X, y, X_test, y_test)):
            first = get_train_val_test_splits()
            second = get_train_val_test_splits()
            changed = get_train_val_test_splits(seed=7)
        self.assertEqual(len(first["X_train"]), 48_000)
        self.assertEqual(len(first["X_val"]), 12_000)
        np.testing.assert_array_equal(np.bincount(first["y_train"]), [4800] * 10)
        np.testing.assert_array_equal(np.bincount(first["y_val"]), [1200] * 10)
        np.testing.assert_array_equal(first["X_train"], second["X_train"])
        self.assertFalse(np.array_equal(first["X_train"], changed["X_train"]))
        train_ids, val_ids = first["X_train"].ravel(), first["X_val"].ravel()
        self.assertEqual(np.intersect1d(train_ids, val_ids).size, 0)
        np.testing.assert_array_equal(np.sort(np.concatenate([train_ids, val_ids])), X.ravel())
        self.assertIs(first["X_train_full"], X)
        self.assertIs(first["X_test"], X_test)
        self.assertIs(first["y_test"], y_test)
        self.assertEqual(first["class_names"], CLASS_NAMES)


class EvaluationTests(unittest.TestCase):
    def test_known_metrics_and_fixed_class_order_are_json_serializable(self):
        metrics = compute_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3 + 0.8) / 10)
        self.assertEqual(metrics["per_class_recall"], [0.5, 1.0] + [0.0] * 8)
        matrix = np.array(metrics["confusion_matrix"])
        self.assertEqual(matrix.shape, (10, 10))
        np.testing.assert_array_equal(matrix[:2, :2], [[1, 1], [0, 2]])
        json.dumps(metrics, allow_nan=False)

    def test_invalid_labels_and_empty_inputs(self):
        for truth, prediction in [([], []), ([0], [0, 1]), ([10], [0]), ([0], [-1])]:
            with self.subTest(truth=truth, prediction=prediction), self.assertRaises(ValueError):
                compute_metrics(truth, prediction)

    def test_counts_fitted_linear_pipeline_and_forest(self):
        X = np.arange(60).reshape(20, 3)
        y = np.arange(20) % 2
        pipeline = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)
        self.assertEqual(count_parameters(pipeline), 4)
        forest = RandomForestClassifier(n_estimators=2, max_depth=2, random_state=42).fit(X, y)
        self.assertEqual(count_parameters(forest), sum(t.tree_.node_count for t in forest.estimators_))

    def test_latency_median_units_and_warmup(self):
        model = Mock()
        X = np.zeros((20, 3))
        with patch("aml.evaluation.time.perf_counter", side_effect=[0, .04, 1, 1.02, 2, 2.06]):
            latency = measure_inference_latency(model, X, n_iterations=3)
        self.assertAlmostEqual(latency, 2.0)  # median .04 seconds / 20 * 1000
        self.assertEqual(model.predict.call_count, 4)
        self.assertEqual(len(model.predict.call_args_list[0].args[0]), 10)

    def test_invalid_latency_inputs(self):
        model = Mock()
        with self.assertRaises(ValueError):
            measure_inference_latency(model, np.zeros((0, 3)))
        for iterations in [0, -1, 1.5, True]:
            with self.subTest(iterations=iterations), self.assertRaises(ValueError):
                measure_inference_latency(model, np.zeros((2, 3)), iterations)
        model.predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
