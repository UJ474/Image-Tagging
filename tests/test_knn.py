"""Unit tests for the k-Nearest Neighbors pipeline: aml.models.knn."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np

from aml.models.knn import (
    build_knn_pipeline,
    predict_in_batches,
    tune_knn,
    evaluate_final_knn,
    make_result_record,
    save_result,
)


def make_synthetic_data(n_samples, n_features=20, seed=0):
    """Creates synthetic multi-class data across 10 classes for testing."""
    rng = np.random.default_rng(seed)
    centers = np.random.default_rng(42).normal(scale=3.0, size=(10, n_features))
    y = np.arange(n_samples, dtype=np.int64) % 10
    X = (centers[y] + rng.normal(size=(n_samples, n_features))).astype(np.float32)
    return X, y


class KnnTests(unittest.TestCase):
    def setUp(self):
        self.X_train, self.y_train = make_synthetic_data(200, seed=1)
        self.X_val, self.y_val = make_synthetic_data(100, seed=2)
        self.X_test, self.y_test = make_synthetic_data(100, seed=3)

    def test_build_pipeline(self):
        model = build_knn_pipeline(n_neighbors=7, weights="distance")
        self.assertEqual(model.n_neighbors, 7)
        self.assertEqual(model.weights, "distance")
        self.assertEqual(model.metric, "euclidean")

    def test_predict_in_batches_matches_predict(self):
        model = build_knn_pipeline(n_neighbors=3)
        model.fit(self.X_train, self.y_train)

        direct_preds = model.predict(self.X_val)
        batch_preds = predict_in_batches(model, self.X_val, batch_size=25)

        np.testing.assert_array_equal(direct_preds, batch_preds)

    def test_tune_knn(self):
        best_params, results = tune_knn(
            self.X_train, self.y_train, self.X_val, self.y_val,
            k_grid=(3, 5), weights_grid=("uniform", "distance")
        )
        self.assertEqual(len(results), 4)
        self.assertIn(best_params["n_neighbors"], (3, 5))
        self.assertIn(best_params["weights"], ("uniform", "distance"))
        for r in results:
            self.assertTrue(0.0 <= r["val_accuracy"] <= 1.0)
            self.assertTrue(0.0 <= r["val_macro_f1"] <= 1.0)

    def test_evaluate_final_knn(self):
        best_params = {"n_neighbors": 3, "weights": "uniform"}
        model, metrics, latency_ms, param_mb = evaluate_final_knn(
            self.X_train, self.y_train, self.X_test, self.y_test, best_params
        )
        self.assertTrue(0.0 <= metrics["accuracy"] <= 1.0)
        self.assertTrue(0.0 <= metrics["macro_f1"] <= 1.0)
        self.assertEqual(len(metrics["per_class_recall"]), 10)
        self.assertGreater(latency_ms, 0.0)
        self.assertGreater(param_mb, 0.0)

    def test_make_result_record_and_save(self):
        tuning_results = [
            {"n_neighbors": 3, "weights": "uniform", "val_accuracy": 0.85, "val_macro_f1": 0.84},
            {"n_neighbors": 5, "weights": "uniform", "val_accuracy": 0.83, "val_macro_f1": 0.82},
        ]
        test_metrics = {
            "accuracy": 0.82,
            "macro_f1": 0.81,
            "per_class_recall": [0.8] * 10,
            "confusion_matrix": [[10] * 10] * 10,
        }
        record = make_result_record(
            features="pixels",
            best_params={"n_neighbors": 3, "weights": "uniform"},
            tuning_results=tuning_results,
            test_metrics=test_metrics,
            latency_ms=0.15,
            param_mb=179.44,
        )
        self.assertEqual(record["model"], "knn")
        self.assertEqual(record["features"], "pixels")
        self.assertEqual(record["parameter_count"], 179.44)

        with TemporaryDirectory() as tmp_dir:
            saved_path = save_result(record, results_dir=tmp_dir)
            self.assertTrue(Path(saved_path).is_file())
            loaded = json.loads(Path(saved_path).read_text())
            self.assertEqual(loaded["model"], "knn")


if __name__ == "__main__":
    unittest.main()
