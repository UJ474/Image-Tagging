"""Unit tests for the Logistic Regression pipeline on synthetic data: aml.models.logreg."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from aml.models.logreg import (
    build_logreg_pipeline,
    evaluate_final_logreg,
    make_result_record,
    save_result,
    tune_logreg,
)

SCHEMA_KEYS = {
    "model", "features", "best_params", "train_n", "val_n", "test_n",
    "validation_accuracy", "validation_macro_f1", "test_accuracy", "test_macro_f1",
    "per_class_recall", "parameter_count", "inference_latency_ms_per_image",
}


def make_blobs(n_samples, n_features=20, seed=0):
    """Ten Gaussian clusters, one per Fashion-MNIST label."""
    rng = np.random.default_rng(seed)
    centers = np.random.default_rng(42).normal(scale=3.0, size=(10, n_features))
    y = np.arange(n_samples, dtype=np.int64) % 10
    X = (centers[y] + rng.normal(size=(n_samples, n_features))).astype(np.float32)
    return X, y


class LogRegTests(unittest.TestCase):
    def setUp(self):
        self.X_train, self.y_train = make_blobs(400, seed=0)
        self.X_val, self.y_val = make_blobs(200, seed=1)

    def test_build_pipeline_steps_and_params(self):
        pipeline = build_logreg_pipeline(C=0.5, max_iter=200)
        self.assertEqual([name for name, _ in pipeline.steps], ["scaler", "model"])
        model = pipeline.named_steps["model"]
        self.assertEqual((model.C, model.solver, model.max_iter, model.random_state),
                         (0.5, "lbfgs", 200, 42))

    def test_tune_returns_best_candidate_without_convergence_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            best_C, results = tune_logreg(
                self.X_train, self.y_train, self.X_val, self.y_val, c_grid=(0.01, 1.0),
            )
        self.assertEqual([result["C"] for result in results], [0.01, 1.0])
        self.assertEqual(best_C, max(results, key=lambda r: r["val_macro_f1"])["C"])
        for result in results:
            self.assertEqual(set(result), {"C", "val_accuracy", "val_macro_f1", "n_iter", "fit_seconds"})
            self.assertTrue(0.0 <= result["val_macro_f1"] <= 1.0)
            self.assertLess(result["n_iter"], 1000)
        self.assertGreater(max(r["val_accuracy"] for r in results), 0.9)

    def test_tune_breaks_ties_with_first_candidate(self):
        constant = {"accuracy": 0.5, "macro_f1": 0.5}
        with patch("aml.models.logreg.compute_metrics", return_value=constant):
            best_C, _ = tune_logreg(
                self.X_train, self.y_train, self.X_val, self.y_val, c_grid=(10.0, 0.1),
            )
        self.assertEqual(best_C, 10.0)

    def test_tune_rejects_empty_grid(self):
        with self.assertRaises(ValueError):
            tune_logreg(self.X_train, self.y_train, self.X_val, self.y_val, c_grid=())

    def test_evaluate_final_counts_parameters_and_latency(self):
        pipeline, metrics, latency_ms, param_count = evaluate_final_logreg(
            self.X_train, self.y_train, self.X_val, self.y_val, best_C=1.0,
        )
        self.assertEqual(pipeline.named_steps["model"].coef_.shape, (10, 20))
        self.assertEqual(param_count, 20 * 10 + 10)
        self.assertGreater(latency_ms, 0.0)
        self.assertEqual(len(metrics["per_class_recall"]), 10)
        self.assertEqual(np.asarray(metrics["confusion_matrix"]).sum(), len(self.y_val))

    def test_result_record_matches_schema_and_round_trips(self):
        tuning = [
            {"C": 0.1, "val_accuracy": 0.80, "val_macro_f1": 0.79},
            {"C": 1.0, "val_accuracy": 0.85, "val_macro_f1": 0.84},
        ]
        test_metrics = {
            "accuracy": 0.83, "macro_f1": 0.82,
            "per_class_recall": [0.8] * 10, "confusion_matrix": np.eye(10, dtype=int).tolist(),
        }
        record = make_result_record(
            "hog", 1.0, tuning, test_metrics, latency_ms=0.01, param_count=np.int64(3250),
            train_n=48000, val_n=12000, test_n=10000,
        )
        self.assertTrue(SCHEMA_KEYS <= set(record))
        self.assertEqual(record["best_params"], {"C": 1.0, "solver": "lbfgs"})
        self.assertEqual((record["validation_accuracy"], record["validation_macro_f1"]), (0.85, 0.84))
        self.assertIsInstance(record["parameter_count"], int)

        with TemporaryDirectory() as directory:
            path = save_result(record, Path(directory) / "results")
            self.assertEqual(path.name, "logreg_hog.json")
            self.assertEqual(json.loads(path.read_text()), record)


if __name__ == "__main__":
    unittest.main()
