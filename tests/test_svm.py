"""Unit tests for the UJ474 Linear SVM and garment error-analysis modules."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np

from aml.error_analysis import (
    extract_misclassifications,
    plot_confusion_matrix,
    plot_garment_confusion_submatrix,
    plot_misclassification_gallery,
)
from aml.models.svm import (
    build_svm_pipeline,
    evaluate_final_svm,
    make_result_record,
    save_result,
    tune_svm,
)


def make_blobs(n_samples, n_features=20, seed=0):
    """Return ten separable synthetic classes in the Fashion-MNIST label range."""
    rng = np.random.default_rng(seed)
    centers = np.random.default_rng(42).normal(scale=3.0, size=(10, n_features))
    y = np.arange(n_samples, dtype=np.int64) % 10
    X = (centers[y] + rng.normal(size=(n_samples, n_features))).astype(np.float32)
    return X, y


class SvmTests(unittest.TestCase):
    def setUp(self):
        self.X_train, self.y_train = make_blobs(400, seed=0)
        self.X_val, self.y_val = make_blobs(200, seed=1)

    def test_build_pipeline_and_input_validation(self):
        pipeline = build_svm_pipeline(C=0.5, max_iter=400)
        self.assertEqual([name for name, _ in pipeline.steps], ["scaler", "model"])
        self.assertEqual(pipeline.named_steps["model"].C, 0.5)
        self.assertEqual(pipeline.named_steps["model"].max_iter, 400)
        with self.assertRaises(ValueError):
            build_svm_pipeline(C=0)
        with self.assertRaises(ValueError):
            build_svm_pipeline(max_iter=0)

    def test_tuning_selects_best_candidate_and_preserves_first_tie(self):
        best_C, results = tune_svm(
            self.X_train, self.y_train, self.X_val, self.y_val, c_grid=(0.1, 1.0),
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(best_C, max(results, key=lambda item: item["val_macro_f1"])["C"])
        self.assertTrue(all(0 <= item["val_macro_f1"] <= 1 for item in results))
        with patch("aml.models.svm.compute_metrics", return_value={"accuracy": 0.5, "macro_f1": 0.5}):
            best_C, _ = tune_svm(self.X_train, self.y_train, self.X_val, self.y_val, c_grid=(10.0, 0.1))
        self.assertEqual(best_C, 10.0)
        with self.assertRaises(ValueError):
            tune_svm(self.X_train, self.y_train, self.X_val, self.y_val, c_grid=())

    def test_final_evaluation_and_result_round_trip(self):
        pipeline, metrics, latency_ms, parameter_count = evaluate_final_svm(
            self.X_train, self.y_train, self.X_val, self.y_val, best_C=1.0,
        )
        self.assertEqual(pipeline.named_steps["model"].coef_.shape, (10, 20))
        self.assertEqual(parameter_count, 10 * 20 + 10)
        self.assertGreater(latency_ms, 0)
        tuning = [{"C": 1.0, "val_accuracy": 0.9, "val_macro_f1": 0.9}]
        record = make_result_record("hog", 1.0, tuning, metrics, latency_ms, parameter_count, 400, 200, 200)
        self.assertEqual(record["best_params"], {"C": 1.0, "max_iter": 2000})
        with TemporaryDirectory() as directory:
            path = save_result(record, Path(directory) / "results")
            self.assertEqual(path.name, "svm_hog.json")
            self.assertEqual(json.loads(path.read_text()), record)


class ErrorAnalysisTests(unittest.TestCase):
    def test_extracts_expected_error_direction(self):
        y_true = np.array([6, 6, 2, 6, 2])
        y_pred = np.array([0, 2, 4, 0, 4])
        images = np.arange(5 * 28 * 28).reshape(5, 28, 28)
        selected, indices = extract_misclassifications(y_true, y_pred, images, 6, 0)
        np.testing.assert_array_equal(indices, [0, 3])
        np.testing.assert_array_equal(selected, images[[0, 3]])
        with self.assertRaises(ValueError):
            extract_misclassifications(y_true, y_pred[:-1], images, 6, 0)

    def test_plots_save_without_zero_row_failures(self):
        cm = np.eye(10, dtype=int)
        cm[6] = 0
        names = [str(index) for index in range(10)]
        images = np.zeros((2, 28, 28), dtype=np.float32)
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            plot_confusion_matrix(cm, names, save_path=directory / "all.png")
            plot_garment_confusion_submatrix(cm, names, save_path=directory / "upper.png")
            plot_misclassification_gallery(images, np.array([4, 7]), "Shirt", "T-shirt/top", directory / "gallery.png")
            self.assertTrue(all((directory / name).is_file() for name in ("all.png", "upper.png", "gallery.png")))


if __name__ == "__main__":
    unittest.main()
