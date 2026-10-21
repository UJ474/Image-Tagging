"""Exercise the notebook with temporary reports; never publish mock scores."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/00_master_aml_comparison.ipynb"


class NotebookTests(unittest.TestCase):
    def run_notebook(self, results_dir):
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(notebook)
        namespace = {}
        first = True
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            with patch("matplotlib.pyplot.show"):
                exec(compile(cell.source, str(NOTEBOOK), "exec"), namespace)
            if first:
                namespace["RESULTS_DIR"] = Path(results_dir)
                namespace["display"] = lambda *args, **kwargs: None
                first = False
        self.addCleanup(plt.close, "all")
        return namespace

    def write_reports(self, directory):
        for model in ("logreg", "knn", "rf", "svm"):
            for features in ("pixels", "hog"):
                result = {
                    "model": model, "features": features, "best_params": {},
                    "train_n": 48000, "val_n": 12000, "test_n": 10000,
                    "validation_accuracy": .8, "validation_macro_f1": .79,
                    "test_accuracy": .78, "test_macro_f1": .77,
                    "per_class_recall": [.78] * 10,
                    "parameter_count": 179.443 if model == "knn" else 7850,
                    "inference_latency_ms_per_image": .1,
                }
                (Path(directory) / f"{model}_{features}.json").write_text(json.dumps(result))

    def test_empty_reports_show_pending_without_comparison(self):
        with TemporaryDirectory() as directory:
            state = self.run_notebook(directory)
        self.assertFalse(state["complete"])
        self.assertTrue(state["summary"].empty)
        self.assertEqual(state["status_table"]["Status"].tolist(), ["Missing"] * 8)

    def test_all_reports_produce_table_charts_and_findings(self):
        with TemporaryDirectory() as directory:
            self.write_reports(directory)
            state = self.run_notebook(directory)
        self.assertTrue(state["complete"])
        self.assertEqual(state["summary"].shape, (8, 7))
        self.assertEqual(len(state["axes"]), 2)
        self.assertEqual(len(state["axes"][0].patches), 8)
        self.assertIn("MiB", state["summary"].iloc[2]["Complexity"])
        self.assertIn("Shirt recall", "\n".join(state["lines"]))
        self.assertIn("<table", state["summary"].style.to_html())

    def test_invalid_score_blocks_final_comparison(self):
        with TemporaryDirectory() as directory:
            self.write_reports(directory)
            path = Path(directory) / "svm_hog.json"
            result = json.loads(path.read_text())
            result["test_accuracy"] = 89  # Percentage instead of fraction.
            path.write_text(json.dumps(result))
            state = self.run_notebook(directory)
        self.assertFalse(state["complete"])
        self.assertEqual(len(state["records"]), 7)
        self.assertTrue(state["status_table"].iloc[-1]["Status"].startswith("Invalid:"))


if __name__ == "__main__":
    unittest.main()
