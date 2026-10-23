"""Linear SVM modeling, validation tuning, and standardized result export."""

import json
from pathlib import Path
import time

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from aml.evaluation import compute_metrics, count_parameters, measure_inference_latency


C_GRID = (0.1, 1.0, 10.0)


def build_svm_pipeline(C=1.0, max_iter=2_000):
    """Return a scaled, deterministic one-vs-rest linear SVM pipeline.

    Scaling happens inside the pipeline, so the validation or test data can
    never influence the fitted feature statistics. ``dual='auto'`` lets
    scikit-learn choose the appropriate solver formulation for the feature
    matrix while preserving LinearSVC's multiclass one-vs-rest strategy.
    """
    if C <= 0:
        raise ValueError("C must be positive.")
    if max_iter < 1:
        raise ValueError("max_iter must be positive.")
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("model", LinearSVC(C=C, max_iter=max_iter, random_state=42, dual="auto")),
    ])


def tune_svm(X_train, y_train, X_val, y_val, c_grid=C_GRID, **pipeline_kwargs):
    """Fit each C candidate and select the one with best validation macro F1.

    The earliest candidate wins a tie, making the default selection stable and
    slightly favoring the more strongly regularized model.
    """
    if not c_grid:
        raise ValueError("c_grid must contain at least one value.")

    results = []
    for C in c_grid:
        print(f"Fitting Linear SVM with C={C}...")
        pipeline = build_svm_pipeline(C=C, **pipeline_kwargs)
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        metrics = compute_metrics(y_val, pipeline.predict(X_val))
        results.append({
            "C": float(C),
            "val_accuracy": float(metrics["accuracy"]),
            "val_macro_f1": float(metrics["macro_f1"]),
            "n_iter": int(pipeline.named_steps["model"].n_iter_),
            "fit_seconds": float(fit_seconds),
        })

    best = max(results, key=lambda result: result["val_macro_f1"])
    return best["C"], results


def evaluate_final_svm(X_train_full, y_train_full, X_test, y_test, best_C, **pipeline_kwargs):
    """Refit the selected SVM on all training examples and evaluate once."""
    pipeline = build_svm_pipeline(C=best_C, **pipeline_kwargs)
    pipeline.fit(X_train_full, y_train_full)
    metrics = compute_metrics(y_test, pipeline.predict(X_test))
    latency_ms = measure_inference_latency(pipeline, X_test[:1_000])
    return pipeline, metrics, latency_ms, count_parameters(pipeline)


def make_result_record(features, best_C, tuning_results, test_metrics, latency_ms,
                       parameter_count, train_n, val_n, test_n, max_iter=2_000):
    """Assemble a JSON-ready record compatible with the team's result schema."""
    if features not in {"pixels", "hog"}:
        raise ValueError("features must be 'pixels' or 'hog'.")
    best = next(result for result in tuning_results if result["C"] == best_C)
    return {
        "model": "svm",
        "features": features,
        "best_params": {"C": float(best_C), "max_iter": int(max_iter)},
        "train_n": int(train_n),
        "val_n": int(val_n),
        "test_n": int(test_n),
        "validation_accuracy": float(best["val_accuracy"]),
        "validation_macro_f1": float(best["val_macro_f1"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "per_class_recall": [float(value) for value in test_metrics["per_class_recall"]],
        "parameter_count": int(parameter_count),
        "inference_latency_ms_per_image": float(latency_ms),
        "test_confusion_matrix": test_metrics["confusion_matrix"],
    }


def save_result(record, results_dir="reports/results"):
    """Write a result record to ``svm_<features>.json`` and return its path."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{record['model']}_{record['features']}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path
