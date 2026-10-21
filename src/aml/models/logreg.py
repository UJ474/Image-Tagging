"""Logistic Regression modeling, hyperparameter tuning, and result export."""

import json
from pathlib import Path
import time

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aml.evaluation import compute_metrics, count_parameters, measure_inference_latency

C_GRID = (0.1, 1.0, 10.0)


def build_logreg_pipeline(C=1.0, solver="lbfgs", max_iter=2000):
    """Return a StandardScaler + multinomial LogisticRegression pipeline.

    With lbfgs, scikit-learn fits one softmax model over all ten classes.
    C is the inverse L2 regularization strength: smaller values shrink the
    weights harder. The scaler is fitted on training data only, inside the
    pipeline, so validation and test images never influence it.

    max_iter defaults to 2000 rather than 1000: on the 48k pixel split,
    lbfgs needs about 1,100 iterations at C=1 and 1,260 at C=10.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=C,
            solver=solver,
            max_iter=max_iter,
            random_state=42,
        )),
    ])


def tune_logreg(X_train, y_train, X_val, y_val, c_grid=C_GRID, **pipeline_kwargs):
    """Fit one pipeline per C on the training split and score it on validation.

    Returns
    -------
    best_C : float
        C with the highest validation macro F1 (the first one wins ties).
    results : list of dict
        One entry per candidate with keys C, val_accuracy, val_macro_f1,
        n_iter (lbfgs iterations used), and fit_seconds.
    """
    if len(c_grid) == 0:
        raise ValueError("c_grid must contain at least one value.")
    results = []
    for C in c_grid:
        print(f"Fitting Logistic Regression with C={C}...")
        pipeline = build_logreg_pipeline(C=C, **pipeline_kwargs)
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start

        metrics = compute_metrics(y_val, pipeline.predict(X_val))
        results.append({
            "C": C,
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "n_iter": int(pipeline.named_steps["model"].n_iter_.max()),
            "fit_seconds": fit_seconds,
        })

    best = max(results, key=lambda result: result["val_macro_f1"])
    return best["C"], results


def evaluate_final_logreg(X_train_full, y_train_full, X_test, y_test, best_C, **pipeline_kwargs):
    """Refit the selected pipeline on all 60,000 images and score the 10,000 test set.

    Returns the fitted pipeline, the compute_metrics() dictionary, the median
    per-image latency in milliseconds (measured on the first 1,000 test
    images), and the parameter count (weights plus intercepts).
    """
    pipeline = build_logreg_pipeline(C=best_C, **pipeline_kwargs)
    pipeline.fit(X_train_full, y_train_full)

    metrics = compute_metrics(y_test, pipeline.predict(X_test))
    latency_ms = measure_inference_latency(pipeline, X_test[:1000])
    # num_features * num_classes weights + num_classes intercepts
    param_count = count_parameters(pipeline)
    return pipeline, metrics, latency_ms, param_count


def make_result_record(features, best_C, tuning_results, test_metrics, latency_ms,
                       param_count, train_n, val_n, test_n, solver="lbfgs"):
    """Assemble the team's standardized result dictionary (docs/team/README.md).

    Validation scores come from the selected C's tuning entry; test scores
    come from the refitted model. The test confusion matrix is included as an
    extra field for the cross-model error analysis.
    """
    best = next(result for result in tuning_results if result["C"] == best_C)
    return {
        "model": "logreg",
        "features": features,
        "best_params": {"C": best_C, "solver": solver},
        "train_n": int(train_n),
        "val_n": int(val_n),
        "test_n": int(test_n),
        "validation_accuracy": best["val_accuracy"],
        "validation_macro_f1": best["val_macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "per_class_recall": test_metrics["per_class_recall"],
        "parameter_count": int(param_count),
        "inference_latency_ms_per_image": latency_ms,
        "test_confusion_matrix": test_metrics["confusion_matrix"],
    }


def save_result(record, results_dir="reports/results"):
    """Write record to <results_dir>/logreg_<features>.json and return the path."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{record['model']}_{record['features']}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path
