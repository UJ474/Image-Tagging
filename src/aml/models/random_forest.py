"""Random Forest modeling, depth tuning, and feature importance analysis."""

import json
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from aml.evaluation import compute_metrics, count_parameters, measure_inference_latency

N_ESTIMATORS_GRID = (50, 100, 200)
MAX_DEPTH_GRID = (10, 20, None)


def build_rf_pipeline(n_estimators=100, max_depth=20, n_jobs=-1):
    """Return a RandomForestClassifier that trains trees in parallel.

    n_jobs=-1 uses every available core. random_state=42 fixes the bootstrap
    samples and the feature subset drawn at each split.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=n_jobs,
    )


def tune_rf(
    X_train,
    y_train,
    X_val,
    y_val,
    n_estimators_grid=N_ESTIMATORS_GRID,
    max_depth_grid=MAX_DEPTH_GRID,
):
    """Grid-search the number of trees and the maximum tree depth.

    Every candidate is scored on the training split and the validation split
    so the depth sweep can show overfitting. The selected pair is the one
    with the highest validation macro F1. Ties keep the earlier candidate,
    which in the default grids is the smaller forest and then the shallower
    depth.

    Returns
    -------
    best_params : dict
        Keys n_estimators and max_depth. max_depth is None when the winning
        forest is grown without a depth limit.
    results : list of dict
        One entry per candidate, with train and validation accuracy and macro F1.
    """
    if len(n_estimators_grid) == 0 or len(max_depth_grid) == 0:
        raise ValueError("n_estimators_grid and max_depth_grid must be nonempty.")

    results = []
    best_params = {"n_estimators": 100, "max_depth": 20}
    best_val_macro_f1 = -1.0

    for n_estimators in n_estimators_grid:
        for max_depth in max_depth_grid:
            print(f"Training Random Forest: n_estimators={n_estimators}, max_depth={max_depth}...")
            model = build_rf_pipeline(n_estimators=n_estimators, max_depth=max_depth)
            model.fit(X_train, y_train)

            train_metrics = compute_metrics(y_train, model.predict(X_train))
            val_metrics = compute_metrics(y_val, model.predict(X_val))
            results.append({
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            })

            if val_metrics["macro_f1"] > best_val_macro_f1:
                best_val_macro_f1 = val_metrics["macro_f1"]
                best_params = {"n_estimators": n_estimators, "max_depth": max_depth}

    return best_params, results


def evaluate_final_rf(X_train_full, y_train_full, X_test, y_test, best_params):
    """Refit the selected forest on all 60,000 images and score the test set.

    Returns the fitted forest, the compute_metrics() dictionary, the median
    per-image latency in milliseconds on the first 1,000 test images, and the
    total node count across trees. That node count is the team's
    parameter_count for a random forest.
    """
    model = build_rf_pipeline(**best_params)
    model.fit(X_train_full, y_train_full)

    metrics = compute_metrics(y_test, model.predict(X_test))
    latency_ms = measure_inference_latency(model, X_test[:1000])
    total_nodes = count_parameters(model)
    return model, metrics, latency_ms, total_nodes


def make_result_record(features, best_params, tuning_results, test_metrics, latency_ms,
                       total_nodes, train_n, val_n, test_n):
    """Assemble the team's standardized result dictionary.

    Validation scores come from the selected depth and tree count. Test scores
    come from the refit on the full training set. max_depth is JSON null when
    the selected forest has no depth limit. The test confusion matrix is
    included for the cross-model error analysis.
    """
    best = next(
        result for result in tuning_results
        if result["n_estimators"] == best_params["n_estimators"]
        and result["max_depth"] == best_params["max_depth"]
    )
    max_depth = best_params["max_depth"]
    return {
        "model": "rf",
        "features": features,
        "best_params": {
            "n_estimators": int(best_params["n_estimators"]),
            "max_depth": None if max_depth is None else int(max_depth),
        },
        "train_n": int(train_n),
        "val_n": int(val_n),
        "test_n": int(test_n),
        "validation_accuracy": float(best["val_accuracy"]),
        "validation_macro_f1": float(best["val_macro_f1"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "per_class_recall": [float(value) for value in test_metrics["per_class_recall"]],
        "parameter_count": int(total_nodes),
        "inference_latency_ms_per_image": float(latency_ms),
        "test_confusion_matrix": test_metrics["confusion_matrix"],
    }


def save_result(record, results_dir="reports/results"):
    """Write record to <results_dir>/rf_<features>.json and return the path."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{record['model']}_{record['features']}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path
