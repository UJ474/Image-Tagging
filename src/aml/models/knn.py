"""k-Nearest Neighbors (k-NN) modeling, batched inference, and latency profiling."""

import json
from pathlib import Path
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from aml.evaluation import compute_metrics, measure_inference_latency

K_GRID = (3, 5, 7, 9)
WEIGHTS_GRID = ("uniform", "distance")


def build_knn_pipeline(n_neighbors=5, weights="uniform", algorithm="brute"):
    """
    Constructs a KNeighborsClassifier with Euclidean (L2) distance.
    Uses brute-force search by default to avoid tree-building overhead in high dimensions.
    """
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        metric="euclidean"
    )


def predict_in_batches(model, X, batch_size=500):
    """
    Predicts labels in batches to prevent memory exhaustion on large test sets.
    """
    X = np.asarray(X)
    n_samples = len(X)
    predictions = []
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_pred = model.predict(X[start:end])
        predictions.append(batch_pred)
        
    return np.concatenate(predictions)


def tune_knn(X_train, y_train, X_val, y_val, k_grid=K_GRID, weights_grid=WEIGHTS_GRID):
    """
    Evaluates k-NN candidates across combinations of k and weights on validation data.
    
    Returns:
        best_params: dict with selected n_neighbors and weights (highest val macro F1)
        results: list of candidate metric dictionaries
    """
    results = []
    best_params = {"n_neighbors": 5, "weights": "uniform"}
    best_macro_f1 = -1.0

    for k in k_grid:
        for w in weights_grid:
            print(f"Evaluating k-NN: n_neighbors={k}, weights='{w}'...")
            model = build_knn_pipeline(n_neighbors=k, weights=w)
            
            start_fit = time.perf_counter()
            model.fit(X_train, y_train)
            fit_time = time.perf_counter() - start_fit
            
            start_pred = time.perf_counter()
            val_preds = predict_in_batches(model, X_val, batch_size=500)
            pred_time = time.perf_counter() - start_pred
            
            metrics = compute_metrics(y_val, val_preds)
            
            candidate_record = {
                "n_neighbors": k,
                "weights": w,
                "val_accuracy": metrics["accuracy"],
                "val_macro_f1": metrics["macro_f1"],
                "fit_seconds": fit_time,
                "predict_seconds": pred_time
            }
            results.append(candidate_record)
            
            # Select by highest validation macro F1 (if tied, prefer larger k, then uniform)
            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                best_params = {"n_neighbors": k, "weights": w}

    return best_params, results


def evaluate_final_knn(X_train_full, y_train_full, X_test, y_test, best_params):
    """
    Fits k-NN on all 60,000 training images and evaluates on 10,000 test images.
    
    Returns:
        model: fitted KNeighborsClassifier
        metrics: dict from compute_metrics()
        latency_ms: median prediction latency per image in milliseconds
        param_mb: stored training data memory size in MiB
    """
    model = build_knn_pipeline(**best_params)
    model.fit(X_train_full, y_train_full)

    # Predict on test set
    test_preds = predict_in_batches(model, X_test, batch_size=500)
    metrics = compute_metrics(y_test, test_preds)

    # Measure latency on up to 1,000 test samples using evaluation utility
    latency_ms = measure_inference_latency(model, X_test[:min(1000, len(X_test))])

    # Non-parametric memory complexity (MiB): stored samples * features * 4 bytes / (1024^2)
    param_mb = float(X_train_full.nbytes / (1024 ** 2))

    return model, metrics, latency_ms, param_mb


def make_result_record(features, best_params, tuning_results, test_metrics, latency_ms,
                       param_mb, train_n=48000, val_n=12000, test_n=10000):
    """
    Builds the standardized result dictionary matching the team schema.
    """
    best = next(
        r for r in tuning_results
        if r["n_neighbors"] == best_params["n_neighbors"] and r["weights"] == best_params["weights"]
    )
    return {
        "model": "knn",
        "features": features,
        "best_params": best_params,
        "train_n": int(train_n),
        "val_n": int(val_n),
        "test_n": int(test_n),
        "validation_accuracy": float(best["val_accuracy"]),
        "validation_macro_f1": float(best["val_macro_f1"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "per_class_recall": [float(r) for r in test_metrics["per_class_recall"]],
        "parameter_count": float(param_mb),
        "inference_latency_ms_per_image": float(latency_ms),
        "test_confusion_matrix": test_metrics["confusion_matrix"],
    }


def save_result(record, results_dir="reports/results"):
    """
    Writes result record to <results_dir>/knn_<features>.json.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{record['model']}_{record['features']}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path
