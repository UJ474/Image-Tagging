# Individual Work Guide: Vipul Sharma

**Project 11:** Automatic Clothing Category Recognition  
**Phase:** Phase 1 — Applied Machine Learning (AML) Baseline  
**Role:** Model 2 — $k$-Nearest Neighbors ($k$-NN) & Latency/Scaling Analysis  
**GitHub Account:** [`vs-code06`](https://github.com/vs-code06)  
**Order of Work:** **Milestone 3 (Parallel Model Development)**  
**Branch:** `feat/vipul-knn`  
**PR Review & Merge:** **Open PR against `main` for Anant Jain ([`anantjain-afk`](https://github.com/anantjain-afk)) to review and merge.**  

---

## 1. Scope & Owned Files

You exclusively own the $k$-Nearest Neighbors ($k$-NN) classifier, its hyperparameter tuning on both representations, the computational latency and memory scaling analysis, and the resulting performance JSON files:

- `src/aml/models/knn.py` (Pipeline definition, batched prediction, and tuning functions)
- `notebooks/04_knn_experiments.ipynb` (Experiment notebook & latency study)
- `reports/results/knn_pixels.json` & `reports/results/knn_hog.json` (Exported metrics)
- `reports/figures/knn/` (Exported latency and dimensional scaling charts)

> [!NOTE]
> Because you only edit files in the paths above, your PR will have **zero merge conflicts** with your teammates' models!

---

## 2. Step-by-Step Implementation Guide

### Step 1: Branch from Updated Main
Once Milestone 2 is merged by Anant:
```bash
git checkout main
git pull origin main
git checkout -b feat/vipul-knn
```

*(Optional early start: If Kartik's PR is pending, you can prototype `knn.py` using synthetic data like `np.random.rand(100, 784)` to verify your logic.)*

### Step 2: Implement $k$-NN Pipeline (`src/aml/models/knn.py`)
```python
"""k-Nearest Neighbors modeling, batched inference, and latency profiling."""
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from aml.evaluation import compute_metrics, measure_inference_latency

def build_knn_pipeline(n_neighbors=5, weights="uniform", algorithm="brute"):
    """
    Constructs KNeighborsClassifier with Euclidean metric.
    """
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        metric="euclidean"
    )

def predict_in_batches(model, X, batch_size=500):
    """
    Batched prediction to avoid memory exhaustion during distance matrix computation.
    """
    n_samples = X.shape[0]
    preds = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_pred = model.predict(X[start:end])
        preds.append(batch_pred)
    return np.concatenate(preds)

def tune_knn(X_train, y_train, X_val, y_val, k_grid=[3, 5, 7, 9], weights_grid=["uniform", "distance"]):
    """
    Tunes k and weighting scheme on validation split.
    Returns:
        best_params: dict of winning hyperparams
        results: list of candidate metric dictionaries
    """
    results = []
    best_params = {"n_neighbors": 5, "weights": "uniform"}
    best_macro_f1 = -1.0
    
    for k in k_grid:
        for w in weights_grid:
            print(f"Evaluating k-NN with k={k}, weights={w}...")
            model = build_knn_pipeline(n_neighbors=k, weights=w)
            model.fit(X_train, y_train)
            
            val_preds = predict_in_batches(model, X_val, batch_size=500)
            metrics = compute_metrics(y_val, val_preds)
            
            results.append({
                "n_neighbors": k,
                "weights": w,
                "val_accuracy": metrics["accuracy"],
                "val_macro_f1": metrics["macro_f1"]
            })
            
            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                best_params = {"n_neighbors": k, "weights": w}
                
    return best_params, results

def evaluate_final_knn(X_train_full, y_train_full, X_test, y_test, best_params):
    """
    Fits k-NN on all 60,000 training images and evaluates on 10,000 test set.
    """
    model = build_knn_pipeline(**best_params)
    model.fit(X_train_full, y_train_full)
    
    test_preds = predict_in_batches(model, X_test, batch_size=500)
    metrics = compute_metrics(y_test, test_preds)
    
    # Latency timing on 1,000 samples
    start = time.perf_counter()
    _ = predict_in_batches(model, X_test[:1000], batch_size=200)
    total_sec = time.perf_counter() - start
    latency_ms = (total_sec / 1000.0) * 1000.0
    
    # Non-parametric memory complexity (MB): samples * features * 4 bytes
    param_mb = (X_train_full.shape[0] * X_train_full.shape[1] * 4) / (1024 * 1024)
    
    return model, metrics, latency_ms, param_mb
```

### Step 3: Run Experiments in `notebooks/04_knn_experiments.ipynb`
1. **Load Data & Features:**
   ```python
   from aml.data import get_train_val_test_splits
   from aml.features import get_features
   
   splits = get_train_val_test_splits()
   X_train_pix = get_features(splits["X_train"], "pixels", "data/cache/train_pixels.npz")
   X_val_pix   = get_features(splits["X_val"],   "pixels", "data/cache/val_pixels.npz")
   X_test_pix  = get_features(splits["X_test"],  "pixels", "data/cache/test_pixels.npz")
   X_train_full_pix = get_features(splits["X_train_full"], "pixels", "data/cache/train_full_pixels.npz")
   
   X_train_hog = get_features(splits["X_train"], "hog", "data/cache/train_hog.npz")
   X_val_hog   = get_features(splits["X_val"],   "hog", "data/cache/val_hog.npz")
   X_test_hog  = get_features(splits["X_test"],  "hog", "data/cache/test_hog.npz")
   X_train_full_hog = get_features(splits["X_train_full"], "hog", "data/cache/train_full_hog.npz")
   ```
2. **Execute Grid Searches:**
   - Tune $k \in [3, 5, 7, 9]$ and weights (`uniform` vs `distance`) on Raw Pixels.
   - Tune $k \in [3, 5, 7, 9]$ and weights (`uniform` vs `distance`) on HOG Descriptors.
3. **Fit on Full 60,000 & Test on 10,000:**
   - Record Test Accuracy, Test Macro F1, Per-class Recall, and Confusion Matrix.
4. **In-Depth Latency & Dimensionality Study:**
   - **Prediction Speed:** Time inference for 1,000 images on Pixels ($784$-d) vs. HOG ($324$-d). Plot latency per batch.
   - **The Curse of Dimensionality:** Explain why Euclidean distance becomes less discriminative as dimensionality grows, and observe whether HOG's reduced dimensionality improves both distance quality and test runtime.
   - **Asymmetric Computational Cost:** Highlight that while $k$-NN training time is $\approx 0$ seconds (lazy indexing), query-time inference requires $\mathcal{O}(N_{\text{train}} \times d)$ distance computations per image, making $k$-NN severely bottlenecked for real-time inventory systems.
   - **Memory Footprint:** 
     - Storing $60,000 \times 784 \times 4 \text{ bytes} \approx 188.16 \text{ MB}$.
     - Storing $60,000 \times 324 \times 4 \text{ bytes} \approx 77.76 \text{ MB}$ (~58% RAM savings).
   - Save figures in `reports/figures/knn/`.
5. **Export Standardized JSON Results:**
   Save outputs to `reports/results/knn_pixels.json` and `reports/results/knn_hog.json`.

---

## 3. Verification & Acceptance Checklist

- [ ] `predict_in_batches()` processes test images cleanly without memory errors.
- [ ] Grid search over $k \in [3, 5, 7, 9]$ and weights (`uniform` vs `distance`) completed for both representations.
- [ ] Latency and dimensional scaling curves plotted and explained.
- [ ] `reports/results/knn_pixels.json` and `reports/results/knn_hog.json` exist and match the standardized schema.
- [ ] Notebook `notebooks/04_knn_experiments.ipynb` executes completely from top to bottom.

---

## 4. Git & PR Submission Instructions

```bash
git add src/aml/models/knn.py notebooks/04_knn_experiments.ipynb reports/results/knn_*.json reports/figures/knn/
git commit -m "feat(knn): implement k-NN tuning, latency analysis, and result JSONs"
git push -u origin feat/vipul-knn
```
Open a Pull Request pointing to `main` and request **Anant Jain** (`anantjain-afk`) to review and merge.
