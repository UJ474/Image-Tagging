# Individual Work Guide: Aditya

**Project 11:** Automatic Clothing Category Recognition  
**Phase:** Phase 1 — Applied Machine Learning (AML) Baseline  
**Role:** Model 3 — Random Forest Classifier & Overfitting / Feature Importance Analysis  
**GitHub Account:** [`AdityaInnovates`](https://github.com/AdityaInnovates)  
**Order of Work:** **Milestone 3 (Parallel Model Development)**  
**Branch:** `feat/aditya-rf`  
**PR Review & Merge:** **Open PR against `main` for Anant Jain ([`anantjain-afk`](https://github.com/anantjain-afk)) to review and merge.**  

---

## 1. Scope & Owned Files

You exclusively own the Random Forest classifier pipeline, its hyperparameter search across both feature sets, the tree depth/overfitting analysis, feature importance mapping, and the resulting performance JSON files:

- `src/aml/models/random_forest.py` (Pipeline definition and tuning functions)
- `notebooks/05_random_forest.ipynb` (Experiment notebook & feature importance study)
- `reports/results/rf_pixels.json` & `reports/results/rf_hog.json` (Exported metrics)
- `reports/figures/rf/` (Exported overfitting curves and importance heatmaps)

> [!NOTE]
> Because you only edit files in the paths above, your PR will have **zero merge conflicts** with your teammates' models!

---

## 2. Step-by-Step Implementation Guide

### Step 1: Branch from Updated Main
Once Milestone 2 is merged by Anant:
```bash
git checkout main
git pull origin main
git checkout -b feat/aditya-rf
```

*(Optional early start: If Kartik's PR is pending, you can prototype `random_forest.py` using synthetic data like `np.random.rand(100, 784)` to verify your logic.)*

### Step 2: Implement Random Forest Pipeline (`src/aml/models/random_forest.py`)
```python
"""Random Forest modeling, depth tuning, and feature importance analysis."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from aml.evaluation import compute_metrics, measure_inference_latency

def build_rf_pipeline(n_estimators=100, max_depth=20, n_jobs=-1):
    """
    Constructs a parallelized RandomForestClassifier.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=n_jobs
    )

def tune_rf(X_train, y_train, X_val, y_val, n_estimators_grid=[50, 100, 200], max_depth_grid=[10, 20, None]):
    """
    Grid search over number of trees and max tree depth.
    Records both train and validation performance to inspect overfitting.
    """
    results = []
    best_params = {"n_estimators": 100, "max_depth": 20}
    best_val_macro_f1 = -1.0
    
    for n in n_estimators_grid:
        for depth in max_depth_grid:
            print(f"Training Random Forest: n_estimators={n}, max_depth={depth}...")
            model = build_rf_pipeline(n_estimators=n, max_depth=depth)
            model.fit(X_train, y_train)
            
            train_preds = model.predict(X_train)
            val_preds = model.predict(X_val)
            
            train_metrics = compute_metrics(y_train, train_preds)
            val_metrics = compute_metrics(y_val, val_preds)
            
            results.append({
                "n_estimators": n,
                "max_depth": depth,
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"]
            })
            
            if val_metrics["macro_f1"] > best_val_macro_f1:
                best_val_macro_f1 = val_metrics["macro_f1"]
                best_params = {"n_estimators": n, "max_depth": depth}
                
    return best_params, results

def evaluate_final_rf(X_train_full, y_train_full, X_test, y_test, best_params):
    """
    Fits optimal forest on all 60,000 training images and evaluates on 10,000 test set.
    """
    model = build_rf_pipeline(**best_params)
    model.fit(X_train_full, y_train_full)
    
    test_preds = model.predict(X_test)
    metrics = compute_metrics(y_test, test_preds)
    latency_ms = measure_inference_latency(model, X_test[:1000])
    
    # Tree complexity: sum of node counts across all trees
    total_nodes = sum(tree.tree_.node_count for tree in model.estimators_)
    
    return model, metrics, latency_ms, total_nodes
```

### Step 3: Run Experiments in `notebooks/05_random_forest.ipynb`
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
   - Tune `n_estimators` $\in [50, 100, 200]$ and `max_depth` $\in [10, 20, \text{None}]$ on Pixels.
   - Tune `n_estimators` $\in [50, 100, 200]$ and `max_depth` $\in [10, 20, \text{None}]$ on HOG.
3. **Overfitting Analysis (Bias-Variance Tradeoff):**
   - Plot Train Accuracy vs. Validation Accuracy across `max_depth` ($10$, $20$, and $\text{None}$).
   - Notice that when `max_depth=None`, the forest memorizes the training data ($>99.8\%$ accuracy), but validation accuracy tops out around $88\%$, visually demonstrating the generalization gap.
4. **Feature Importance Mapping:**
   - Extract MDI (Mean Decrease in Impurity / Gini) feature importances from `model.feature_importances_`.
   - **Pixels Importance Heatmap:** Reshape the 784 importances into a $28 \times 28$ image (`plt.imshow(importances, cmap="viridis")`). Observe which pixel locations have the highest split frequency (center torso mass, trouser leg separation, shoe boundaries).
   - **HOG Importance:** Identify top gradient orientation blocks.
   - Save figures in `reports/figures/rf/`.
5. **Export Standardized JSON Results:**
   Save outputs to `reports/results/rf_pixels.json` and `reports/results/rf_hog.json`.

---

## 3. Verification & Acceptance Checklist

- [ ] `build_rf_pipeline()` successfully utilizes multi-core processing (`n_jobs=-1`).
- [ ] Tuning matrix across tree counts and depths completed for both representations.
- [ ] Train vs. validation overfitting curve plotted and interpreted.
- [ ] $28 \times 28$ Gini feature importance heatmap rendered and saved.
- [ ] `reports/results/rf_pixels.json` and `reports/results/rf_hog.json` exist and match the standardized schema.
- [ ] Notebook `notebooks/05_random_forest.ipynb` executes completely from top to bottom.

---

## 4. Git & PR Submission Instructions

```bash
git add src/aml/models/random_forest.py notebooks/05_random_forest.ipynb reports/results/rf_*.json reports/figures/rf/
git commit -m "feat(rf): implement random forest tuning, overfitting curves, and feature importance"
git push -u origin feat/aditya-rf
```
Open a Pull Request pointing to `main` and request **Anant Jain** (`anantjain-afk`) to review and merge.
