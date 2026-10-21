# Individual Work Guide: Krishna Gehlot

**Project 11:** Automatic Clothing Category Recognition  
**Phase:** Phase 1 — Applied Machine Learning (AML) Baseline  
**Role:** Model 1 — Logistic Regression & Weight Coefficient Heatmap Analysis  
**GitHub Account:** [`NSTKrishna`](https://github.com/NSTKrishna)  
**Order of Work:** **Milestone 3 (Parallel Model Development)**  
**Branch:** `feat/krishna-logreg`  
**PR Review & Merge:** **Open PR against `main` for Anant Jain ([`anantjain-afk`](https://github.com/anantjain-afk)) to review and merge.**  

---

## 1. Scope & Owned Files

You exclusively own the Logistic Regression classifier pipeline, its hyperparameter search across both feature sets, the weight coefficient visual analysis, and the resulting performance JSON files:

- `src/aml/models/logreg.py` (Pipeline definition and tuning functions)
- `notebooks/02_logistic_regression.ipynb` (Experiment notebook & weight visualization)
- `reports/results/logreg_pixels.json` & `reports/results/logreg_hog.json` (Exported metrics)
- `reports/figures/logreg/` (Exported weight heatmaps)

> [!NOTE]
> Because you only edit files in the paths above, your PR will have **zero merge conflicts** with your teammates' models!

---

## 2. Step-by-Step Implementation Guide

### Step 1: Branch from Updated Main
Once Milestone 2 is merged by Anant:
```bash
git checkout main
git pull origin main
git checkout -b feat/krishna-logreg
```

*(Optional early start: If Kartik's PR is pending, you can prototype `logreg.py` using synthetic data like `np.random.rand(100, 784)` to verify your logic.)*

### Step 2: Implement Logistic Regression (`src/aml/models/logreg.py`)
```python
"""Logistic Regression modeling and hyperparameter tuning."""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from aml.evaluation import compute_metrics, measure_inference_latency

def build_logreg_pipeline(C=1.0, solver="lbfgs", max_iter=1000):
    """
    Constructs a Pipeline with StandardScaler and multinomial LogisticRegression.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=C,
            solver=solver,
            max_iter=max_iter,
            random_state=42
        ))
    ])

def tune_logreg(X_train, y_train, X_val, y_val, c_grid=[0.1, 1.0, 10.0]):
    """
    Evaluates Logistic Regression across a grid of C values.
    Returns:
        best_C: selected C with highest validation macro F1
        results: list of candidate metric dictionaries
    """
    results = []
    best_C = c_grid[0]
    best_macro_f1 = -1.0
    
    for C in c_grid:
        print(f"Fitting Logistic Regression with C={C}...")
        pipeline = build_logreg_pipeline(C=C)
        pipeline.fit(X_train, y_train)
        
        val_preds = pipeline.predict(X_val)
        metrics = compute_metrics(y_val, val_preds)
        
        results.append({
            "C": C,
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"]
        })
        
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_C = C
            
    return best_C, results

def evaluate_final_logreg(X_train_full, y_train_full, X_test, y_test, best_C):
    """
    Fits optimal pipeline on all 60,000 training images and evaluates on 10,000 test set.
    """
    pipeline = build_logreg_pipeline(C=best_C)
    pipeline.fit(X_train_full, y_train_full)
    
    test_preds = pipeline.predict(X_test)
    metrics = compute_metrics(y_test, test_preds)
    latency_ms = measure_inference_latency(pipeline, X_test[:1000])
    
    # Model parameters: (num_features * num_classes) + num_classes intercepts
    raw_model = pipeline.named_steps["model"]
    param_count = raw_model.coef_.size + raw_model.intercept_.size
    
    return pipeline, metrics, latency_ms, param_count
```

### Step 3: Run Experiments in `notebooks/02_logistic_regression.ipynb`
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
   - Run `tune_logreg` on $C \in [0.1, 1.0, 10.0]$ for Raw Pixels.
   - Run `tune_logreg` on $C \in [0.1, 1.0, 10.0]$ for HOG Descriptors.
3. **Fit on Full 60,000 & Test on 10,000:**
   - Measure Test Accuracy, Test Macro F1, Per-class Recall, Latency (ms), and Parameter Count.
   - Note: Pixels parameter count = $784 \times 10 + 10 = 7,850$. HOG parameter count = $324 \times 10 + 10 = 3,250$.
4. **Weight Coefficient Heatmaps:**
   - Extract the $10 \times 784$ weight matrix from `pipeline_pix.named_steps["model"].coef_`.
   - For each of the 10 classes, reshape its 784 weights back into a $28 \times 28$ matrix.
   - Plot the 10 heatmaps using a diverging colormap (`plt.imshow(w, cmap="coolwarm")`).
   - Discuss what you see: Positive weights (red) highlight the template silhouette for that category (e.g. trouser legs, ankle boot height, bag handles), while negative weights (blue) penalize pixels characteristic of other classes.
   - Save figures in `reports/figures/logreg/`.
5. **Export Standardized JSON Results:**
   Save outputs to `reports/results/logreg_pixels.json` and `reports/results/logreg_hog.json`.

---

## 3. Verification & Acceptance Checklist

- [ ] `build_logreg_pipeline()` converges cleanly without `ConvergenceWarning`.
- [ ] Tuning over $C \in [0.1, 1.0, 10.0]$ completed for both Pixels and HOG.
- [ ] Weight heatmaps generated for all 10 garment classes.
- [ ] `reports/results/logreg_pixels.json` and `reports/results/logreg_hog.json` exist and match the standardized schema.
- [ ] Notebook `notebooks/02_logistic_regression.ipynb` runs end-to-end.

---

## 4. Git & PR Submission Instructions

```bash
git add src/aml/models/logreg.py notebooks/02_logistic_regression.ipynb reports/results/logreg_*.json reports/figures/logreg/
git commit -m "feat(logreg): implement logistic regression tuning, weight visualizations, and result JSONs"
git push -u origin feat/krishna-logreg
```
Open a Pull Request pointing to `main` and request **Anant Jain** (`anantjain-afk`) to review and merge.
