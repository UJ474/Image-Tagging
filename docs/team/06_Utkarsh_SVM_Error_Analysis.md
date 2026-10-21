# Individual Work Guide: Utkarsh Jain

**Project 11:** Automatic Clothing Category Recognition  
**Phase:** Phase 1 — Applied Machine Learning (AML) Baseline  
**Role:** Model 4 — Support Vector Machine (SVM) & Garment Confusion Error Analysis  
**GitHub Account:** [`UJ474`](https://github.com/UJ474)  
**Order of Work:** **Milestone 3 (Parallel Model Development) & Milestone 4 (Error Analysis Report)**  
**Branch:** `feat/utkarsh-svm-error-analysis`  
**PR Review & Merge:** **Open PR against `main` for Anant Jain ([`anantjain-afk`](https://github.com/anantjain-afk)) to review and merge.**  

---

## 1. Scope & Owned Files

You exclusively own the Support Vector Machine (Linear SVM) pipeline, the cross-model error analysis tools, the visual misclassification gallery, and the core Phase 1 synthesis report that articulates "The Gap" to motivate Phase 2:

- `src/aml/models/svm.py` (Pipeline definition and tuning functions)
- `src/aml/error_analysis.py` (Confusion matrix plotting and misclassification extractors)
- `notebooks/05_svm_and_error_analysis.ipynb` (Experiment notebook & confusion deep-dive)
- `reports/error_analysis.md` (Academic synthesis: why AML fails on similar apparel)
- `reports/results/svm_pixels.json` & `reports/results/svm_hog.json` (Exported metrics)
- `reports/figures/errors/` (Exported confusion matrices and visual error galleries)

> [!NOTE]
> Because you only edit files in the paths above, your PR will have **zero merge conflicts** with your teammates' models!

---

## 2. Step-by-Step Implementation Guide

### Step 1: Branch from Updated Main
Once Milestone 2 is merged by Anant:
```bash
git checkout main
git pull origin main
git checkout -b feat/utkarsh-svm-error-analysis
```

*(Optional early start: If Kartik's PR is pending, you can prototype `svm.py` and `error_analysis.py` using synthetic data like `np.random.rand(100, 784)` to verify your logic.)*

### Step 2: Implement SVM Pipeline (`src/aml/models/svm.py`)
```python
"""Support Vector Machine (Linear SVM) modeling and hyperparameter tuning."""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from aml.evaluation import compute_metrics, measure_inference_latency

def build_svm_pipeline(C=1.0, max_iter=2000):
    """
    Constructs a Pipeline with StandardScaler and LinearSVC.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearSVC(
            C=C,
            max_iter=max_iter,
            random_state=42
        ))
    ])

def tune_svm(X_train, y_train, X_val, y_val, c_grid=[0.1, 1.0, 10.0]):
    """
    Evaluates LinearSVC across a grid of C penalty values.
    Returns:
        best_C: selected C with highest validation macro F1
        results: list of candidate metric dictionaries
    """
    results = []
    best_C = c_grid[0]
    best_macro_f1 = -1.0
    
    for C in c_grid:
        print(f"Fitting Linear SVM with C={C}...")
        pipeline = build_svm_pipeline(C=C)
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

def evaluate_final_svm(X_train_full, y_train_full, X_test, y_test, best_C):
    """
    Fits optimal SVM on all 60,000 training images and evaluates on 10,000 test set.
    """
    pipeline = build_svm_pipeline(C=best_C)
    pipeline.fit(X_train_full, y_train_full)
    
    test_preds = pipeline.predict(X_test)
    metrics = compute_metrics(y_test, test_preds)
    latency_ms = measure_inference_latency(pipeline, X_test[:1000])
    
    # Linear parameters: (num_features * num_classes) + num_classes intercepts
    raw_model = pipeline.named_steps["model"]
    param_count = raw_model.coef_.size + raw_model.intercept_.size
    
    return pipeline, metrics, latency_ms, param_count
```

### Step 3: Implement Reusable Error Analysis Tools (`src/aml/error_analysis.py`)
```python
"""Reusable confusion matrix and visual error inspection utilities."""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", save_path=None):
    """
    Plots a normalized confusion matrix heatmap with percentage annotations.
    """
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("True Category", fontsize=11)
    ax.set_xlabel("Predicted Category", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    return fig

def plot_garment_confusion_submatrix(cm, class_names, subset_indices=[0, 2, 4, 6], title="Upper-Body Garment Confusion", save_path=None):
    """
    Zooms specifically into the most confused apparel classes:
    0: T-shirt/top, 2: Pullover, 4: Coat, 6: Shirt.
    """
    sub_cm = cm[np.ix_(subset_indices, subset_indices)]
    sub_names = [class_names[i] for i in subset_indices]
    sub_norm = sub_cm.astype('float') / sub_cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        sub_norm,
        annot=True,
        fmt=".1%",
        cmap="Reds",
        xticklabels=sub_names,
        yticklabels=sub_names,
        ax=ax
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("True Category", fontsize=11)
    ax.set_xlabel("Predicted Category", fontsize=11)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    return fig

def extract_misclassifications(y_true, y_pred, X_images, true_class, pred_class, max_samples=5):
    """
    Finds sample images where y_true == true_class but model predicted pred_class.
    """
    indices = np.where((y_true == true_class) & (y_pred == pred_class))[0]
    selected_indices = indices[:max_samples]
    return X_images[selected_indices], selected_indices
```

### Step 4: Run Experiments & Write Gap Report (`notebooks/05_svm_and_error_analysis.ipynb` & `reports/error_analysis.md`)
1. **Train & Tune SVM:**
   - Tune $C \in [0.1, 1.0, 10.0]$ on Pixels and HOG.
   - Refit optimal models on 60,000 train set and evaluate on 10,000 test set.
   - Save metrics to `reports/results/svm_pixels.json` and `reports/results/svm_hog.json`.
2. **Comprehensive Clothing Error Analysis:**
   - **The Shirt Dilemma:** Plot the upper-body confusion submatrix (`plot_garment_confusion_submatrix`). Highlight that **Shirt (Class 6)** consistently exhibits the lowest recall (~55-75%) across all models, with over 15-20% of Shirts predicted as T-shirts or Pullovers!
   - **Misclassified Clothing Gallery:** 
     - Use `extract_misclassifications` to extract 5 actual $28 \times 28$ test images of Shirts predicted as T-shirts, and 5 Pullovers predicted as Coats.
     - Plot them in a grid so the team can inspect what caused the failure.
     - Save figures into `reports/figures/errors/`.
3. **Write `reports/error_analysis.md` (The Academic Motivation for Phase 2):**
   - Synthesize "The Gap" between handcrafted visual descriptors and true visual categorization:
     1. *Flattened Pixels:* Completely destroy 2D spatial locality and translation invariance.
     2. *HOG Descriptors:* Fix edge orientation bins beforehand without understanding semantic apparel concepts. While HOG works well for distinct silhouettes (e.g., trousers, bags, ankle boots), it completely misses the fine-grained localized differences (collar shapes, button rows, fabric weave, neckline cuts) that distinguish a dress shirt from a casual crewneck t-shirt or wool sweater.
     3. *Conclusion & Bridge to Phase 2:* These structural limitations prove why an end-to-end **Convolutional Neural Network (CNN)** with hierarchical spatial feature learning is necessary for inventory image tagging.

---

## 3. Verification & Acceptance Checklist

- [ ] `build_svm_pipeline()` converges within `max_iter=2000` without warning.
- [ ] Tuning over $C \in [0.1, 1.0, 10.0]$ completed for both Pixels and HOG.
- [ ] Upper-body garment confusion submatrix generated and saved.
- [ ] Visual misclassification gallery rendered with test image examples.
- [ ] `reports/error_analysis.md` thoroughly documents "The Gap" and Phase 2 justification.
- [ ] `reports/results/svm_pixels.json` and `reports/results/svm_hog.json` exist and match the standardized schema.
- [ ] Notebook `notebooks/05_svm_and_error_analysis.ipynb` executes completely from top to bottom.

---

## 4. Git & PR Submission Instructions

```bash
git add src/aml/models/svm.py src/aml/error_analysis.py notebooks/05_svm_and_error_analysis.ipynb reports/error_analysis.md reports/results/svm_*.json reports/figures/errors/
git commit -m "feat(svm): implement Linear SVM tuning, garment confusion analysis, and Phase 2 gap report"
git push -u origin feat/utkarsh-svm-error-analysis
```
Open a Pull Request pointing to `main` and request **Anant Jain** (`anantjain-afk`) to review and merge.
