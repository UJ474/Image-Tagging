# Individual Work Guide: Anant Jain

**Project 11:** Automatic Clothing Category Recognition  
**Phase:** Phase 1 — Applied Machine Learning (AML) Baseline  
**Role:** Repository Scaffolding, Data Pipeline & Master Notebook Integration  
**GitHub Account:** [`anantjain-afk`](https://github.com/anantjain-afk)  
**Order of Work:** **Milestone 1 (First to implement and merge)** & **Milestone 4 (Final Synthesis)**  
**Branch:** `feat/anant-data-setup` (and later `feat/anant-master-notebook`)  
**Merge Authority:** **You are the repository owner and the sole person who merges all Pull Requests into `main`.**  

---

## 1. Scope & Owned Files

You are responsible for establishing the repository foundation so your teammates can work smoothly, writing the official data loading and validation splitting logic, defining shared evaluation utilities, and compiling the final master comparison notebook:

- `requirements.txt` & `pyproject.toml`
- `.gitignore`
- `src/aml/__init__.py`
- `src/aml/data.py` (Official dataset downloader, normalizer, and stratified splits)
- `src/aml/evaluation.py` (Metric computation, parameter counter, and latency benchmark)
- `notebooks/01_exploratory_data_analysis.ipynb` (Comprehensive dataset exploration, class balance, archetypes, and PCA)
- `notebooks/00_master_aml_comparison.ipynb` (Final deliverable notebook integrating all 8 models)
- `reports/figures/eda/` (Exported dataset analysis figures)
- `README.md` (Project root documentation)

---

## 2. Step-by-Step Implementation Guide

### Stage 1: Milestone 1 (Do This First)

#### Step 1: Set Up Project Dependencies & Packaging
Create the environment configuration files in the repository root:
1. `requirements.txt`:
   ```text
   numpy>=1.24.0
   pandas>=2.0.0
   scipy>=1.10.0
   scikit-learn>=1.3.0
   scikit-image>=0.21.0
   matplotlib>=3.7.0
   seaborn>=0.12.0
   jupyterlab>=4.0.0
   joblib>=1.3.0
   ```
2. `pyproject.toml`:
   ```toml
   [build-system]
   requires = ["setuptools>=61.0"]
   build-backend = "setuptools.build_meta"

   [project]
   name = "aml"
   version = "0.1.0"
   description = "Applied Machine Learning Baseline for Clothing Category Recognition"
   readme = "README.md"
   requires-python = ">=3.10"
   dependencies = [
       "numpy",
       "scipy",
       "scikit-learn",
       "scikit-image",
       "matplotlib",
       "seaborn",
       "pandas",
       "joblib"
   ]

   [tool.setuptools.packages.find]
   where = ["src"]
   ```
3. `.gitignore`: Ensure temporary files, cache, and large binaries are ignored:
   ```text
   .venv/
   __pycache__/
   *.pyc
   .ipynb_checkpoints/
   data/raw/
   data/cache/
   artifacts/
   dist/
   build/
   *.egg-info/
   ```

#### Step 2: Implement the Data Pipeline (`src/aml/data.py`)
Implement the official dataset loading, $[0.0, 1.0]$ normalization, and stratified train/val split:
```python
"""Data loading and preprocessing module for Fashion-MNIST."""
import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import fetch_openml

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

def load_fashion_mnist(data_dir="data/raw"):
    """
    Downloads or loads Fashion-MNIST dataset.
    Returns:
        X_train_raw: (60000, 28, 28) float32 in [0.0, 1.0]
        y_train_raw: (60000,) int64 in [0..9]
        X_test: (10000, 28, 28) float32 in [0.0, 1.0]
        y_test: (10000,) int64 in [0..9]
    """
    os.makedirs(data_dir, exist_ok=True)
    cache_file = os.path.join(data_dir, "fashion_mnist_raw.npz")
    
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return data["X_train_raw"], data["y_train_raw"], data["X_test"], data["y_test"]
    
    # Fetch from OpenML
    fmnist = fetch_openml("Fashion-MNIST", as_frame=False, parser="auto")
    X = fmnist.data.astype(np.float32) / 255.0  # Normalize to [0, 1]
    y = fmnist.target.astype(np.int64)
    
    # Reshape to (N, 28, 28)
    X = X.reshape(-1, 28, 28)
    
    # Official split: first 60,000 train, last 10,000 test
    X_train_raw, y_train_raw = X[:60000], y[:60000]
    X_test, y_test = X[60000:], y[60000:]
    
    np.savez_compressed(
        cache_file,
        X_train_raw=X_train_raw, y_train_raw=y_train_raw,
        X_test=X_test, y_test=y_test
    )
    return X_train_raw, y_train_raw, X_test, y_test

def get_train_val_test_splits(data_dir="data/raw", val_size=12000, seed=42):
    """
    Returns stratified 48,000 train, 12,000 validation, and 10,000 test splits.
    """
    X_train_raw, y_train_raw, X_test, y_test = load_fashion_mnist(data_dir=data_dir)
    
    # Stratified split: 48,000 train, 12,000 validation
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(sss.split(X_train_raw, y_train_raw))
    
    return {
        "X_train": X_train_raw[train_idx],
        "y_train": y_train_raw[train_idx],
        "X_val": X_train_raw[val_idx],
        "y_val": y_train_raw[val_idx],
        "X_train_full": X_train_raw,
        "y_train_full": y_train_raw,
        "X_test": X_test,
        "y_test": y_test,
        "class_names": CLASS_NAMES
    }
```

#### Step 3: Implement Shared Evaluation Utilities (`src/aml/evaluation.py`)
Ensure all teammates report identical metric definitions:
```python
"""Evaluation and benchmarking utilities for AML classifiers."""
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

def compute_metrics(y_true, y_pred):
    """
    Computes accuracy, macro F1, per-class recall, and confusion matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, labels=list(range(10)), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class_recall": [float(r) for r in recalls],
        "confusion_matrix": cm.tolist()
    }

def measure_inference_latency(model, X_sample, n_iterations=5):
    """
    Measures median inference latency per image in milliseconds.
    """
    # Warmup
    _ = model.predict(X_sample[:min(10, len(X_sample))])
    
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = model.predict(X_sample)
        elapsed = time.perf_counter() - start
        times.append(elapsed / len(X_sample))
    
    median_time_per_image_ms = np.median(times) * 1000.0
    return float(median_time_per_image_ms)
```

#### Step 4: Verify and Merge Milestone 1 to `main`
Run local checks:
```bash
python3 -c "from aml.data import get_train_val_test_splits; d = get_train_val_test_splits(); print('X_train:', d['X_train'].shape, 'X_val:', d['X_val'].shape, 'X_test:', d['X_test'].shape)"
```
Expected output:
`X_train: (48000, 28, 28) X_val: (12000, 28, 28) X_test: (10000, 28, 28)`

Commit and merge directly into `main`:
```bash
git checkout -b feat/anant-data-setup
git add pyproject.toml requirements.txt .gitignore src/aml/
git commit -m "feat(setup): add project foundation, data pipeline, and evaluation utilities"
git checkout main
git merge feat/anant-data-setup
git push origin main
```
Notify **Kartik Mehra** that Milestone 1 is merged and he can begin Milestone 2.

---

### Stage 2: Milestone 4 (Master Comparison Deliverable)

After Kartik, Krishna, Vipul, Aditya, and Utkarsh have merged their PRs into `main`:

#### Step 5: Build `notebooks/00_master_aml_comparison.ipynb`
1. Pull all teammate contributions:
   ```bash
   git checkout main
   git pull origin main
   ```
2. In the notebook:
   - Read all 8 JSON files from `reports/results/`:
     - `logreg_pixels.json`, `logreg_hog.json`
     - `knn_pixels.json`, `knn_hog.json`
     - `rf_pixels.json`, `rf_hog.json`
     - `svm_pixels.json`, `svm_hog.json`
   - Build the **Master Summary Table**:
     | Model | Feature Set | Val Accuracy | Test Accuracy | Test Macro F1 | Complexity | Latency (ms/img) |
     |---|---|---|---|---|---|---|
     | Logistic Regression | Pixels | ... | ... | ... | 7,850 weights | ... |
     | Logistic Regression | HOG | ... | ... | ... | 3,250 weights | ... |
     | $k$-NN | Pixels | ... | ... | ... | 188 MB memory | ... |
     | $k$-NN | HOG | ... | ... | ... | 78 MB memory | ... |
     | Random Forest | Pixels | ... | ... | ... | Total nodes | ... |
     | Random Forest | HOG | ... | ... | ... | Total nodes | ... |
     | Linear SVM | Pixels | ... | ... | ... | 7,850 weights | ... |
     | Linear SVM | HOG | ... | ... | ... | 3,250 weights | ... |
   - Generate side-by-side grouped bar charts for **Test Accuracy** and **Test Macro F1** across both feature representations.
   - Summarize overall findings: highlight why HOG improves linear models, but why both representations hit an accuracy ceiling (~88-89%) and struggle on Shirts, setting up the Phase 2 CNN.
3. Save and commit:
   ```bash
   git checkout -b feat/anant-master-notebook
   git add notebooks/00_master_aml_comparison.ipynb
   git commit -m "docs(deliverable): assemble master AML comparison notebook and benchmark table"
   git checkout main
   git merge feat/anant-master-notebook
   git push origin main
   ```

---

## 3. Merge Authority Checklist for Anant

As PRs are submitted by teammates, verify:
- [ ] PR for Milestone 2 from Kartik: `src/aml/features.py`, caching script, and `01_feature_exploration.ipynb`. Merge to `main`.
- [ ] PRs for Milestone 3 from Krishna, Vipul, Aditya, and Utkarsh:
  - Check that each member only modified their assigned model, notebook, and JSON files.
  - Check that test metrics are populated in `reports/results/`.
  - Merge each PR into `main` using a clean merge commit.
