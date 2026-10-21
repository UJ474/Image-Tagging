# Individual Work Guide: Kartik Mehra

**Project 11:** Automatic Clothing Category Recognition  
**Phase:** Phase 1 — Applied Machine Learning (AML) Baseline  
**Role:** Feature Engineering (Flattened Pixels & HOG), Caching Utility & Feature Exploration  
**GitHub Account:** [`KartikMehra22`](https://github.com/KartikMehra22)  
**Order of Work:** **Milestone 2 (Begins immediately after Anant merges Milestone 1)**  
**Branch:** `feat/kartik-features`  
**PR Review & Merge:** **Open PR against `main` for Anant Jain ([`anantjain-afk`](https://github.com/anantjain-afk)) to review and merge.**  

---

## 1. Scope & Owned Files

You own the feature extraction pipeline, the local caching utility that enables instantaneous feature loading for your teammates, and the feature exploration notebook:

- `src/aml/features.py` (Flattened pixel and HOG extractors + disk caching)
- `notebooks/01_feature_exploration.ipynb` (HOG gradient visualizations and feature comparisons)
- `reports/figures/features/` (Exported HOG visualization figures)
- `data/cache/` (Local feature cache directory)

---

## 2. Step-by-Step Implementation Guide

### Step 1: Pull Latest Main
Before beginning, ensure Anant has merged Milestone 1, then pull `main`:
```bash
git checkout main
git pull origin main
git checkout -b feat/kartik-features
```

### Step 2: Implement Feature Pipeline (`src/aml/features.py`)
Implement the functions to convert 2D images into 1D feature vectors:
```python
"""Feature extraction module for Fashion-MNIST (Pixels and HOG)."""
import os
import numpy as np
from skimage.feature import hog

def extract_pixels(images):
    """
    Flattens (N, 28, 28) images into (N, 784) float32 vectors.
    """
    if images.ndim == 2:
        images = images[np.newaxis, ...]
    assert images.shape[1:] == (28, 28), f"Expected shape (N, 28, 28), got {images.shape}"
    return images.reshape(images.shape[0], -1).astype(np.float32)

def extract_single_hog(image, orientations=9, pixels_per_cell=(7, 7), cells_per_block=(2, 2)):
    """
    Computes HOG for a single (28, 28) image.
    Returns:
        (324,) float32 feature vector
    """
    return hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        transform_sqrt=False,
        feature_vector=True
    ).astype(np.float32)

def extract_hog_batch(images, orientations=9, pixels_per_cell=(7, 7), cells_per_block=(2, 2)):
    """
    Computes HOG for a batch of (N, 28, 28) images.
    Returns:
        (N, 324) float32 feature matrix
    """
    if images.ndim == 2:
        images = images[np.newaxis, ...]
    
    n_samples = images.shape[0]
    hog_features = [
        extract_single_hog(
            images[i],
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block
        )
        for i in range(n_samples)
    ]
    return np.array(hog_features, dtype=np.float32)

def get_features(images, feature_type="pixels", cache_path=None):
    """
    Extracts or loads cached features.
    Args:
        images: (N, 28, 28) float32 array
        feature_type: "pixels" or "hog"
        cache_path: optional path to .npz cache file
    Returns:
        Feature matrix: (N, 784) for pixels or (N, 324) for hog
    """
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["features"]
    
    if feature_type == "pixels":
        features = extract_pixels(images)
    elif feature_type == "hog":
        features = extract_hog_batch(images)
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}. Use 'pixels' or 'hog'.")
    
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, features=features)
        
    return features

def precompute_and_cache_all(data_dir="data/raw", cache_dir="data/cache"):
    """
    Convenience CLI function to precompute and cache features for train, val, and test splits.
    """
    from aml.data import get_train_val_test_splits
    print("Loading data splits...")
    splits = get_train_val_test_splits(data_dir=data_dir)
    os.makedirs(cache_dir, exist_ok=True)
    
    for split_name in ["train", "val", "test", "train_full"]:
        imgs = splits[f"X_{split_name}"]
        print(f"Extracting pixels for {split_name} ({len(imgs)} images)...")
        get_features(imgs, feature_type="pixels", cache_path=os.path.join(cache_dir, f"{split_name}_pixels.npz"))
        
        print(f"Extracting HOG for {split_name} ({len(imgs)} images)...")
        get_features(imgs, feature_type="hog", cache_path=os.path.join(cache_dir, f"{split_name}_hog.npz"))
        
    print("All features successfully cached in", cache_dir)

if __name__ == "__main__":
    precompute_and_cache_all()
```

### Step 3: Build Feature Exploration Notebook (`notebooks/01_feature_exploration.ipynb`)
1. **Visualizing HOG Gradient Maps:**
   - For a representative sample of each of the 10 clothing categories, run `hog(image, ..., visualize=True)` to extract both the feature descriptor and the gradient visualization image.
   - Display a $10 \times 2$ grid: Original $28 \times 28$ grayscale image on the left, HOG orientation map on the right.
   - Explain what HOG captures: dominant edges (collar line of shirts, legs of trousers, outlines of bags and ankle boots).
2. **The Geometric Information Tradeoff:**
   - Raw Pixels ($784$ dimensions): Preserves fine intensity values but discards spatial neighborhood structure when flattened into a 1D vector.
   - HOG ($324$ dimensions): Enforces local edge gradient orientation and normalizes block illumination, but completely discards interior texture and button/pocket details.
3. **Save Figures:** Save generated visual comparison plots into `reports/figures/features/`.

---

## 3. Verification & Acceptance Checklist

- [ ] `extract_pixels` produces `(N, 784)` array with `float32` dtype.
- [ ] `extract_hog_batch` produces `(N, 324)` array with `float32` dtype.
- [ ] Running `python -m aml.features` successfully runs and populates `.npz` files in `data/cache/`.
- [ ] Notebook `notebooks/01_feature_exploration.ipynb` runs completely and displays the side-by-side HOG visualizations.

---

## 4. Git & PR Submission Instructions

```bash
# Verify changes
git status

# Stage and commit your files
git add src/aml/features.py notebooks/01_feature_exploration.ipynb reports/figures/features/
git commit -m "feat(features): implement pixel and HOG extraction with disk caching and visual exploration"
git push -u origin feat/kartik-features
```
- Open a Pull Request on GitHub: base `main` $\leftarrow$ compare `feat/kartik-features`.
- Request review from **Anant Jain** (`anantjain-afk`).
- Anant merges the PR, enabling all 4 model owners to begin Milestone 3.
