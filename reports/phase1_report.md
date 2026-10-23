# Phase 1 Report: Applied Machine Learning Baseline
## Automatic Clothing Category Recognition on Fashion-MNIST

**Course Project:** Project 11 | Team T112 | Section D  
**Phase:** Phase 1 — Applied Machine Learning (AML) Baseline  
**Dataset:** Fashion-MNIST (Zalando Research) — 70,000 grayscale 28×28 images  
**Date:** October 2026

---

## Team

| # | Member | GitHub | Contribution |
|---|--------|--------|-------------|
| 1 | **Anant Jain** | `anantjain-afk` | Repo setup, data pipeline, master notebook |
| 2 | **Kartik Mehra** | `KartikMehra22` | Feature extraction (Pixels & HOG), caching |
| 3 | **Krishna Gehlot** | `NSTKrishna` | Logistic Regression baseline |
| 4 | **Vipul Sharma** | `vs-code06` | k-NN baseline & latency profiling |
| 5 | **Aditya** | `AdityaInnovates` | Random Forest baseline |
| 6 | **Utkarsh Jain** | `UJ474` | Linear SVM & garment error analysis |

---

## 1. Problem Statement

Commercial inventory systems must reliably tag thousands of clothing images per day. Fashion-MNIST provides a standardized 10-class benchmark for this task. Phase 1 evaluates four classical ML algorithms on two visual representations (raw pixels and HOG descriptors) to establish a rigorous baseline. The primary research question is:

> **Can fixed, handcrafted visual descriptors adequately classify clothing, or does fine-grained category confusion necessitate hierarchical features learned by a CNN?**

---

## 2. Experimental Setup

### 2.1 Dataset & Splits

| Split | Size | Purpose |
|-------|------|---------|
| Training (dev) | 48,000 | Model selection & hyperparameter tuning |
| Validation | 12,000 | Held-out evaluation during tuning |
| Test (official) | 10,000 | Final one-time evaluation |

All splits use stratified sampling with `random_state=42`. Pixel values normalized from `uint8 [0, 255]` → `float32 [0.0, 1.0]`.

### 2.2 Class Labels

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | T-shirt/top | 5 | Sandal |
| 1 | Trouser | 6 | Shirt |
| 2 | Pullover | 7 | Sneaker |
| 3 | Dress | 8 | Bag |
| 4 | Coat | 9 | Ankle boot |

### 2.3 Feature Representations

| Feature | Dimensions | Description |
|---------|-----------|-------------|
| **Pixels** | 784-d | Flattened 28×28 normalized pixel values |
| **HOG** | 324-d | 9 orientations, 7x7 px/cell, 2x2 cells/block, L2-Hys normalization |

### 2.4 Models & Best Hyperparameters

| Model | Best (Pixels) | Best (HOG) |
|-------|-------------|-----------|
| **Logistic Regression** | C=0.1, lbfgs | C=0.1, lbfgs |
| **k-NN** | k=7, distance | k=5, distance |
| **Random Forest** | 200 trees, depth=20 | 200 trees, depth=None |
| **Linear SVM** | C=0.1, 2000 iter | C=1.0, 2000 iter |

---

## 3. Results

### 3.1 Master Comparison Table

| Rank | Model | Features | Test Acc | Macro F1 | Parameters | Latency (ms/img) |
|------|-------|----------|----------|----------|-----------|-----------------|
| 1 | **Random Forest** | **Pixels** | **87.43%** | **87.31%** | 1,693,570 nodes | 0.0264 |
| 2 | Random Forest | HOG | 84.35% | 84.30% | 2,458,016 nodes | 0.0291 |
| 3 | k-NN | Pixels | 85.41% | 85.40% | 179.4 MiB | 0.2741 |
| 4 | Logistic Regression | HOG | 84.55% | 84.49% | 3,250 | 0.0004 |
| 5 | k-NN | HOG | 83.84% | 83.85% | 74.2 MiB | 0.0980 |
| 6 | Logistic Regression | Pixels | 84.32% | 84.24% | 7,850 | 0.0007 |
| 7 | Linear SVM | HOG | 84.23% | 84.06% | 3,250 | 0.0006 |
| 8 | Linear SVM | Pixels | 83.80% | 83.57% | 7,850 | 0.0013 |

> **Best overall:** Random Forest on pixels (87.43% accuracy, 87.31% macro F1).

### 3.2 Key Findings

**Finding 1 — Random Forest Dominates Overall**  
Random Forest on pixels achieves the highest accuracy (+2.02pp over k-NN, +3.11pp over Logistic Regression, +3.63pp over Linear SVM). Non-linear decision boundaries from 200 trees decisively outperform linear classifiers.

**Finding 2 — HOG Helps Linear Models, Hurts Non-Linear Ones**  
For linear models (LogReg, SVM), HOG features improve accuracy by +0.23% (LogReg) and +0.43% (SVM) while reducing parameter count by 58.6% (7,850 -> 3,250). For Random Forest and k-NN, HOG reduces accuracy by 3.08pp and 1.57pp — the fixed HOG vocabulary discards pixel-level detail that tree ensembles exploit.

**Finding 3 — k-NN Memory Trade-off**  
Switching from pixels (179.4 MiB) to HOG (74.2 MiB) reduces k-NN memory by 58.7% and inference latency by 2.4x (0.274 -> 0.098 ms/img), at the cost of only 1.57pp accuracy.

**Finding 4 — Linear Classifiers Are Fastest**  
Logistic Regression on HOG is the fastest model at 0.0004 ms/image — 685x faster than k-NN on pixels. Highly suitable for real-time inventory tagging.

**Finding 5 — Shirt is Universally the Hardest Class**  
Mean Shirt recall across all 8 configurations: **57.7%**. No model exceeds 62.6% Shirt recall.

---

## 4. Per-Class Recall Analysis

### Best Model — Random Forest (Pixels)

| Class | Label | Recall |
|-------|-------|--------|
| 0 | T-shirt/top | 85.9% |
| 1 | Trouser | 96.1% |
| 2 | Pullover | 78.9% |
| 3 | Dress | 90.5% |
| 4 | Coat | 81.1% |
| 5 | Sandal | 95.6% |
| 6 | **Shirt** | **59.3% ⚠** |
| 7 | Sneaker | 95.2% |
| 8 | Bag | 97.3% |
| 9 | Ankle boot | 94.4% |

### Cross-Model Shirt Recall

| Model | Pixels | HOG |
|-------|--------|-----|
| Logistic Regression | 57.1% | 57.8% |
| k-NN | 60.2% | 62.6% |
| Random Forest | 59.3% | 59.4% |
| Linear SVM | 50.7% | 54.6% |
| **Mean** | **56.8%** | **58.6%** |

---

## 5. Error Analysis: The Shirt Dilemma

Shirt (class 6) is the most visually ambiguous upper-body label. Its 28x28 silhouette overlaps with T-shirts (similar collar/sleeve), Pullovers (similar torso coverage), and Coats (similar outer silhouette). The confusion is directional:

- Shirts misclassified as T-shirts/top: ~15-16% of Shirt test images
- Shirts misclassified as Pullovers: ~12-14% of Shirt test images

### Why Fixed Representations Fail

**Flattened Pixels (784-d):** Linear classifiers assign one weight per pixel position, with no notion that a collar shifted upward is the same discriminative cue. Small translations corrupt many coordinates while the garment category is unchanged.

**HOG Descriptors (324-d):** HOG captures global edge orientations, helping with distinct silhouettes (trousers, bags, sandals). However, it cannot learn that a button-row edge pattern near the neckline is more diagnostic than a similar edge elsewhere. Cell pooling removes exactly the fine spatial cues needed for upper-body distinction.

**Linear Decision Boundaries (LogReg, SVM):** A hyperplane cannot create discriminative evidence that is absent in the representation. If Shirt and T-shirt share overlapping distributions, adjusting the margin penalty C cannot resolve the overlap.

### Evidence Artifacts

The SVM error analysis confirms these findings visually:

1. Full 10-class normalized confusion matrix (`svm_pixels_confusion_matrix.png`)
2. Focused 4x4 sub-matrix: T-shirt/Pullover/Coat/Shirt (`svm_pixels_upper_body_confusion.png`)
3. Gallery: Pullover images misclassified as Coat (`svm_pixels_2_as_4_gallery.png`)
4. Gallery: Shirt images misclassified as T-shirt/top (`svm_pixels_6_as_0_gallery.png`)

---

## 6. Efficiency Analysis

For real-time inventory tagging (< 1 ms/image budget), the viable options are:

| Model | Features | Latency | Accuracy |
|-------|----------|---------|---------|
| Logistic Regression | HOG | 0.0004 ms | 84.55% |
| Logistic Regression | Pixels | 0.0007 ms | 84.32% |
| Linear SVM | HOG | 0.0006 ms | 84.23% |
| Random Forest | Pixels | 0.026 ms | 87.43% |

k-NN (both features) and Random Forest are the only options requiring > 10 ms/image, making them unsuitable for real-time deployment at scale.

**Accuracy-per-parameter winner:** Logistic Regression on HOG — 84.55% accuracy with only 3,250 parameters.

---

## 7. Bridge to Phase 2

The consistent Shirt confusion (mean recall 57.7%) across all 8 Phase 1 models motivates Phase 2. This is not a hyperparameter problem — it is a representation problem. A CNN is appropriate because:

1. **Convolutions share detectors** across locations — a collar detector fires wherever the collar appears.
2. **Stacked layers compose features** — edges → textures → collar shape → shirt category.
3. **Pooling provides translation tolerance** — small position shifts do not corrupt the feature vector.
4. **End-to-end learning** — the network learns which local patterns are diagnostic.

### Phase 2 Success Criteria

- Shirt recall above **60%** (Phase 1 ceiling across all 8 models)
- Reduction in Shirt → T-shirt/Pullover confusion entries in the 4x4 upper-body sub-matrix
- Same held-out test split; same metric suite for direct comparability

---

## 8. Conclusion

Phase 1 establishes a rigorous 8-configuration AML baseline on Fashion-MNIST:

| Dimension | Winner | Value |
|-----------|--------|-------|
| Best accuracy | RF + Pixels | 87.43% |
| Best macro F1 | RF + Pixels | 87.31% |
| Fastest inference | LogReg + HOG | 0.0004 ms/img |
| Most compact model | LogReg + HOG / SVM + HOG | 3,250 params |
| Hardest class | Shirt (class 6) | 57.7% mean recall |

All four models plateau at the same fundamental limit: fine-grained upper-body garment categories (Shirt, T-shirt, Pullover, Coat) require spatial feature composition that fixed descriptors and linear boundaries cannot provide. Phase 2 will directly test whether a CNN closes this gap.

---

*All metrics evaluated on the official Fashion-MNIST held-out test set (10,000 images). Source data in `reports/results/*.json`. Full visualizations in `notebooks/00_master_aml_comparison.ipynb`.*
