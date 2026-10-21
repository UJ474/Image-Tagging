# Image-Tagging: Automatic Clothing Category Recognition

**Title:** Inventory Image Tagging via Handcrafted Features and a Small CNN  
**Course Project:** Project 11 | Team T112 | Section D (6 Members)  
**Dataset:** [Fashion-MNIST (Official Zalando Dataset)](https://github.com/zalandoresearch/fashion-mnist)  
**Current Milestone:** **Phase 1 (Applied Machine Learning Baseline)**

---

## 1. Project Overview & The "Hybrid" Logic

Can fixed, handcrafted visual descriptors adequately classify clothing for an inventory catalogue, or does fine-grained category confusion necessitate hierarchical visual representations learned by a Convolutional Neural Network (CNN)?

To rigorously answer this, our project is structured into two phases:
1. **Phase 1 (Current): Applied Machine Learning (AML) Baseline**
   - Normalize $28 \times 28$ grayscale apparel images into $[0.0, 1.0]$.
   - Compare two input representations:
     - **Flattened Raw Pixels** ($784$ dimensions)
     - **Histogram of Oriented Gradients (HOG)** ($324$ dimensions, $9$ orientations, $7\times 7$ cells, $2\times 2$ blocks)
   - Evaluate across four distinct classification algorithms:
     1. **Logistic Regression** (Linear baseline)
     2. **k-Nearest Neighbors (k-NN)** (Instance-based baseline)
     3. **Random Forest** (Non-linear tree ensemble)
     4. **Support Vector Machine (SVM)** (Maximum margin baseline)
   - Benchmark all 8 combinations on accuracy, macro F1, per-class recall, parameter count, and inference latency.
   - **Identify The Gap:** Conduct in-depth error analysis on visually ambiguous garments (**T-shirt/top, Shirt, Pullover, Coat**), exposing why fixed descriptors struggle with fine-grained neckline/sleeve/button geometry.
2. **Phase 2 (Future Work): Deep Learning (CNN) Model**
   - Train a 2-to-3 block CNN with convolutional layers, pooling, dropout, and early stopping.
   - Compare CNN confusion matrices against Phase 1 AML baselines to verify whether learned hierarchical features resolve the clothing categorization gap.

---

## 2. Team Structure & Phase 1 Assignments

The workload is evenly distributed across all 6 members with zero sequential blockers:

| Member | GitHub | Phase 1 Assignment | Deliverables |
|---|---|---|---|
| **Anant Jain** | [`anantjain-afk`](https://github.com/anantjain-afk) | **Repository Setup, Data Pipeline & Master Notebook** | `src/aml/data.py`, `src/aml/evaluation.py`, `notebooks/00_master_aml_comparison.ipynb` |
| **Kartik Mehra** | [`KartikMehra22`](https://github.com/KartikMehra22) | **Feature Engineering (Pixels & HOG), Caching & Feature Analysis** | `src/aml/features.py`, `notebooks/01_feature_exploration.ipynb`, `data/cache/` |
| **Krishna Gehlot** | [`NSTKrishna`](https://github.com/NSTKrishna) | **Model 1: Logistic Regression & Weight Analysis** | `src/aml/models/logreg.py`, `notebooks/02_logistic_regression.ipynb`, `reports/results/logreg_*.json` |
| **Vipul Sharma** | [`vs-code06`](https://github.com/vs-code06) | **Model 2: k-Nearest Neighbors (k-NN) & Latency Profiling** | `src/aml/models/knn.py`, `notebooks/03_knn_experiments.ipynb`, `reports/results/knn_*.json` |
| **Aditya** | [`AdityaInnovates`](https://github.com/AdityaInnovates) | **Model 3: Random Forest & Overfitting / Feature Importance** | `src/aml/models/random_forest.py`, `notebooks/04_random_forest.ipynb`, `reports/results/rf_*.json` |
| **Utkarsh Jain** | [`UJ474`](https://github.com/UJ474) | **Model 4: Support Vector Machine (SVM) & Garment Error Analysis** | `src/aml/models/svm.py`, `src/aml/error_analysis.py`, `notebooks/05_svm_and_error_analysis.ipynb`, `reports/error_analysis.md` |

For individual guides with detailed steps, refer to [**`docs/team/README.md`**](docs/team/README.md).

---

## 3. Directory Layout

```
Image-Tagging/
├── README.md                                  # Project landing page
├── prompt.md                                  # Prompt template for AI-assisted dev
├── requirements.txt                           # Core dependencies
├── pyproject.toml                             # Package configuration
├── docs/team/                                 # Individual teammate guides (Markdown)
│   ├── README.md
│   ├── 01_Anant_Setup_Data_MasterNotebook.md
│   ├── 02_Kartik_Feature_Extraction_HOG.md
│   ├── 03_Krishna_Logistic_Regression.md
│   ├── 04_Vipul_KNN_Classifier.md
│   ├── 05_Aditya_Random_Forest.md
│   └── 06_Utkarsh_SVM_Error_Analysis.md
├── src/aml/                                   # Source package
│   ├── __init__.py
│   ├── data.py                                # Fashion-MNIST loader & 48k/12k split
│   ├── features.py                            # Pixels (784) & HOG (324) extraction
│   ├── evaluation.py                          # Metric calculation & latency timer
│   ├── error_analysis.py                      # Confusion matrices & error galleries
│   └── models/
│       ├── __init__.py
│       ├── logreg.py                          # Logistic Regression
│       ├── knn.py                             # k-Nearest Neighbors
│       ├── random_forest.py                   # Random Forest
│       └── svm.py                             # Support Vector Machine
├── notebooks/                                 # Individual & Master Jupyter Notebooks
└── reports/
    ├── error_analysis.md                      # The Gap: Detailed error analysis
    ├── figures/                               # Exported charts & visualizations
    └── results/                               # 8 standardized JSON result files
```

---

## 4. Getting Started

```bash
# Clone the repository
git clone https://github.com/anantjain-afk/Image-Tagging.git
cd Image-Tagging

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows PowerShell: .\.venv\Scripts\Activate.ps1

# Install dependencies in editable mode
pip install -r requirements.txt
pip install -e . --no-deps
```
