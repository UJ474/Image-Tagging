# AI Assistant Prompt Guide for Team Members

This document contains standardized prompt templates for teammates working with AI coding assistants (such as **Antigravity**, **Claude Code**, **Cursor**, **ChatGPT**, or **GitHub Copilot**) to implement their assigned Phase 1 modules.

---

## 1. Universal Prompt Template

If you want to customize your prompt, copy and paste this into your AI coding tool:

```text
You are an expert AI software engineer assisting me with our college/course project:
"Project 11: Inventory Image Tagging via Handcrafted Features and a Small CNN"

We are currently implementing PHASE 1: Applied Machine Learning (AML) Baseline on Fashion-MNIST.
Repository: anantjain-afk/Image-Tagging
Repo Owner & Sole PR Merger: Anant Jain (anantjain-afk)

Please do the following:
1. Read docs/team/README.md to understand the project architecture, shared technical standards, and sequential 1-to-6 execution order.
2. Read my individual markdown guide: <PATH_TO_MY_GUIDE>
3. Work strictly on the branch specified in my guide.
4. Only create or modify the files in my "Owned Files" list. Do not edit other members' files.
5. Adhere to the shared data formats:
   - Raw images: float32 in [0.0, 1.0], shape (N, 28, 28)
   - Flattened Pixels: float32, shape (N, 784)
   - HOG Descriptors: float32, shape (N, 324) (9 orientations, 7x7 cells, 2x2 blocks, L2-Hys)
   - Labels: int64 in range [0..9]
6. If an upstream teammate's module is not yet merged, use mock/synthetic data (e.g., np.random.rand) to develop and verify all functions independently without being blocked.
7. Implement clean, robust, well-documented code with complete docstrings.
8. Run the verification checklist in my guide, execute tests, and confirm that all outputs match the expected shapes and formats.
9. Export any required metric files to reports/results/ matching the agreed JSON schema.
10. Summarize changed files, report test outcomes, and provide git commit and PR commands ready for Anant to review.
```

---

## 2. Ready-to-Copy Prompts for Each Team Member

Simply copy the specific block below that corresponds to your assignment:

### 1. Anant Jain (Milestone 1 & Final Integration)
```text
I am Anant Jain (GitHub: anantjain-afk). I own Milestone 1 (Repo Foundation, Data Pipeline & EDA) and Milestone 4 (Final Master Notebook Integration) for Project 11: Image-Tagging.

Please read docs/team/README.md and my individual guide: docs/team/01_Anant_Setup_Data_MasterNotebook.md.

Tasks:
1. Set up the project foundation: requirements.txt, pyproject.toml, .gitignore, and src/aml/__init__.py.
2. Implement src/aml/data.py with official Fashion-MNIST downloading, [0, 1] normalization, and stratified 48k/12k/10k splits.
3. Implement src/aml/evaluation.py with accuracy, macro F1, per-class recall, confusion matrix, and latency timer.
4. Build notebooks/01_exploratory_data_analysis.ipynb conducting full EDA: class balance, pixel intensity sparsity, mean silhouettes, variance heatmaps, and 2D PCA cluster projection.
5. Verify all functions and provide git commands to commit and merge Milestone 1 into main.
6. When all 8 model JSON results exist in reports/results/, build notebooks/00_master_aml_comparison.ipynb to generate the master comparison table and plots.
```

---

### 2. Kartik Mehra (Milestone 2: Features & Caching)
```text
I am Kartik Mehra (GitHub: KartikMehra22). I own Milestone 2 (Feature Extraction & HOG Caching) for Project 11: Image-Tagging.

Please read docs/team/README.md and my individual guide: docs/team/02_Kartik_Feature_Extraction_HOG.md.

Tasks:
1. Work on branch feat/kartik-features.
2. Implement src/aml/features.py:
   - extract_pixels (flattens 28x28 to 784-d vector)
   - extract_hog_batch (skimage.feature.hog with 9 orientations, 7x7 cells, 2x2 blocks, L2-Hys -> 324-d vector)
   - get_features with disk caching in data/cache/*.npz
3. Provide CLI runner in features.py to precompute and cache features for train, val, and test splits.
4. Build notebooks/02_feature_exploration.ipynb with side-by-side original vs. HOG gradient visualizations, upper-body ambiguity analysis, and orientation profiles across categories.
5. Verify output dimensions, run checks, and prepare PR for Anant Jain (anantjain-afk) to merge.
```

---

### 3. Krishna Gehlot (Milestone 3: Logistic Regression)
```text
I am Krishna Gehlot (GitHub: NSTKrishna). I own Model 1 (Logistic Regression Classifier) for Project 11: Image-Tagging.

Please read docs/team/README.md and my individual guide: docs/team/03_Krishna_Logistic_Regression.md.

Tasks:
1. Work on branch feat/krishna-logreg.
2. Implement src/aml/models/logreg.py using StandardScaler + LogisticRegression(solver="lbfgs", max_iter=1000).
3. Tune regularization C in [0.1, 1.0, 10.0] on validation set for both Raw Pixels and HOG Descriptors.
4. Fit best pipeline on full 60k training set and evaluate on 10k test set.
5. In notebooks/03_logistic_regression.ipynb:
   - Present validation tuning table and test metrics.
   - Reshape the 10x784 weight matrix into 28x28 heatmaps per class and analyze linear template features.
6. Export standardized results to reports/results/logreg_pixels.json and reports/results/logreg_hog.json.
7. Verify all metrics and prepare PR for Anant Jain (anantjain-afk) to merge.
```

---

### 4. Vipul Sharma (Milestone 3: $k$-NN Classifier)
```text
I am Vipul Sharma (GitHub: vs-code06). I own Model 2 (k-Nearest Neighbors & Latency Analysis) for Project 11: Image-Tagging.

Please read docs/team/README.md and my individual guide: docs/team/04_Vipul_KNN_Classifier.md.

Tasks:
1. Work on branch feat/vipul-knn.
2. Implement src/aml/models/knn.py with KNeighborsClassifier and batched prediction helper (predict_in_batches) to prevent memory issues.
3. Tune k in [3, 5, 7, 9] and weights in ['uniform', 'distance'] on validation set for both Raw Pixels and HOG Descriptors.
4. Fit best model on full 60k training set and evaluate on 10k test set.
5. In notebooks/04_knn_experiments.ipynb:
   - Present tuning table and test metrics.
   - Conduct an in-depth Latency & Dimensionality study: profile prediction latency per image on 784-d pixels vs 324-d HOG, and document memory footprint (storing 60,000 reference vectors).
6. Export standardized results to reports/results/knn_pixels.json and reports/results/knn_hog.json.
7. Verify all metrics and prepare PR for Anant Jain (anantjain-afk) to merge.
```

---

### 5. Aditya (Milestone 3: Random Forest)
```text
I am Aditya (GitHub: AdityaInnovates). I own Model 3 (Random Forest Classifier & Feature Importance) for Project 11: Image-Tagging.

Please read docs/team/README.md and my individual guide: docs/team/05_Aditya_Random_Forest.md.

Tasks:
1. Work on branch feat/aditya-rf.
2. Implement src/aml/models/random_forest.py with parallelized RandomForestClassifier(n_jobs=-1).
3. Tune n_estimators in [50, 100, 200] and max_depth in [10, 20, None] on validation set for both Raw Pixels and HOG Descriptors.
4. Fit best forest on full 60k training set and evaluate on 10k test set.
5. In notebooks/05_random_forest.ipynb:
   - Present tuning table and test metrics.
   - Conduct Overfitting Analysis: plot Train vs. Val accuracy across tree depth, highlighting memorization when max_depth=None.
   - Map Gini Feature Importances into a 28x28 spatial heatmap.
6. Export standardized results to reports/results/rf_pixels.json and reports/results/rf_hog.json.
7. Verify all metrics and prepare PR for Anant Jain (anantjain-afk) to merge.
```

---

### 6. Utkarsh Jain (Milestone 3: SVM & Garment Error Analysis)
```text
I am Utkarsh Jain (GitHub: UJ474). I own Model 4 (Linear SVM) and the Cross-Model Garment Error Analysis for Project 11: Image-Tagging.

Please read docs/team/README.md and my individual guide: docs/team/06_Utkarsh_SVM_Error_Analysis.md.

Tasks:
1. Work on branch feat/utkarsh-svm-error-analysis.
2. Implement src/aml/models/svm.py with StandardScaler + LinearSVC, tuning C in [0.1, 1.0, 10.0] on Pixels and HOG.
3. Implement src/aml/error_analysis.py with plot_confusion_matrix, plot_garment_confusion_submatrix, and extract_misclassifications.
4. In notebooks/06_svm_and_error_analysis.ipynb:
   - Present SVM tuning and test metrics.
   - Analyze why Shirt (Class 6) has low recall and frequently confuses with T-shirt, Pullover, and Coat.
   - Render a visual gallery of actual misclassified apparel images.
5. Write reports/error_analysis.md defining "The Gap" (why fixed HOG and flattened pixels fail on fine clothing details, formally motivating the Phase 2 CNN).
6. Export standardized results to reports/results/svm_pixels.json and reports/results/svm_hog.json.
7. Verify all metrics and prepare PR for Anant Jain (anantjain-afk) to merge.
```

---

## 3. General AI Assistant Rules & Best Practices

- **Zero Merge Conflicts:** Only write to your assigned files. Do not modify files owned by another teammate.
- **Strict Verification:** Always run the test commands included in your guide to verify array shapes, dtypes, and metric ranges before finishing.
- **Git Push & PR:** Provide exact git commands:
  ```bash
  git checkout -b <your-branch>
  git add <your-owned-files>
  git commit -m "feat(<scope>): <short description>"
  git push -u origin <your-branch>
  ```
  Remind the user to open a Pull Request pointing to `main` and tag **Anant Jain (`anantjain-afk`)** for review and merge.