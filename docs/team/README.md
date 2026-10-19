# Project 11 team work guides

Team T112, Section D. Each member works on their own computer and contributes through their own GitHub account. Send a member their Word guide; the matching Markdown copy is readable on GitHub. Each guide includes setup, shared technical rules, personal implementation steps, checks, commit examples and handoff instructions.

**Current status:** these are implementation instructions. The dataset loader, feature extractor, models, runner and notebooks described in the guides are future work assigned to the team. No experiment scores, merged contributions or trained models are claimed here.

## Individual guides

| Member | GitHub | Assignment | Word document | Markdown guide |
| --- | --- | --- | --- | --- |
| Anant Jain | [anantjain-afk](https://github.com/anantjain-afk) | Repository foundation, experiment runner, final evaluation and integration | [Download Word](01_Anant_Setup_Evaluation_Integration.docx) | [Read guide](01_Anant_Setup_Evaluation_Integration.md) |
| Kartik Mehra | [KartikMehra22](https://github.com/KartikMehra22) | Official dataset loader, reproducible splits and data exploration | [Download Word](02_Kartik_Dataset_Preprocessing.docx) | [Read guide](02_Kartik_Dataset_Preprocessing.md) |
| Krishna Gehlot | [NSTKrishna](https://github.com/NSTKrishna) | Shared pixel and HOG extraction, feature checks and HOG illustrations | [Download Word](03_Krishna_Pixels_HOG_Features.docx) | [Read guide](03_Krishna_Pixels_HOG_Features.md) |
| Vipul Sharma | [vs-code06](https://github.com/vs-code06) | Logistic Regression and KNN on both image representations | [Download Word](04_Vipul_Logistic_Regression_KNN.docx) | [Read guide](04_Vipul_Logistic_Regression_KNN.md) |
| Aditya | [AdityaInnovates](https://github.com/AdityaInnovates) | Random Forest comparisons, overfitting analysis and forest complexity | [Download Word](05_Aditya_Random_Forest.docx) | [Read guide](05_Aditya_Random_Forest.md) |
| Utkarsh Jain | [UJ474](https://github.com/UJ474) | Linear SVM, reusable error plots and interpretation of final predictions | [Download Word](06_Utkarsh_SVM_Error_Analysis.docx) | [Read guide](06_Utkarsh_SVM_Error_Analysis.md) |

## How to start

1. Anant builds and merges the foundation first. Utkarsh reviews it without waiting for his own model. Each member can read their guide and prepare their owned work while dependencies are pending.
2. Kartik merges the official dataset loader and reproducible splits.
3. Krishna merges the shared pixel and HOG extractor.
4. Vipul, Aditya and Utkarsh run their model searches in parallel on the same development split and commit their selected configurations.
5. Anant freezes all eight selections, refits on the official training set, evaluates the official test set and benchmarks on one computer.
6. Utkarsh interprets the frozen predictions; Anant integrates the final notebook and contribution evidence.

## Ownership and review

| Contributor | Reviewer | Branch |
| --- | --- | --- |
| Anant | Utkarsh Jain (UJ474) | `feat/anant-project-foundation` |
| Kartik | Anant Jain (anantjain-afk) | `feat/kartik-data-preprocessing` |
| Krishna | Kartik Mehra (KartikMehra22) | `feat/krishna-feature-extraction` |
| Vipul | Krishna Gehlot (NSTKrishna) | `feat/vipul-logreg-knn` |
| Aditya | Vipul Sharma (vs-code06) | `feat/aditya-random-forest` |
| Utkarsh | Aditya (AdityaInnovates) | `feat/utkarsh-svm-analysis` |

Anant uses `feat/anant-integration` after the foundation. Utkarsh uses `feat/utkarsh-final-error-analysis` after his model and helpers merge. Anant coordinates all merges using **Create a merge commit**, preserving individual commits. Each member uses their own GitHub-linked commit email and opens their own PR. The review ring does not require a reviewer to finish their own implementation first.

## Shared agreement

- AML phase only: flattened pixels and HOG with Logistic Regression, KNN, Random Forest and Linear SVM; eight final combinations.
- Official 60,000/10,000 split; seed-42 stratified 48,000/12,000 development split. Never select settings on test results.
- Normalize pixels once to float32 in [0, 1]; retain labels, image orientation and sample order.
- Pixel width 784; HOG width 324 with 9 orientations, 7 by 7 cells, 2 by 2 cell blocks and L2-Hys normalization.
- Choose by validation macro F1, then accuracy, then the defined simplicity rule. Freeze all selections before final test evaluation.
- Report accuracy, macro F1, class recall, confusion examples, model-specific complexity, model size and comparable inference timings.
- Keep raw downloads, feature caches, environments and trained models outside Git tracking; commit code, small actual result files and notes.
- Use milestones rather than invented deadlines. Fill contribution records with real PR and commit links when the work is ready.

## Contribution evidence to fill later

| Member | Merged PR | Author linked commit | Handoff confirmed |
| --- | --- | --- | --- |
| Anant Jain | | | |
| Kartik Mehra | | | |
| Krishna Gehlot | | | |
| Vipul Sharma | | | |
| Aditya | | | |
| Utkarsh Jain | | | |

The empty evidence cells are intentional. A guide is a work assignment, not evidence that the assignment has already been completed.
