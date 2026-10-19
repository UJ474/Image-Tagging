# Utkarsh Linear SVM and Classification Error Analysis

**Project 11 Automatic Clothing Category Recognition**

Team T112 | Section D | AML phase | Individual work guide

Utkarsh Jain, your job is linear svm, reusable error plots and interpretation of final predictions. We are building a program that looks at a small clothing image and predicts one of ten categories, such as shirt, trouser or sneaker. The current comparison uses traditional machine learning with raw pixel values and edge descriptors. Follow this guide on your own computer and submit your own work through GitHub.

| Your assignment | Details |
| --- | --- |
| GitHub account | [UJ474](https://github.com/UJ474) |
| Repository | [anantjain-afk/Image-Tagging ](https://github.com/anantjain-afk/Image-Tagging) |
| Main task branch | feat/utkarsh-svm-analysis |
| Reviewer for your work | Aditya (AdityaInnovates) |
| Person whose work you review | Anant Jain (anantjain-afk) |

**Readiness:** these documents define future implementation work. At the time the guides are introduced, the repository contains the project README and team guides. The modules, experiments and datasets described below must be created by their owners before dependent commands can run. No result values or contribution records are prefilled.

## Your owned files

- `src/aml/models/svm.py`
- `src/aml/analysis.py`
- `notebooks/05_svm_analysis.ipynb`
- `reports/svm_notes.md`
- `reports/error_analysis.md`
- `reports/validation/svm/` (generated outputs in this folder)
- `reports/figures/errors/` (generated outputs in this folder)
- `reports/selected/svm_pixels.json`
- `reports/selected/svm_hog.json`

Keep your implementation within these areas. Request a coordinated change if you need a shared interface adjusted. You also maintain your own work guide if the team changes your assignment; keep its Word and Markdown copies aligned.

## Set up your computer and GitHub identity

Use your own computer and your own GitHub account, `UJ474`. Install Git and Python 3.12, plus VS Code or another editor. Anant must invite you as a collaborator; accept the invitation before pushing. GitHub authentication for pushing is separate from the name and email stored in a commit. Use GitHub Desktop, a credential manager or your own configured SSH key for authentication.

Open Terminal on macOS/Linux or PowerShell on Windows. Run these commands one line at a time. If you already have a clone, open that folder instead of cloning twice.

```bash
git --version
git clone https://github.com/anantjain-afk/Image-Tagging.git
cd Image-Tagging
git config user.name "Utkarsh Jain"
git config user.email "YOUR_GITHUB_LINKED_EMAIL"
git config --get user.name
git config --get user.email
```

Replace `YOUR_GITHUB_LINKED_EMAIL` with your own verified account email or the exact private noreply address shown in GitHub Settings > Emails. Do not copy another member's email. These commands set this repository's identity only.

Create and activate the Python environment on macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version
```

On Windows PowerShell, use these instead:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

If PowerShell blocks activation, skip activation and replace `python` in later commands with `.\.venv\Scripts\python.exe`. This avoids changing machine-wide execution policy. On macOS/Linux without activation, use `.venv/bin/python`.

Wait for Anant to merge the foundation before the installation below. A missing requirements.txt or pyproject.toml means that prerequisite is not ready; do not create a competing version.

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pip check
```

Create your branch from current main, with a clean working tree:

```bash
git status
git switch main
git pull --ff-only origin main
git switch -c feat/utkarsh-svm-analysis
```

Use `git switch feat/utkarsh-svm-analysis` when returning to an existing branch. Run all project commands from the Image-Tagging root. Start notebooks with `python -m jupyterlab`; choose this environment's kernel, named AML Python 3.12 after Anant registers it. In notebooks stored under notebooks/, set the working directory to the repository root before loading data or using relative paths. Re-activate the environment in each new terminal.

## Shared experiment rules

These are team decisions. Use the same rules even when you run experiments on a different laptop. A representation describes the input image; a classifier learns to predict its label. We compare 2 representations with 4 classifiers, making 8 final combinations. CNN training belongs to the later DL phase.

| Decision | Team agreement |
| --- | --- |
| Dataset | Official Fashion-MNIST, 28 by 28 grayscale images, labels 0 through 9 |
| Development split | 48,000 training and 12,000 validation images from the official training set |
| Final test | Original official 10,000 test images, in their original order |
| Reproducibility | Seed 42, one shared dependency configuration, saved split fingerprints |
| Normalization | Divide original pixels by 255 once; output float32 values in [0, 1] |
| Features | pixels has 784 columns; hog has 324 columns |
| HOG | 9 orientations, cells (7, 7), blocks (2, 2), L2-Hys, transform_sqrt=False |
| Model selection | Highest validation macro F1, then validation accuracy, then the simpler candidate |
| Final fitting | Freeze all 8 selections first; fit fresh selected pipelines on all 60,000 training images |

Keep image orientation and labels intact. Do not transpose, rotate, augment, apply PCA, or add extra representations in these required experiments. Small balanced subsets are for smoke checks only; label such outputs clearly and keep them out of the final table. Every reported candidate must use the same full training and validation sets. Fit learned preprocessing only on the training portion, inside a Pipeline. Refit that preprocessing on all official training images during final fitting. The test set never chooses settings.

The search is fixed: Logistic Regression C in [0.1, 1, 10]; KNN n_neighbors in [3, 5, 7] and weights in [uniform, distance]; Random Forest n_estimators in [100, 200] and max_depth in [20, None]; LinearSVC C in [0.1, 1, 10]. Run every candidate for both features. If both scores tie exactly, prefer lower C for the linear models; for KNN prefer larger k, then uniform weights; for Random Forest prefer depth 20, then fewer trees. These tie rules are conventions to make selection reproducible.

Use class order [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] everywhere. The names are T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag and Ankle boot. In plots, confusion-matrix rows are actual classes and columns are predicted classes.

Report accuracy and macro F1 as fractions from 0 to 1 in files. Calculate F1 with average="macro", labels=range(10), zero_division=0. Per-class recall means correctly recognized examples of a class divided by all actual examples of that class. Report model-specific complexity and serialized bytes separately: there is no universal parameter count across these four algorithms. All official timing comparisons run on Anant's computer with a recorded environment, rather than comparing times from different laptops.

## Shared interfaces and output agreement

The following files and commands are specifications for the team to implement. They are not supplied by these guides. Work from the repository root after Anant's foundation merges. The Python package is named aml and lives under src/aml.

```python
# Kartik owns aml.data
load_official(data_dir="data/raw", download=True)
load_splits(data_dir="data/raw", download=True)
# Both return dictionaries of named splits.
# A split is {"images": X, "labels": y, "ids": ids}.
# load_official keys: train, test (60000, 10000).
# load_splits keys: train, val, test (48000, 12000, 10000).
# X: float32 (N, 28, 28); y and ids: int64 (N,).
# ids refer to indices in the relevant official source split.

# Krishna owns aml.features
extract_features(images, representation, cache_dir=None)
# representation is "pixels" or "hog".
# Return a finite float32 array (N, 784) or (N, 324).

# Model owners return fresh UNFITTED objects.
make_logreg(C=1.0)
make_knn(n_neighbors=5, weights="uniform")
make_random_forest(n_estimators=100, max_depth=20)
make_svm(C=1.0)

# Anant owns aml.evaluation
evaluate_predictions(y_true, y_pred)
# Return accuracy, macro_f1, recall (length 10),
# confusion_matrix (10, 10), and support (length 10).
```

Anant owns the runner and its output writers; model owners use that runner instead of creating competing result formats. A validation command saves every candidate to reports/validation/<model>/<features>.csv and one selection to reports/selected/<model>_<features>.json. Model IDs are logreg, knn, rf and svm; feature IDs are pixels and hog. Owners commit only their own model IDs. The model directory and empty models/__init__.py are created by Anant.

Each validation CSV row contains model, features, candidate_id, params_json, seed, train_n, val_n, split_fingerprint, package_versions, validation_accuracy, validation_macro_f1, fit_seconds, converged and warning_message. Also include training_accuracy and training_macro_f1 for overfitting inspection. A selection JSON records the winning params and scores, the same seed and fingerprint, training counts, package versions and git revision. Store JSON null for unrestricted forest depth. Exact numeric scores come from actual runs.

Final evaluation saves reports/metrics/summary.csv (8 rows), per_class_recall.csv (80 rows), one confusion CSV and one predictions CSV per combination. Prediction columns are sample_id, y_true, y_pred, ordered by official test sample_id 0 through 9999. Save all 8 prediction files before interpreting errors. Local trained pipelines go in artifacts/models/ and are not committed. Small CSVs, selections, notes and figures are committed; data/raw/, data/cache/, .venv/, artifacts/ and notebook checkpoints are ignored.

## Your prerequisites and two working stages

Your SVM experiments need Anant's foundation, Kartik's loader and Krishna's extractor. Your analysis utilities can be prepared with synthetic arrays before any final results exist. Your final interpretation needs Anant's eight frozen test-prediction CSVs. Do not wait for final testing before submitting your model and reusable plotting functions.

Use feat/utkarsh-svm-analysis for the SVM and plotting helpers. After it merges and final prediction outputs are available, start feat/utkarsh-final-error-analysis from updated main for interpretation and final error figures. Aditya reviews both PRs. You also review Anant's foundation and integration PRs, so you help keep the shared experiment valid.

## Implement your assignment step by step

### Step 1 Understand the linear SVM

An SVM learns boundaries that separate categories while controlling errors and margin size. We use LinearSVC as the required linear SVM baseline for practical full-dataset laptop experiments. Clearly label it Linear SVM in tables; do not imply it uses an RBF kernel. The original AML brief permits an SVM without specifying a kernel.

### Step 2 Implement the scaled SVM constructor

Create src/aml/models/svm.py with make_svm(C=1.0). Return a fresh sklearn Pipeline with named steps scaler and model. Use StandardScaler() followed by LinearSVC(C=C, dual="auto", tol=1e-4, max_iter=10000, random_state=42). The returned object is unfitted; no dataset access occurs at module import.

The pipeline fits its scaler on the 48,000 training rows during development and only transforms validation rows. Final refitting learns scaler statistics from all 60,000 official training rows. Use the same constructor settings for pixels and HOG. You do not need predict_proba for the requested evaluation; decision_function scores are not calibrated probabilities.

### Step 3 Smoke check and run the full search

Use the shared runner once dependencies and your constructor exist:

```bash
python -m aml.experiments --model svm --features all --stage smoke
python -m aml.experiments --model svm --features all --stage validation
```

Run C [0.1, 1, 10] for each representation, making 6 full validation fits. Check 3 rows per feature CSV and select each winner by macro F1, accuracy and then lower C for an exact tie. Capture ConvergenceWarning and solver iteration information. If a candidate does not converge at 10,000 iterations, coordinate an increase to 20,000 across the whole SVM grid, record it and rerun both representations. Keep the agreed budget in versioned code/selection metadata for final refitting. Remaining warnings block freezing until resolved with Anant; do not silently ignore them or drop a losing candidate.

Write reports/svm_notes.md with actual results, selected configurations, split fingerprint, settings and convergence evidence. Prepare the notebook's SVM section using saved tables rather than hidden interactive state.

### Step 4 Build reusable confusion and error helpers

In src/aml/analysis.py implement these interfaces. Helpers return matplotlib Figure objects and save a figure only when output_path is provided. Avoid model fitting inside plotting functions.

```python
plot_confusion(y_true, y_pred, normalize=False,
               output_path=None)
plot_misclassifications(images, y_true, y_pred,
                        sample_ids, max_examples=12,
                        output_path=None)
```

plot_confusion uses the common ten-class order, actual classes on rows and predictions on columns. With normalize=False show integer counts; with normalize=True divide each row by that actual-class support, handle zero-support rows with zeros and label the scale clearly. Use a readable colormap and class names. A perfect synthetic example should produce a diagonal plot.

plot_misclassifications first checks that images, truth, predictions and sample_ids have equal lengths. Select only mismatched rows, ordered by sample_id, and display at most max_examples. Use a grayscale image with actual label, predicted label and sample ID in every caption. If there are no errors, return a readable figure stating that there are no misclassifications. Do not return a confusing empty grid or raise an accidental indexing error.

### Step 5 Submit the SVM and plotting utilities first

Check the helpers on synthetic predictions and development images. Use validation predictions to illustrate behavior while developing. Commit the constructor, reusable helpers, notebook's SVM section, notes, two candidate CSVs and two selection JSONs. Keep final-test conclusions unwritten until Anant supplies the actual frozen predictions. Ask Aditya to review and Anant to merge this first PR so final evaluation can use your constructor.

### Step 6 Validate the final prediction handoff

For the second PR, update main and create feat/utkarsh-final-error-analysis. Read Anant's reports/metrics/<model>_<features>_predictions.csv files for all 8 combinations. Load official test images with load_official and align by sample_id; never assume a separately shuffled notebook array has the same order. Require exactly IDs 0..9999, no duplicates or missing IDs, and y_true identical across all files and the official labels.

Recompute confusion counts and recall using the common evaluation function. Counts must sum to 10,000. An alignment failure stops the analysis until fixed; otherwise it can show the wrong clothing image under a valid-looking caption.

### Step 7 Compare confusions and choose transparent examples

Save confusion figures for all eight combinations under reports/figures/errors/. For detailed examples, use Anant's overall winning combination, chosen by its frozen validation scores using the same ranking; if tied across models/features, use a documented alphabetical model-feature order. Choose the first 12 misclassified IDs to avoid selectively showing only striking examples. Also compare pixels versus HOG for this same classifier and show the lowest-ID cases where one is correct and the other is wrong, up to 6 in each direction.

Inspect the actual-class rows for Shirt (6), T-shirt/top (0) and Pullover (2). Report both directions of important confusion pairs because Shirt -> T-shirt/top and T-shirt/top -> Shirt have different actual-class denominators. Include counts and row-based rates. Relate observations to visible silhouette, sleeves, neckline, low resolution and fixed descriptor limitations when supported by the displayed examples. Treat explanations as interpretations of evidence, not proven causes. Also identify one easier category from observed recall and explain what its examples suggest.

### Step 8 Write the final concise error analysis

In reports/error_analysis.md include the overall frozen comparison, the most-confused category pairs, supporting sample IDs and figure paths, how pixels and HOG differ, and limitations. Explain that Fashion-MNIST has centered, low-resolution grayscale items and does not establish performance on real catalogue photos. State what the later CNN comparison should investigate; do not invent CNN improvements or promise that it resolves every error.

Update notebooks/05_svm_analysis.ipynb so the final section reads committed predictions and reproduces the figures. Provide Anant with the short analysis paragraphs and exact figure paths for the final notebook. Anant owns that combined notebook; coordinate its edits through his integration branch.

## Your verification and acceptance checklist

- SVM constructors are fresh, unfitted and train their scaler inside the pipeline only.
- There are exactly 6 full validation fits and 2 valid selected configurations, with shared fingerprint/counts and resolved convergence evidence.
- Confusion helpers preserve all ten classes and label row normalization correctly, including zero-support synthetic classes.
- Misclassification helpers handle no-error inputs and show only actual mistakes with matching image IDs.
- The eight final prediction files align with the official 10,000 test labels; recomputed metrics agree with Anant's summary.
- Error examples use a stated deterministic selection rule, and category comparisons include counts or rates with clear denominators.
- Every claimed outcome has a real table, plot or sample ID behind it. The notebook restarts and runs, and the report distinguishes observations from interpretations.

## Your handoff and assessment practice

First hand Anant the SVM constructor, selections and reusable helpers after Aditya's review. Later hand him the final analysis, figures and notebook section from the second PR. Review his final notebook for correct prediction alignment, class order and statements about the future CNN phase.

Be able to explain a linear decision boundary, C and regularization, why scaling belongs inside the pipeline, a confusion-matrix row, recall versus accuracy, sample alignment, and why inspecting errors must not turn into test-set tuning.

The primary classifier reference is [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html).

## Commit your work and open a pull request

Make a commit when a meaningful piece works and you can explain it. Suggested milestones for your assignment are:

1. `feat: add scaled linear SVM and validation search`
2. `feat: add confusion and misclassification plotting helpers`
3. `docs: analyze frozen AML classification errors`

Commit frequency follows actual work; these examples are not a quota. Before committing, run the checks in your verification section, save the relevant outputs and inspect the diff. Stage owned paths explicitly. The following staging command is an example: use only the listed paths that you have actually created, and add your other completed owned files as needed.

```bash
git status --short
git diff
git add src/aml/models/svm.py
git add src/aml/analysis.py
git add notebooks/05_svm_analysis.ipynb
git diff --cached
git commit -m "feat: add scaled linear SVM and validation search"
git log -1 --format=fuller
git push -u origin feat/utkarsh-svm-analysis
```

For later commits on the same branch, `git push` is sufficient. Open https://github.com/anantjain-afk/Image-Tagging, select Compare and pull request, choose base main and your task branch, then request review from Aditya (AdityaInnovates). If GitHub asks for credentials, use your own configured authentication; never send credentials in the PR.

Use this PR body and fill it from your actual work:

```text
Purpose
What this change adds and who will use it:

Implementation
Owned modules and interfaces completed:

Verification
Commands run, expected checks and actual outcomes:
Dataset size, feature configuration and environment:

Evidence
Links to committed notes, small result files or figures:
Warnings or unfinished prerequisites:

Handoff
Next person and the steps they can now run:
```

Review the work assigned to you: Anant Jain (anantjain-afk). Check that their owned code follows the common interface and that evidence supports their claims. You may review an upstream PR before your own module is ready. Your reviewer does not need their model trained to inspect your implementation, so the review ring does not block the foundation.

Anant coordinates merges after review. Use **Create a merge commit** to preserve the original individual commits. After merging, check the repository's main-branch commit page and confirm that your author name links to `UJ474`. If attribution is missing, check the author email against GitHub Settings > Emails. Profile contribution graphs can take up to 24 hours to refresh; the actual commit page is the first evidence to check. A PR link and author-linked commit link are stronger evidence than a screenshot of local Git alone.

When a dependency merges, first commit your current work, then update your branch:

```bash
git fetch origin
git merge origin/main
```

If there is a conflict, read both versions and coordinate with the file owner. After resolving, stage the resolved files, make the merge commit and rerun the relevant checks. `git merge --abort` cancels an unfinished merge if you need to start again. Do not overwrite another person's module or rewrite shared history to fix a routine conflict. After your PR merges, start additional work from updated main on a fresh branch.

## Completion and handoff record

Fill this record when the work is ready. Blank fields are intentional and must contain real evidence later.

| Item | Your record |
| --- | --- |
| Pull request URL | |
| Author linked commit URL | |
| Reviewer and review outcome | |
| Commands run and outcome | |
| Package versions and Git revision | |
| Remaining issue and owner | |
| Handoff recipient and acknowledgement | |

## Common problems and how to recover

- **Permission denied when pushing:** accept the collaborator invitation and check that your Git authentication uses your own account. Confirm origin points to anantjain-afk/Image-Tagging.
- **ModuleNotFoundError for aml:** activate the correct environment, return to the repository root and rerun the editable-install command after the foundation exists.
- **Different scores from a teammate:** compare Git revision, package versions, selected settings, split fingerprint and preprocessing. Confirm that both runs used the full agreed splits.
- **Notebook runs only on your laptop:** remove absolute personal paths, select the correct kernel, set the root directory and restart the kernel before Run All. Put reusable logic in your owned Python module.
- **Large or slow experiment:** do a balanced small smoke check first, then process data or predictions in batches. Keep the agreed full data and settings for reportable runs; ask Anant to coordinate a shared-machine rerun if your laptop cannot complete one.
- **Changed shared interface needed:** explain the proposed change to Anant before editing config or another owner's module. Update every affected guide and caller together if the team adopts it.

## References

The split, search grid, ownership and handoff rules above are team choices. The links below explain the dataset, APIs and GitHub behavior.

- [Fashion-MNIST official repository](https://github.com/zalandoresearch/fashion-mnist)
- [scikit-image HOG API](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog)
- [scikit-learn leakage and preprocessing guidance](https://scikit-learn.org/stable/common_pitfalls.html)
- [Python virtual environments](https://docs.python.org/3/library/venv.html)
- [GitHub commit email configuration](https://docs.github.com/en/account-and-profile/how-tos/email-preferences/setting-your-commit-email-address)
- [GitHub merge strategies](https://docs.github.com/en/pull-requests/reference/pull-request-merges)
- [GitHub contribution troubleshooting](https://docs.github.com/en/account-and-profile/how-tos/contribution-settings/troubleshooting-missing-contributions)
