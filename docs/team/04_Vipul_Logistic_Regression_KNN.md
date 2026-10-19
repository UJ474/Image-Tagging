# Vipul Logistic Regression and KNN Experiments

**Project 11 Automatic Clothing Category Recognition**

Team T112 | Section D | AML phase | Individual work guide

Vipul Sharma, your job is logistic regression and knn on both image representations. We are building a program that looks at a small clothing image and predicts one of ten categories, such as shirt, trouser or sneaker. The current comparison uses traditional machine learning with raw pixel values and edge descriptors. Follow this guide on your own computer and submit your own work through GitHub.

| Your assignment | Details |
| --- | --- |
| GitHub account | [vs-code06](https://github.com/vs-code06) |
| Repository | [anantjain-afk/Image-Tagging ](https://github.com/anantjain-afk/Image-Tagging) |
| Main task branch | feat/vipul-logreg-knn |
| Reviewer for your work | Krishna Gehlot (NSTKrishna) |
| Person whose work you review | Aditya (AdityaInnovates) |

**Readiness:** these documents define future implementation work. At the time the guides are introduced, the repository contains the project README and team guides. The modules, experiments and datasets described below must be created by their owners before dependent commands can run. No result values or contribution records are prefilled.

## Your owned files

- `src/aml/models/logreg_knn.py`
- `notebooks/03_logreg_knn.ipynb`
- `reports/logreg_knn_notes.md`
- `reports/validation/logreg/` (generated outputs in this folder)
- `reports/validation/knn/` (generated outputs in this folder)
- `reports/selected/logreg_pixels.json`
- `reports/selected/logreg_hog.json`
- `reports/selected/knn_pixels.json`
- `reports/selected/knn_hog.json`

Keep your implementation within these areas. Request a coordinated change if you need a shared interface adjusted. You also maintain your own work guide if the team changes your assignment; keep its Word and Markdown copies aligned.

## Set up your computer and GitHub identity

Use your own computer and your own GitHub account, `vs-code06`. Install Git and Python 3.12, plus VS Code or another editor. Anant must invite you as a collaborator; accept the invitation before pushing. GitHub authentication for pushing is separate from the name and email stored in a commit. Use GitHub Desktop, a credential manager or your own configured SSH key for authentication.

Open Terminal on macOS/Linux or PowerShell on Windows. Run these commands one line at a time. If you already have a clone, open that folder instead of cloning twice.

```bash
git --version
git clone https://github.com/anantjain-afk/Image-Tagging.git
cd Image-Tagging
git config user.name "Vipul Sharma"
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
git switch -c feat/vipul-logreg-knn
```

Use `git switch feat/vipul-logreg-knn` when returning to an existing branch. Run all project commands from the Image-Tagging root. Start notebooks with `python -m jupyterlab`; choose this environment's kernel, named AML Python 3.12 after Anant registers it. In notebooks stored under notebooks/, set the working directory to the repository root before loading data or using relative paths. Re-activate the environment in each new terminal.

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

## Your prerequisites and expected outputs

Before full experiments, merge Anant's foundation, Kartik's loader and Krishna's extractor into your branch. You can prepare constructor functions and small synthetic checks while these are being reviewed. You own four final combinations: Logistic Regression with pixels/HOG and KNN with pixels/HOG. These two comparatively simple models form your assignment; other members own the heavier single-model and analysis tasks.

Expected validation work is 6 Logistic Regression fits and 12 KNN fits. Give Anant four selected JSON configurations, the complete candidate tables and the notes needed to reproduce them. Your local timing is useful for debugging but does not enter the official timing comparison.

## Implement your assignment step by step

### Step 1 Understand the two classifiers

Logistic Regression learns weights that connect feature values to the ten category scores. Although its name says regression, it is a classification method here. KNN stores labeled training examples and predicts a new image from nearby feature vectors. The feature representation defines what nearby means. Neither classifier learns convolutional image features.

### Step 2 Write fresh unfitted constructors

In src/aml/models/logreg_knn.py implement make_logreg(C=1.0) and make_knn(n_neighbors=5, weights="uniform"). Import shared constants from config.py and follow the common naming. Constructors return a new object every call and never read a dataset or fit during import.

make_logreg returns a sklearn.pipeline.Pipeline with named steps scaler and model. Use StandardScaler() followed by LogisticRegression(C=C, solver="lbfgs", max_iter=2000, tol=1e-4, random_state=42). Do not add deprecated multi_class arguments. The scaler is fitted only through pipeline.fit on the current training data. Keep its learned values with the fitted pipeline for validation and final inference.

make_knn returns KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm="brute", metric="minkowski", p=2, n_jobs=1). The distance is Euclidean. It uses the normalized pixels or HOG as supplied, without an extra StandardScaler. Keep these fixed choices consistent for both representations and record them in your notes. Batch predictions through Anant's runner rather than allocating a full validation-by-training distance matrix yourself.

### Step 3 Smoke check before the full search

After implementing the constructors and updating your branch, use the shared runner. These commands become available when Anant implements it and your dependencies exist:

```bash
python -m aml.experiments --model logreg --features all --stage smoke
python -m aml.experiments --model knn --features all --stage smoke
```

Confirm both representations produce predictions shaped (N,) and labels in 0..9. Check that constructor calls do not share fitted state. Use a small artificial classification example to confirm the Logistic Regression pipeline can fit and predict and KNN can use both weight modes. Smoke outputs must not replace full validation tables.

### Step 4 Run the fixed full validation searches

Run all Logistic Regression C values [0.1, 1, 10] and all KNN pairs from n_neighbors [3, 5, 7] with weights [uniform, distance], for both pixels and hog. Use the 48,000 training images and 12,000 validation images from Kartik unchanged. The shared runner owns iteration, scoring and file formats:

```bash
python -m aml.experiments --model logreg --features all --stage validation
python -m aml.experiments --model knn --features all --stage validation
```

The logreg CSV for each feature must have 3 candidates; the knn CSV must have 6. Confirm seed, fingerprint, feature dimensions and counts before comparing scores. Inspect the selected configuration for each of your four combinations. Choose by validation macro F1, then validation accuracy and the agreed exact-tie preference. Do not inspect test performance while selecting.

### Step 5 Handle convergence and computational cost explicitly

Watch for sklearn.exceptions.ConvergenceWarning in Logistic Regression. Record it instead of hiding it. If any candidate fails to converge at 2,000 iterations, coordinate one increase to max_iter=5000 for the entire Logistic Regression grid and rerun both representations with that limit recorded. Iteration budget is a solver correction, not a new accuracy-tuned parameter. Resolve remaining warnings with Anant before freezing; do not quietly drop the affected candidate or change solver on only the winning model. Store the agreed iteration limit in the selected configuration or in versioned constructor code so final fitting uses it too.

KNN has little fitting computation but may take longer to predict because it compares against stored examples. Do not describe its low fit time as evidence of faster inference. If your machine runs out of memory, reduce prediction batch size consistently and coordinate with Anant; keep all training examples and log the batch size. KNN training-set accuracy can be unusually high because a training example is its own neighbor. Explain this in the overfitting discussion rather than treating training accuracy as an unbiased estimate.

### Step 6 Write the notebook and practical interpretation

In notebooks/03_logreg_knn.ipynb import the constructors and load the saved validation outputs. Present each model's pixel versus HOG comparison, candidate tables and chosen settings. Explain macro F1 in simple words and note any category-specific weakness from validation predictions. Store concise interpretation in reports/logreg_knn_notes.md. Include configuration, package versions, split fingerprint, convergence status and full-data commands.

For model complexity later, Logistic Regression's classifier coefficient count is coef_.size + intercept_.size. KNN has no comparable trained coefficient count; record stored examples, feature dimension and stored values. Anant will calculate these from the final 60,000-example fits and measure serialized pipeline sizes consistently. Your development KNN stores 48,000 examples, so do not copy that number into final complexity results.

### Step 7 Hand over configurations for final testing

Commit the Python file containing both constructors, notebook, notes, four candidate CSVs and four selection JSONs. The JSONs must include the selected params and actual validation evidence; do not hand over only a pickle on your laptop. Anant will refit fresh selected pipelines on all 60,000 official training examples, run final testing and generate the official comparison. You can help reproduce problems without changing your frozen selections after test scores appear.

## Your verification and acceptance checklist

- Both constructors return fresh unfitted objects, with no data read or fitting at import time.
- Logistic Regression includes its scaler inside Pipeline and fits only the current training portion; KNN uses exactly the supplied common features.
- Predictions have the right length and valid labels; no train/validation sample IDs overlap.
- All 18 full validation fits are present: 3 C values times 2 representations, plus 6 KNN pairs times 2 representations.
- CSV rows and four selected JSONs record full counts, the same fingerprint, actual scores and fixed settings. Recomputing selection from the CSV gives the same winner.
- Convergence warnings are resolved or explicitly block freezing; no candidate is silently omitted.
- Notebook execution does not require cached hidden state, personal absolute paths or local model files that were never shared.
- Small reports are staged and raw data, feature arrays and serialized trained models remain ignored.

## Your handoff and assessment practice

Request review from Krishna. Send Anant your merged revision, four selection files and notes. Review Aditya's Random Forest PR for valid split use and a complete grid. Be able to explain the meaning of C, why scaling helps Logistic Regression, what k changes, uniform versus distance voting, why KNN prediction can be expensive, and why HOG may change nearest neighbors.

Useful classifier references are [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) and [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).

## Commit your work and open a pull request

Make a commit when a meaningful piece works and you can explain it. Suggested milestones for your assignment are:

1. `feat: add Logistic Regression and KNN constructors`
2. `feat: record full pixel and HOG validation searches`
3. `docs: explain Logistic Regression and KNN comparisons`

Commit frequency follows actual work; these examples are not a quota. Before committing, run the checks in your verification section, save the relevant outputs and inspect the diff. Stage owned paths explicitly. The following staging command is an example: use only the listed paths that you have actually created, and add your other completed owned files as needed.

```bash
git status --short
git diff
git add src/aml/models/logreg_knn.py
git add notebooks/03_logreg_knn.ipynb
git add reports/logreg_knn_notes.md
git diff --cached
git commit -m "feat: add Logistic Regression and KNN constructors"
git log -1 --format=fuller
git push -u origin feat/vipul-logreg-knn
```

For later commits on the same branch, `git push` is sufficient. Open https://github.com/anantjain-afk/Image-Tagging, select Compare and pull request, choose base main and your task branch, then request review from Krishna Gehlot (NSTKrishna). If GitHub asks for credentials, use your own configured authentication; never send credentials in the PR.

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

Review the work assigned to you: Aditya (AdityaInnovates). Check that their owned code follows the common interface and that evidence supports their claims. You may review an upstream PR before your own module is ready. Your reviewer does not need their model trained to inspect your implementation, so the review ring does not block the foundation.

Anant coordinates merges after review. Use **Create a merge commit** to preserve the original individual commits. After merging, check the repository's main-branch commit page and confirm that your author name links to `vs-code06`. If attribution is missing, check the author email against GitHub Settings > Emails. Profile contribution graphs can take up to 24 hours to refresh; the actual commit page is the first evidence to check. A PR link and author-linked commit link are stronger evidence than a screenshot of local Git alone.

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
