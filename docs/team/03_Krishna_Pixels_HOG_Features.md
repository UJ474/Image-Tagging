# Krishna Pixel and HOG Feature Extraction

**Project 11 Automatic Clothing Category Recognition**

Team T112 | Section D | AML phase | Individual work guide

Krishna Gehlot, your job is shared pixel and hog extraction, feature checks and hog illustrations. We are building a program that looks at a small clothing image and predicts one of ten categories, such as shirt, trouser or sneaker. The current comparison uses traditional machine learning with raw pixel values and edge descriptors. Follow this guide on your own computer and submit your own work through GitHub.

| Your assignment | Details |
| --- | --- |
| GitHub account | [NSTKrishna](https://github.com/NSTKrishna) |
| Repository | [anantjain-afk/Image-Tagging ](https://github.com/anantjain-afk/Image-Tagging) |
| Main task branch | feat/krishna-feature-extraction |
| Reviewer for your work | Kartik Mehra (KartikMehra22) |
| Person whose work you review | Vipul Sharma (vs-code06) |

**Readiness:** these documents define future implementation work. At the time the guides are introduced, the repository contains the project README and team guides. The modules, experiments and datasets described below must be created by their owners before dependent commands can run. No result values or contribution records are prefilled.

## Your owned files

- `src/aml/features.py`
- `notebooks/02_features.ipynb`
- `reports/feature_notes.md`
- `reports/figures/features/` (generated outputs in this folder)

Keep your implementation within these areas. Request a coordinated change if you need a shared interface adjusted. You also maintain your own work guide if the team changes your assignment; keep its Word and Markdown copies aligned.

## Set up your computer and GitHub identity

Use your own computer and your own GitHub account, `NSTKrishna`. Install Git and Python 3.12, plus VS Code or another editor. Anant must invite you as a collaborator; accept the invitation before pushing. GitHub authentication for pushing is separate from the name and email stored in a commit. Use GitHub Desktop, a credential manager or your own configured SSH key for authentication.

Open Terminal on macOS/Linux or PowerShell on Windows. Run these commands one line at a time. If you already have a clone, open that folder instead of cloning twice.

```bash
git --version
git clone https://github.com/anantjain-afk/Image-Tagging.git
cd Image-Tagging
git config user.name "Krishna Gehlot"
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
git switch -c feat/krishna-feature-extraction
```

Use `git switch feat/krishna-feature-extraction` when returning to an existing branch. Run all project commands from the Image-Tagging root. Start notebooks with `python -m jupyterlab`; choose this environment's kernel, named AML Python 3.12 after Anant registers it. In notebooks stored under notebooks/, set the working directory to the repository root before loading data or using relative paths. Re-activate the environment in each new terminal.

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

## Your prerequisites and first handoff

Anant's foundation and Kartik's dataset loader must be merged before your integration checks. You need no trained classifier. You can prepare feature functions on small synthetic arrays first, then verify them on Kartik's real normalized images. Keep the common split fingerprint with any feature-cache metadata.

## Implement your assignment step by step

### Step 1 Validate the image input

Implement extract_features in src/aml/features.py. Accept a finite float32 image array shaped (N, 28, 28), normalized to [0, 1], and representation "pixels" or "hog". Reject unknown representations and incorrect image shapes with a helpful ValueError. Do not normalize already normalized data again. Preserve input arrays rather than modifying them in place.

### Step 2 Build pixel vectors

For pixels, use reshape with C order to turn each image into a 784-value row. This keeps the same row-major pixel order for all samples. Convert the output to a contiguous float32 array. For N images, return (N, 784). This representation retains pixel values but does not explicitly encode neighboring-pixel relationships for these classifiers.

### Step 3 Build fixed HOG descriptors

Use skimage.feature.hog separately on each untransposed normalized image. Import the fixed configuration from aml.config; do not maintain a second different set of arguments. The required call is:

```python
hog(image,
    orientations=9,
    pixels_per_cell=(7, 7),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    transform_sqrt=False,
    visualize=False,
    feature_vector=True,
    channel_axis=None)
```

HOG describes edge directions in local cells. A 28 by 28 image contains 4 by 4 cells. Moving a 2 by 2-cell block one cell at a time creates 3 by 3 block positions. Each block has 2 x 2 x 9 = 36 values, so the final descriptor has 3 x 3 x 36 = 324 values. Stack rows in input order and cast the result to contiguous float32. Empty input should return shape (0, 324), and empty pixels input should return (0, 784).

HOG normalization is part of the fixed descriptor. Model-specific learned standardization belongs inside Logistic Regression and SVM pipelines, not in your shared extractor. Never fit a scaler on all splits here.

### Step 4 Illustrate what HOG sees

In notebooks/02_features.ipynb, choose training examples from all ten classes. For a gallery, call hog with visualize=True using the same descriptor settings; unpack both the descriptor and visualization. For actual extraction, continue to use visualize=False so the returned object is the descriptor alone. Display original images beside HOG images using consistent captions. Save figures to reports/figures/features/.

Explain that HOG emphasizes local shape and edges while discarding some raw intensity detail. Do not claim it must outperform pixels; that is what the experiments will test. Include shirt, T-shirt/top and pullover examples to prepare for the later error discussion.

### Step 5 Add safe optional caching

The simplest valid implementation is extraction without caching; cache_dir=None always computes the features. If you add caching, use ignored data/cache/ and key entries by a SHA-256 hash of the input image bytes and shape, representation, every HOG argument and scikit-image version. Return rows in exactly the input order. A cache keyed only by "train" or "hog" is unsafe because different splits or settings can reuse the wrong data.

Store metadata beside each feature array and validate it before reuse. Write cache outputs atomically so an interrupted extraction does not leave a valid-looking partial file. Turn caching off for feature timing; timing a warm cache is not the cost of extracting HOG. Let model owners cache locally rather than commit hundreds of megabytes to Git.

### Step 6 Demonstrate both representations to model owners

After both modules are ready, this usage must work from the repository root:

```python
from aml.data import load_splits
from aml.features import extract_features

splits = load_splits()
images = splits["train"]["images"][:20]
pixels = extract_features(images, "pixels")
hog_features = extract_features(images, "hog")
assert pixels.shape == (20, 784)
assert hog_features.shape == (20, 324)
```

Show full-split dimensions in your notebook after extraction: pixels (48000, 784), (12000, 784), (10000, 784); HOG (48000, 324), (12000, 324), (10000, 324). These arrays are local intermediate data. Write reports/feature_notes.md describing settings, orientation, dimensions, extraction behavior and cache rules.

## Your verification and acceptance checklist

- Pixel output row zero reshaped back to (28, 28) matches input image zero exactly. Extracting selected image rows gives the matching selected feature rows.
- Both representations return finite contiguous float32 arrays with the agreed dimensions and unchanged sample order.
- HOG results match a direct skimage call with the same configuration on at least a few real images.
- Repeated no-cache extraction gives the same features. A blank image gives finite values; no NaN appears from an absent edge.
- Invalid representation, wrong dimensions and non-finite data produce clear errors. The input images remain unchanged.
- If caching exists, changing input order, image content, settings or library version invalidates the old cache. With cache_dir=None, no cache is used.
- Full-split dimensions and the source split fingerprint are documented. The notebook restarts and runs cleanly, with no raw arrays staged for Git.

## Your handoff and assessment practice

Request review from Kartik. After merge, send the main revision, usage snippet and feature notes to Vipul, Aditya, Utkarsh and Anant. Tell them that their rows must remain aligned with Kartik's labels and IDs. Review Vipul's PR for accidental double normalization, incorrect feature selection or a scaler fitted outside the training pipeline.

Be able to explain the 784 and 324 dimensions, the purpose of cells and blocks, why orientation matters, why HOG is called handcrafted, what its fixed representation may miss, and why cached-feature time must not be reported as extraction time.

## Commit your work and open a pull request

Make a commit when a meaningful piece works and you can explain it. Suggested milestones for your assignment are:

1. `feat: add consistent pixel and HOG feature extraction`
2. `feat: verify feature dimensions and sample alignment`
3. `docs: illustrate HOG features and extraction behavior`

Commit frequency follows actual work; these examples are not a quota. Before committing, run the checks in your verification section, save the relevant outputs and inspect the diff. Stage owned paths explicitly. The following staging command is an example: use only the listed paths that you have actually created, and add your other completed owned files as needed.

```bash
git status --short
git diff
git add src/aml/features.py
git add notebooks/02_features.ipynb
git add reports/feature_notes.md
git diff --cached
git commit -m "feat: add consistent pixel and HOG feature extraction"
git log -1 --format=fuller
git push -u origin feat/krishna-feature-extraction
```

For later commits on the same branch, `git push` is sufficient. Open https://github.com/anantjain-afk/Image-Tagging, select Compare and pull request, choose base main and your task branch, then request review from Kartik Mehra (KartikMehra22). If GitHub asks for credentials, use your own configured authentication; never send credentials in the PR.

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

Review the work assigned to you: Vipul Sharma (vs-code06). Check that their owned code follows the common interface and that evidence supports their claims. You may review an upstream PR before your own module is ready. Your reviewer does not need their model trained to inspect your implementation, so the review ring does not block the foundation.

Anant coordinates merges after review. Use **Create a merge commit** to preserve the original individual commits. After merging, check the repository's main-branch commit page and confirm that your author name links to `NSTKrishna`. If attribution is missing, check the author email against GitHub Settings > Emails. Profile contribution graphs can take up to 24 hours to refresh; the actual commit page is the first evidence to check. A PR link and author-linked commit link are stronger evidence than a screenshot of local Git alone.

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
