# Kartik Dataset Loading and Preprocessing

**Project 11 Automatic Clothing Category Recognition**

Team T112 | Section D | AML phase | Individual work guide

Kartik Mehra, your job is official dataset loader, reproducible splits and data exploration. We are building a program that looks at a small clothing image and predicts one of ten categories, such as shirt, trouser or sneaker. The current comparison uses traditional machine learning with raw pixel values and edge descriptors. Follow this guide on your own computer and submit your own work through GitHub.

| Your assignment | Details |
| --- | --- |
| GitHub account | [KartikMehra22](https://github.com/KartikMehra22) |
| Repository | [anantjain-afk/Image-Tagging ](https://github.com/anantjain-afk/Image-Tagging) |
| Main task branch | feat/kartik-data-preprocessing |
| Reviewer for your work | Anant Jain (anantjain-afk) |
| Person whose work you review | Krishna Gehlot (NSTKrishna) |

**Readiness:** these documents define future implementation work. At the time the guides are introduced, the repository contains the project README and team guides. The modules, experiments and datasets described below must be created by their owners before dependent commands can run. No result values or contribution records are prefilled.

## Your owned files

- `src/aml/data.py`
- `notebooks/01_data_exploration.ipynb`
- `reports/data_overview.md`
- `reports/figures/data/` (generated outputs in this folder)

Keep your implementation within these areas. Request a coordinated change if you need a shared interface adjusted. You also maintain your own work guide if the team changes your assignment; keep its Word and Markdown copies aligned.

## Set up your computer and GitHub identity

Use your own computer and your own GitHub account, `KartikMehra22`. Install Git and Python 3.12, plus VS Code or another editor. Anant must invite you as a collaborator; accept the invitation before pushing. GitHub authentication for pushing is separate from the name and email stored in a commit. Use GitHub Desktop, a credential manager or your own configured SSH key for authentication.

Open Terminal on macOS/Linux or PowerShell on Windows. Run these commands one line at a time. If you already have a clone, open that folder instead of cloning twice.

```bash
git --version
git clone https://github.com/anantjain-afk/Image-Tagging.git
cd Image-Tagging
git config user.name "Kartik Mehra"
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
git switch -c feat/kartik-data-preprocessing
```

Use `git switch feat/kartik-data-preprocessing` when returning to an existing branch. Run all project commands from the Image-Tagging root. Start notebooks with `python -m jupyterlab`; choose this environment's kernel, named AML Python 3.12 after Anant registers it. In notebooks stored under notebooks/, set the working directory to the repository root before loading data or using relative paths. Re-activate the environment in each new terminal.

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

Begin after Anant merges the foundation with the aml package, config.py, requirements.txt and the agreed interfaces. You do not need Krishna's features or any classifier. Your loader is the first data dependency for everyone else. Read the official dataset description and inspect the original image orientation before preprocessing.

## Implement your assignment step by step

### Step 1 Download the four official files

Implement download helpers in src/aml/data.py using pathlib, urllib.request, hashlib and gzip from the standard library. Store raw downloads under ignored data/raw/. Use the official repository's data/fashion raw files over HTTPS, such as the base below; each filename is appended to it.

```text
https://raw.githubusercontent.com/zalandoresearch/
fashion-mnist/master/data/fashion/
```

The base URL is shown on two lines for readability. Join it without whitespace in your code. Keep the following filenames and MD5 checksums in one manifest. Validate the compressed downloaded bytes before opening them. MD5 here is the dataset's published integrity check.

| Filename | Expected MD5 |
| --- | --- |
| train-images-idx3-ubyte.gz | 8d4fb7e6c68d591d4c3dfef9ec88bf0d |
| train-labels-idx1-ubyte.gz | 25c81989df183df01b3e8a0aad5dffbe |
| t10k-images-idx3-ubyte.gz | bef4ecab320f06d8554ea6380940ec79 |
| t10k-labels-idx1-ubyte.gz | bb300cfdad3c16e7a12a480ee83cd310 |

Download to a temporary file, verify its checksum and rename it only after verification. Existing valid files should be reused. If a checksum fails, discard only that invalid download and retry with a clear error. With download=False, raise an informative FileNotFoundError for missing data instead of silently downloading. Handle network timeouts and corrupt gzip files with the offending filename in the error. Keep temporary files out of the final data manifest.

### Step 2 Read the IDX headers and image bytes

Use gzip.open to decompress and struct.unpack with big-endian integers to read the IDX header. Image files have magic number 2051, followed by count, rows and columns (16 header bytes). Label files have magic number 2049, followed by count (8 header bytes). Use numpy.frombuffer with dtype uint8 for the payload. Confirm the byte count exactly matches the header before reshape.

Reshape images into (N, 28, 28). Confirm 60,000 train images and 10,000 test images, with the same number of labels for each. Check image-label pairing, label range 0..9 and 28 by 28 dimensions. Keep the pixel order as supplied. Never guess a correction by transposing until you have checked the original loader and image display. Do not remove visually similar images or relabel confusing clothing.

### Step 3 Normalize and implement the official loader

Convert images using images.astype(np.float32) / 255.0 exactly once, and labels to int64. Return load_official's train and test dictionaries using the common images, labels and ids keys. Official train IDs are 0..59999 and test IDs are 0..9999. The ID namespaces are separate, so a test ID equal to a train ID does not mean the same image. Keep original ordering in both returned official splits.

### Step 4 Implement the shared stratified split

Use sklearn.model_selection.train_test_split on the official train IDs. Use test_size=0.2, random_state=42, stratify=training_labels and the default shuffle=True. Use the resulting train_ids and val_ids to index both images and labels. Return keys train, val and test in load_splits; keep the test arrays unchanged. Do not split all 70,000 images together.

There should be 48,000 training and 12,000 validation images, with 4,800 and 1,200 examples per class respectively. Calculate the SHA-256 fingerprint of the train IDs followed by val IDs using little-endian int64 bytes, as Anant specifies. Record it in your notes and display it in the notebook. Optional local split caches must include the seed, source checksums and fingerprint; recompute if any changes.

### Step 5 Explore the data and save useful figures

In notebooks/01_data_exploration.ipynb, import the loader instead of rewriting it. Show train/validation/test shapes, dtypes, ranges, label names and counts. Use matplotlib imshow with cmap="gray", vmin=0, vmax=1, and turn axes off. Plot a reproducible training-only gallery with at least three examples per class and a class-count chart for training and validation. Save readable figures to reports/figures/data/.

Write reports/data_overview.md with the official source, verified files, split method, fingerprint, observed balance and any orientation observations. Explain that a balanced dataset can still have visually difficult categories. Test-set structure checks are allowed, but model decisions and exploratory example selection should use development data.

### Step 6 Demonstrate the handoff from a clean environment

After the module exists and the editable package is installed, run the following in a Python session or notebook. This is a usage example of the interface you implement.

```python
from aml.data import load_splits

splits = load_splits()
for name in ("train", "val", "test"):
    item = splits[name]
    print(name, item["images"].shape, item["labels"].shape)
```

Your PR should explain how a teammate downloads the files on their own machine and verifies the same fingerprint. Share code and reproducibility evidence, not a personal absolute path or the whole dataset in Git.

## Your verification and acceptance checklist

- Both loaders return the exact dictionary keys and shapes in the shared contract. Images are float32, labels/IDs int64, pixels finite and within [0, 1].
- Train/validation IDs have no overlap, cover exactly 0..59999 together, and keep labels aligned with their original images.
- Repeated calls return identical IDs, labels and fingerprint with the same frozen dependency configuration.
- load_official preserves source order; load_splits preserves the official test order exactly.
- Count plots show 4,800 training and 1,200 validation examples per class. All ten category names match the official label mapping.
- A small malformed header or truncated-payload example fails clearly before reshape. Missing data with download=False fails clearly. A corrupted compressed file does not become an accepted download.
- The notebook runs after restart and Run All from the project root. The Git diff contains code, notes and figures, but no raw data or cache binaries.

## Your handoff and assessment practice

Request review from Anant. After merge, send the main revision, loader example, fingerprint and data overview to Krishna, Vipul, Aditya and Utkarsh. Krishna needs normalized image arrays; model owners need the exact same split. Remain available if any laptop gets a different checksum or split fingerprint.

Be able to explain why normalization divides by 255, why the official test set is separate, how stratification works, what an IDX header contains, why a stable ID matters, and how an incorrect transpose could damage HOG features. If downloads fail, retry the verified official files; do not substitute a shuffled CSV export without agreement.

## Commit your work and open a pull request

Make a commit when a meaningful piece works and you can explain it. Suggested milestones for your assignment are:

1. `feat: load and verify official Fashion-MNIST files`
2. `feat: add normalized stratified dataset splits`
3. `docs: add dataset exploration and reproducibility evidence`

Commit frequency follows actual work; these examples are not a quota. Before committing, run the checks in your verification section, save the relevant outputs and inspect the diff. Stage owned paths explicitly. The following staging command is an example: use only the listed paths that you have actually created, and add your other completed owned files as needed.

```bash
git status --short
git diff
git add src/aml/data.py
git add notebooks/01_data_exploration.ipynb
git add reports/data_overview.md
git diff --cached
git commit -m "feat: load and verify official Fashion-MNIST files"
git log -1 --format=fuller
git push -u origin feat/kartik-data-preprocessing
```

For later commits on the same branch, `git push` is sufficient. Open https://github.com/anantjain-afk/Image-Tagging, select Compare and pull request, choose base main and your task branch, then request review from Anant Jain (anantjain-afk). If GitHub asks for credentials, use your own configured authentication; never send credentials in the PR.

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

Review the work assigned to you: Krishna Gehlot (NSTKrishna). Check that their owned code follows the common interface and that evidence supports their claims. You may review an upstream PR before your own module is ready. Your reviewer does not need their model trained to inspect your implementation, so the review ring does not block the foundation.

Anant coordinates merges after review. Use **Create a merge commit** to preserve the original individual commits. After merging, check the repository's main-branch commit page and confirm that your author name links to `KartikMehra22`. If attribution is missing, check the author email against GitHub Settings > Emails. Profile contribution graphs can take up to 24 hours to refresh; the actual commit page is the first evidence to check. A PR link and author-linked commit link are stronger evidence than a screenshot of local Git alone.

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
