# Anant Project Setup Evaluation and Integration

**Project 11 Automatic Clothing Category Recognition**

Team T112 | Section D | AML phase | Individual work guide

Anant Jain, your job is repository foundation, experiment runner, final evaluation and integration. We are building a program that looks at a small clothing image and predicts one of ten categories, such as shirt, trouser or sneaker. The current comparison uses traditional machine learning with raw pixel values and edge descriptors. Follow this guide on your own computer and submit your own work through GitHub.

| Your assignment | Details |
| --- | --- |
| GitHub account | [anantjain-afk](https://github.com/anantjain-afk) |
| Repository | [anantjain-afk/Image-Tagging ](https://github.com/anantjain-afk/Image-Tagging) |
| Main task branch | feat/anant-project-foundation |
| Reviewer for your work | Utkarsh Jain (UJ474) |
| Person whose work you review | Kartik Mehra (KartikMehra22) |

**Readiness:** these documents define future implementation work. At the time the guides are introduced, the repository contains the project README and team guides. The modules, experiments and datasets described below must be created by their owners before dependent commands can run. No result values or contribution records are prefilled.

## Your owned files

- `pyproject.toml`
- `requirements.txt`
- `.gitignore`
- `src/aml/__init__.py`
- `src/aml/models/__init__.py`
- `src/aml/config.py`
- `src/aml/experiments.py`
- `src/aml/evaluation.py`
- `src/aml/benchmark.py`
- `notebooks/00_final_aml.ipynb`
- `reports/final_comparison.md`
- `reports/metrics/` (generated outputs in this folder)
- `reports/selected/frozen_manifest.json`
- `reports/README.md`
- `README.md`

Keep your implementation within these areas. Request a coordinated change if you need a shared interface adjusted. You also maintain your own work guide if the team changes your assignment; keep its Word and Markdown copies aligned.

## Set up your computer and GitHub identity

Use your own computer and your own GitHub account, `anantjain-afk`. Install Git and Python 3.12, plus VS Code or another editor. Anant must invite you as a collaborator; accept the invitation before pushing. GitHub authentication for pushing is separate from the name and email stored in a commit. Use GitHub Desktop, a credential manager or your own configured SSH key for authentication.

Open Terminal on macOS/Linux or PowerShell on Windows. Run these commands one line at a time. If you already have a clone, open that folder instead of cloning twice.

```bash
git --version
git clone https://github.com/anantjain-afk/Image-Tagging.git
cd Image-Tagging
git config user.name "Anant Jain"
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

Your first task is to create and verify the foundation files below. The next two installation commands become available after you create requirements.txt and pyproject.toml.

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
git switch -c feat/anant-project-foundation
```

Use `git switch feat/anant-project-foundation` when returning to an existing branch. Run all project commands from the Image-Tagging root. Start notebooks with `python -m jupyterlab`; choose this environment's kernel, named AML Python 3.12 after Anant registers it. In notebooks stored under notebooks/, set the working directory to the repository root before loading data or using relative paths. Re-activate the environment in each new terminal.

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

## Your prerequisites and the team sequence

You work in two stages. The foundation begins immediately, before any dataset or model code exists. Final integration begins after Kartik's loader, Krishna's features and all three model-owner PRs have merged. Utkarsh's visual analysis helpers may merge with his model; his interpretation follows your final prediction outputs.

The order is foundation -> dataset -> features -> parallel model experiments -> final evaluation -> error analysis and final notebook. Everyone can read, plan or prepare their own modules while waiting, but training results must come from merged shared dependencies. Your reviewer is Utkarsh; he can review the foundation without waiting for his own model work.

## Stage one build the foundation

### Step 1 Set up collaborators and boundaries

In GitHub repository Settings, invite KartikMehra22, NSTKrishna, vs-code06, AdityaInnovates and UJ474 as collaborators. Ask them to accept. Enable merge commits. If the repository supports it, require a review before merging into main; keep the workflow usable with the available GitHub plan. Create one GitHub issue per member with their assignment and acceptance checklist. Issue and invitation creation are your future tasks, not actions already performed by these documents.

Create src/aml and src/aml/models with empty __init__.py files. Create notebooks/ and reports/ areas, letting their owners add files when work begins. Do not fill another person's data, feature or model module. Add ignore rules for .venv/, data/raw/, data/cache/, artifacts/, __pycache__/, *.pyc and .ipynb_checkpoints/. Keep small reports and figures tracked. Do not globally ignore all CSV files because the results must be committed.

### Step 2 Establish one reproducible environment

Use Python 3.12. Create a setuptools pyproject.toml using the src layout and package discovery for aml. The project must support `python -m pip install -e . --no-deps`. Avoid duplicating conflicting dependency versions in project metadata and the requirements file.

In a clean environment, install numpy, pandas, scikit-image, scikit-learn, matplotlib, jupyterlab, ipykernel and joblib. Verify imports and pip check, then write the resolved compatible dependencies to requirements.txt. Do not claim an untested version list is verified. Record Python and package versions in reports/README.md. Register the notebook kernel:

```bash
python -m ipykernel install --user --name aml-project --display-name "AML Python 3.12"
```

Test the frozen requirements in a second fresh environment before merging. Teammates install this common file rather than independently upgrading libraries. Their installed versions must match before results are combined. Discuss a required dependency change through a PR and rerun affected checks.

### Step 3 Write shared configuration and metric functions

In config.py define SEED=42, VALIDATION_SIZE=0.2, class names, feature IDs, fixed HOG arguments and the agreed model grids. Keep config free of dataset downloads and model fitting. Specify that split_fingerprint is SHA-256 of the training IDs followed by validation IDs, encoded as little-endian int64 bytes. Save or display the fingerprint with experiments so teammates can verify alignment.

Implement evaluate_predictions in evaluation.py using sklearn.metrics.accuracy_score, f1_score, recall_score and confusion_matrix. Always use all ten labels for class-based outputs and zero_division=0. Validate equal-length one-dimensional integer inputs with labels in 0..9, then return JSON-friendly values. Support is the per-class actual count. A confusion matrix contains counts, with actual labels as rows. Unit-check these functions with small hand-computable arrays; no dataset is needed.

### Step 4 Build the shared runner without blocking on missing modules

Implement an argparse CLI in experiments.py. Accept model values logreg, knn, rf, svm and all; feature values pixels, hog and all; stages smoke, validation, final and benchmark. Add a separate --freeze action. Lazy-import a model module only when its model is selected. Importing aml.config, displaying CLI help and checking evaluation must succeed before the other members' modules exist. For an unavailable dependency, explain the missing owner/module in the error instead of creating a fake implementation.

```bash
python -m aml.experiments --help
```

Smoke uses a seed-42 stratified 1,000 training / 200 validation subset and the first candidate only. Write smoke outputs under ignored artifacts/smoke/ and mark them non-reportable. Validation runs every candidate against the full common 48,000/12,000 split, one fit at a time. Use sklearn.base.clone or a fresh constructor for each fit. Compute training and validation scores, record convergence warnings and save the shared CSV and selection JSON formats. Each owner runs the commands for their model IDs. The runner must not import another owner's notebook.

Expose the constructor registry: logreg and knn from aml.models.logreg_knn, rf from aml.models.random_forest and svm from aml.models.svm. Translate a candidate's parameter dictionary into its constructor keyword arguments. Candidate grids live in config.py and ownership-specific fit logic stays in constructors. Log seed, fingerprint, dimensions, installed versions and Git revision in every run.

The --freeze action requires all eight full-data selections and no unresolved convergence warning. Save their SHA-256 hashes plus the common split fingerprint to reports/selected/frozen_manifest.json. Final fitting refuses missing or changed selections. Benchmarking loads existing final pipelines without choosing settings or repeating a parameter search. This keeps the final test from influencing model selection.

### Step 5 Merge the foundation before dependent training

Verify packaging, help, synthetic metric checks and output-format writing. Open the foundation PR and ask Utkarsh to review. Explain that dataset/model commands become usable as their owners merge their modules. Once approved, merge with a merge commit and tell everyone which main revision contains the foundation. Review Kartik's loader next. No member needs to wait for final integration to start their own work.

## Stage two integrate and evaluate

### Step 6 Start the integration branch and validate handoffs

After the foundation PR merges, create feat/anant-integration from updated main. Keep it updated as dependencies land. Check load_splits shapes, feature dimensions and all constructors. Confirm every selected JSON comes from the same official split, complete search, package configuration and HOG settings. There must be 32 validation fits overall: 6 Logistic Regression, 12 KNN, 8 Random Forest and 6 SVM. Exclude smoke results from this count. Every model needs a selection for pixels and for hog.

After all eight selections are committed, freeze them:

```bash
python -m aml.experiments --freeze
```

The runner should support --freeze on its own, without requiring model, features or stage flags. Commit the frozen manifest before interpreting test results.

### Step 7 Refit all selected pipelines and save final predictions

Use Kartik's load_official train split, which preserves the original 60,000-example order. Get its features using Krishna's extractor. Build fresh selected models and fit each once on all 60,000 training images. A pipeline refits its scaler here; never keep the scaler learned on only 48,000 examples. Predict all official 10,000 test images in batches to keep KNN memory manageable.

```bash
python -m aml.experiments --model all --features all --stage final
```

For every combination, write predictions with sample_id 0..9999, y_true and y_pred; save the confusion counts, per-class recall and summary row. Save the fitted pipeline locally with joblib under artifacts/models/. Model owners keep their original validation tables intact. Give all eight committed prediction CSVs to Utkarsh for final error analysis. Do not change model settings after seeing test scores; discuss any later exploration separately from the frozen comparison.

### Step 8 Benchmark consistently on one computer

Implement benchmarking in benchmark.py with time.perf_counter. Record machine model/CPU, RAM if known, OS, Python/package versions, thread settings and batch size. Use one process and one worker per estimator for official timing; set estimator n_jobs=1 where supported and record the numerical-library thread limits. Use threadpoolctl.threadpool_limits(limits=1) if included in the frozen environment, or document equivalent environment settings before Python starts.

Use the first 1,000 official test images, in original order, and batches of 100. Warm up each measurement once, then repeat 5 times and report the median. Time feature extraction on the in-memory normalized images with cache_dir=None. Time pipeline.predict on already extracted features, including the trained pipeline's scaler. Report seconds per 1,000 images and milliseconds per image. Exclude loading, downloads, disk cache lookup, model fitting and report creation. Process batches rather than constructing an enormous KNN distance matrix. All eight models use the same protocol.

```bash
python -m aml.experiments --model all --features all --stage benchmark
```

Summary columns are model, features, test_n, accuracy, macro_f1, feature_seconds_per_1000, predict_seconds_per_1000, feature_ms_per_image, predict_ms_per_image, complexity_kind, complexity_count and model_size_bytes. Put detailed environment information in reports/metrics/benchmark_environment.json. Explain that adding the two timing components is an estimate of processing latency, not a separately measured end-to-end benchmark.

### Step 9 Describe model size honestly

For fitted Logistic Regression and LinearSVC, use coef_.size + intercept_.size as classifier trainable coefficients. Report StandardScaler mean and scale storage separately, rather than silently adding it to the classifier count. For KNN, report stored training examples and stored feature values; it has no comparable trained weight count. For Random Forest, use Aditya's total-node calculation. Record uncompressed serialized full-pipeline byte size for all models using a consistent joblib format. These different measures need a clear type label, not a single universal parameter-count ranking.

### Step 10 Assemble the final notebook and evidence

The final notebook order is project question, dataset and split evidence, representative images, feature explanation, model grids, validation selection table, frozen test comparison, per-class recall, confusion plots, wrong-image examples, error analysis and limitations. Reuse functions from modules instead of copying teammates' implementations. Load committed result files for presentation; show the exact commands needed to reproduce expensive runs. Restart the kernel and run the presentation cells from the repository root.

Work with Utkarsh to state observations supported by the images and counts. Do not write that a CNN improved results because no CNN has been trained in this phase. Add each member's actual module, PR and author-linked commit evidence to the README. Check every contributor against main history. Keep a small contribution table with real URLs when ready.

## Your verification and acceptance checklist

- Fresh Python 3.12 environment installs the frozen dependencies and editable package, and pip check passes.
- CLI help and evaluation checks pass before dataset/model modules exist. Missing-module messages identify the dependency.
- A perfect synthetic prediction returns accuracy 1, macro F1 1 if all ten classes are represented, and a diagonal confusion matrix. A deliberate single mistake gives the expected changed cell and recall.
- Training and validation IDs are disjoint, contain all 60,000 source IDs together and reproduce the same fingerprint.
- The validation stage produces the expected 32 full-data candidate rows across files, plus exactly 8 selections. Final stage requires the committed frozen manifest.
- The final table has 8 unique model-feature rows; recall has 80 unique rows; each prediction file has exactly 10,000 ordered IDs and valid labels.
- Every confusion matrix sums to 10,000; its trace divided by 10,000 equals reported accuracy. Recomputing metrics from prediction CSVs matches the final table.
- All timings come from the documented common machine and exclude training. No large generated caches or model binaries appear in git diff --cached.
- Utkarsh can build the analysis from your prediction files; the final notebook runs cleanly; all six members have actual linked commits merged into main.

## Your handoff and assessment practice

First send the foundation revision, install instructions and runner contract to all five members. Later send prediction and benchmark outputs to Utkarsh, review his interpretation, then send the final notebook and reproduction commands to the whole team. Completion is an integrated, reproducible AML comparison, not merely a merged code tree.

Be able to explain why the test set is locked, why macro F1 matters, why timing on six laptops is unfair, why model-size measures differ, and how one PR merge preserves another person's authorship. If an interface changes, update the six relevant instructions together before asking teammates to continue.

## Commit your work and open a pull request

Make a commit when a meaningful piece works and you can explain it. Suggested milestones for your assignment are:

1. `chore: establish reproducible AML project foundation`
2. `feat: add shared experiment and evaluation workflow`
3. `feat: integrate frozen AML comparisons and benchmarks`
4. `docs: assemble final AML notebook and contribution evidence`

Commit frequency follows actual work; these examples are not a quota. Before committing, run the checks in your verification section, save the relevant outputs and inspect the diff. Stage owned paths explicitly. The following staging command is an example: use only the listed paths that you have actually created, and add your other completed owned files as needed.

```bash
git status --short
git diff
git add pyproject.toml
git add requirements.txt
git add .gitignore
git diff --cached
git commit -m "chore: establish reproducible AML project foundation"
git log -1 --format=fuller
git push -u origin feat/anant-project-foundation
```

For later commits on the same branch, `git push` is sufficient. Open https://github.com/anantjain-afk/Image-Tagging, select Compare and pull request, choose base main and your task branch, then request review from Utkarsh Jain (UJ474). If GitHub asks for credentials, use your own configured authentication; never send credentials in the PR.

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

Review the work assigned to you: Kartik Mehra (KartikMehra22). Check that their owned code follows the common interface and that evidence supports their claims. You may review an upstream PR before your own module is ready. Your reviewer does not need their model trained to inspect your implementation, so the review ring does not block the foundation.

Anant coordinates merges after review. Use **Create a merge commit** to preserve the original individual commits. After merging, check the repository's main-branch commit page and confirm that your author name links to `anantjain-afk`. If attribution is missing, check the author email against GitHub Settings > Emails. Profile contribution graphs can take up to 24 hours to refresh; the actual commit page is the first evidence to check. A PR link and author-linked commit link are stronger evidence than a screenshot of local Git alone.

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
