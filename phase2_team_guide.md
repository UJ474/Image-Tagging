# Phase 2 Team Work Guide: Execution Plan & Ownership

**Project:** Inventory Image Tagging via Handcrafted Features and Deep Learning (CNN & RegNet)  
**Phase:** Phase 2 — Deep Learning Architecture, Transfer Learning & Ablation  
**Repository:** `anantjain-afk/Image-Tagging`  
**Team:** T112 | Section D | 6 Members  

---

## 1. Workflow & Execution Order

To make development as simple and fast as possible while guaranteeing **zero Git merge conflicts**, work is partitioned into strictly modular, disjoint tasks. 

> [!TIP]
> **Direct Push Workflow:** Because every teammate owns completely mutually exclusive files and directories, you do **not** need complex feature branch and pull-request procedures. Everyone can work directly on `main`:
> ```bash
> git pull origin main
> git add <your-owned-files>
> git commit -m "feat(<scope>): <short description>"
> git push origin main
> ```

```mermaid
flowchart TD
    subgraph Milestone 1 ["Milestone 2.1: Foundation Pipeline"]
        A["<b>Anant Jain</b><br/>PyTorch Data Pipeline<br/>(dl_data.py)"]
        K["<b>Kartik Mehra</b><br/>Training Loop & Early Stopping<br/>(dl_training.py)"]
        V1["<b>Vipul Sharma</b><br/>CNN Model Architectures & Tests<br/>(cnn.py, test_cnn.py)"]
    end

    Milestone 1 -->|Push to main| Milestone 2

    subgraph Milestone 2 ["Milestone 2.2: Parallel Experiments & Deliverables"]
        KR["<b>Krishna Gehlot</b><br/>Main CNN Benchmark & Curves<br/>(07_cnn_experiments.ipynb)"]
        V2["<b>Vipul Sharma</b><br/>Controlled Ablation Study<br/>(08_ablation_study.ipynb)"]
        AD["<b>Aditya</b><br/>RegNet-Y-400MF Fine-Tuning<br/>(regnet.py, 09_regnet_transfer.ipynb)"]
    end

    Milestone 2 -->|Push to main| Milestone 3

    subgraph Milestone 3 ["Milestone 2.3: Synthesis, Reporting & Presentation"]
        U["<b>Utkarsh Jain</b><br/>AML vs CNN Error Analysis & IEEE Report<br/>(phase2_error_analysis.md, report/main.tex)"]
        V3["<b>Vipul Sharma</b><br/>Phase 2 Presentation Deck<br/>(phase2_presentation.pptx)"]
    end
```

---

## 2. Team Member Ownership Matrix

Every team member owns distinct, non-overlapping files:

| Order | Member | GitHub Handle | Phase 2 Responsibilities | Owned Files |
|:---:|:---|:---|:---|:---|
| **1st** | **Anant Jain** | `@anantjain-afk` | PyTorch Data Pipeline, Loaders & Foundation Setup | `requirements.txt`, `src/aml/dl_data.py` |
| **1st** | **Kartik Mehra** | `@KartikMehra22` | Training Loop, Optimization & Early Stopping | `src/aml/dl_training.py` |
| **2nd** | **Vipul Sharma** | `@vs-code06` | CNN Architectures, Unit Tests, Ablation Study & Presentation | `src/aml/models/cnn.py`, `tests/test_cnn.py`, `notebooks/08_ablation_study.ipynb`, `reports/results/ablation_study.json`, `reports/phase2_presentation.pptx` |
| **3rd** | **Krishna Gehlot** | `@NSTKrishna` | SimpleCNN vs DeepCNN Training & Benchmark | `notebooks/07_cnn_experiments.ipynb`, `reports/results/cnn_simple.json`, `reports/results/cnn_deep.json`, `reports/figures/cnn/training_curves_*.png` |
| **3rd** | **Aditya** | `@AdityaInnovates` | Pretrained RegNet-Y-400MF Transfer Learning | `src/aml/models/regnet.py`, `notebooks/09_regnet_transfer_learning.ipynb`, `reports/results/regnet_finetuned.json`, `reports/figures/cnn/regnet_*.png` |
| **4th** | **Utkarsh Jain** | `@UJ474` | AML vs CNN Error Analysis & IEEE LaTeX Report | `reports/phase2_error_analysis.md`, `reports/figures/cnn/aml_vs_cnn_*.png`, `report/*` |

---

## 3. Individual Work Guides

---

### Member 1: Anant Jain (`@anantjain-afk`)
**Role:** Data Pipeline, Preprocessing & Foundation  
**Owned Files:**
- `requirements.txt` (Adds PyTorch dependencies)
- `src/aml/dl_data.py` (PyTorch Dataset and DataLoader wrappers)

#### Tasks:
1. Update `requirements.txt` with `torch>=2.0.0`, `torchvision>=0.15.0`, and `torchinfo>=1.8.0`.
2. Implement `FashionMNISTDataset(torch.utils.data.Dataset)` in `src/aml/dl_data.py` wrapping normalized arrays from `get_train_val_test_splits()`.
3. Add single-channel dimension: `(N, 1, 28, 28)`.
4. Add training data augmentation: `RandomHorizontalFlip(p=0.5)` and `RandomRotation(degrees=10)`.
5. Implement `get_dataloaders(batch_size=128)` returning train, validation, and test loaders.
6. Commit and push directly to `main`.

---

### Member 2: Kartik Mehra (`@KartikMehra22`)
**Role:** Training Engine, Optimization & Early Stopping  
**Owned Files:**
- `src/aml/dl_training.py` (Reusable training and evaluation loops)

#### Tasks:
1. Implement `EarlyStopping(patience=5, mode='max')` tracking validation Macro F1 to stop training when validation performance plateaus.
2. Implement `train_epoch(model, loader, criterion, optimizer, device)` computing cross-entropy loss and gradient backpropagation.
3. Implement `evaluate_model(model, loader, device)` computing predictions and returning accuracy, macro F1, and per-class recall using Phase 1's `compute_metrics()`.
4. Implement `fit_model(model, train_loader, val_loader, epochs=30, lr=1e-3, ...)` with `ReduceLROnPlateau` and saving best model weights.
5. Commit and push directly to `main`.

---

### Member 3: Vipul Sharma (`@vs-code06`)
**Role:** CNN Model Architecture, Unit Testing, Ablation Study & Presentation Deck  
**Owned Files:**
- `src/aml/models/cnn.py` (SimpleCNN and DeepCNN model architectures)
- `tests/test_cnn.py` (Unit tests verifying forward pass, output shapes, and gradient flow)
- `notebooks/08_ablation_study.ipynb` (5-variable controlled ablation notebook)
- `reports/results/ablation_study.json` (Ablation metric records)
- `reports/phase2_presentation.pptx` (Phase 2 presentation deck)

#### Tasks:
1. **Model Architectures (`src/aml/models/cnn.py`):**
   - **`SimpleCNN` (2-Block CNN):**
     - Block 1: `Conv2d(1, 32, 3, padding=1)` $\rightarrow$ `BatchNorm2d(32)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2)` ($14 \times 14$) $\rightarrow$ `Dropout2d(0.25)`.
     - Block 2: `Conv2d(32, 64, 3, padding=1)` $\rightarrow$ `BatchNorm2d(64)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2)` ($7 \times 7$) $\rightarrow$ `Dropout2d(0.25)`.
     - Head: `Flatten()` $\rightarrow$ `Linear(64*7*7, 128)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Linear(128, 10)`.
   - **`DeepCNN` (3-Block CNN with configurable ablation toggles):**
     - Block 3: `Conv2d(64, 128, 3, padding=1)` $\rightarrow$ `BatchNorm2d(128)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2)` ($3 \times 3$) $\rightarrow$ `Dropout2d(0.25)`.
     - Head: `Flatten()` $\rightarrow$ `Linear(128*3*3, 256)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Linear(256, 10)`.
     - Add parameters: `use_batchnorm=True`, `conv_dropout=0.25`, `fc_dropout=0.5`.
2. **Unit Tests (`tests/test_cnn.py`):**
   - Verify input shape `(B, 1, 28, 28)` produces output logits `(B, 10)`.
   - Verify backpropagation gradient updates and parameter counts (~410K for SimpleCNN, ~440K for DeepCNN).
3. **Controlled Ablation Study (`notebooks/08_ablation_study.ipynb`):**
   - Run 5 systematic ablation settings based on `DeepCNN`:
     1. Baseline (Full `DeepCNN` with BatchNorm, Dropout, Augmentation, Early Stopping)
     2. No Dropout (`conv_dropout=0.0`, `fc_dropout=0.0` to examine overfitting)
     3. No BatchNorm (examine training stability and convergence rate)
     4. No Data Augmentation (examine impact of spatial transformations)
     5. 2-Block vs. 3-Block Depth (isolate the benefit of the third convolutional layer on Shirt recall)
   - Save comparison table and export `reports/results/ablation_study.json`.
4. **Phase 2 Presentation (`reports/phase2_presentation.pptx`):**
   - Build the slide deck covering Phase 2 motivation, network diagrams, training dynamics, AML vs. CNN benchmark comparisons, ablation findings, and error resolution.
5. Commit and push directly to `main`.

---

### Member 4: Krishna Gehlot (`@NSTKrishna`)
**Role:** Main CNN Training, Benchmarking & Confusion Matrices  
**Owned Files:**
- `notebooks/07_cnn_experiments.ipynb`
- `reports/results/cnn_simple.json`
- `reports/results/cnn_deep.json`
- `reports/figures/cnn/training_curves_simple.png`
- `reports/figures/cnn/training_curves_deep.png`
- `reports/figures/cnn/confusion_matrix_deep.png`

#### Tasks:
1. In `notebooks/07_cnn_experiments.ipynb`, import `SimpleCNN` and `DeepCNN` from `src/aml/models/cnn.py`.
2. Train both models using `fit_model()` from `src/aml/dl_training.py` with early stopping.
3. Plot training vs. validation loss and Macro F1 curves across epochs.
4. Evaluate both trained models on the held-out 10k test set.
5. Generate the normalized $10 \times 10$ test confusion matrix for `DeepCNN`.
6. Export standardized metrics to `reports/results/cnn_simple.json` and `reports/results/cnn_deep.json`.
7. Commit and push directly to `main`.

---

### Member 5: Aditya (`@AdityaInnovates`)
**Role:** Hybrid Innovation: Pretrained RegNet Transfer Learning  
**Owned Files:**
- `src/aml/models/regnet.py`
- `notebooks/09_regnet_transfer_learning.ipynb`
- `reports/results/regnet_finetuned.json`
- `reports/figures/cnn/regnet_finetuning_curves.png`

#### Tasks:
1. In `src/aml/models/regnet.py`, adapt `torchvision.models.regnet_y_400mf`:
   - Adapt stem convolution from 3 channels to 1 channel (averaging pretrained RGB weights).
   - Replace linear classifier head with `Linear(440, 10)`.
2. In `notebooks/09_regnet_transfer_learning.ipynb`:
   - Stage 1: Freeze backbone, train classification head for 3 epochs.
   - Stage 2: Unfreeze entire network, fine-tune at lower learning rate ($lr = 10^{-4}$) with early stopping.
3. Evaluate on held-out test set and export `reports/results/regnet_finetuned.json`.
4. Commit and push directly to `main`.

---

### Member 6: Utkarsh Jain (`@UJ474`)
**Role:** AML vs. CNN Error Analysis & IEEE LaTeX Report Draft  
**Owned Files:**
- `reports/phase2_error_analysis.md`
- `reports/figures/cnn/aml_vs_cnn_confusion.png`
- `reports/figures/cnn/shirt_recovery_gallery.png`
- `report/main.tex` (IEEE Format Report)
- `report/references.bib`

#### Tasks:
1. Compare test predictions between Phase 1 best models (Random Forest, SVM) and Phase 2 CNN models.
2. In `reports/phase2_error_analysis.md`, evaluate whether the upper-body bottleneck was resolved:
   - Check if Shirt recall increased from 50.7%–62.6% to $\ge 80.0\%$.
   - Check if total upper-body shirt confusion decreased to $< 15.0\%$.
3. Extract and visualize misclassification examples:
   - Create `shirt_recovery_gallery.png` showing garments where Phase 1 failed but CNN succeeded.
   - Analyze remaining failure cases.
4. Set up the IEEE 2-column LaTeX report (`report/main.tex`) integrating Phase 1 and Phase 2 results.
5. Commit and push directly to `main`.

---

## 4. Master Deliverables Mapping

| Required Deliverable | Primary Owner(s) | Primary Location / Output |
|:---|:---|:---|
| **1. Architecture Logic** | Vipul Sharma | `src/aml/models/cnn.py`, `tests/test_cnn.py` |
| **2. Hybrid Innovation** | Aditya | `src/aml/models/regnet.py`, `notebooks/09_regnet_transfer_learning.ipynb` |
| **3. Ablation Studies** | Vipul Sharma | `notebooks/08_ablation_study.ipynb`, `reports/results/ablation_study.json` |
| **4. DL Literature Review** | Utkarsh Jain & Team | Section 3 of `report/main.tex`, `report/references.bib` |
| **5. DL Dataset & Regularization** | Anant Jain & Kartik Mehra | `src/aml/dl_data.py`, `src/aml/dl_training.py` |
| **6. Technical Validation** | Krishna Gehlot | `notebooks/07_cnn_experiments.ipynb`, test confusion matrices |
| **7. Theoretical Rigor** | Team | Documented in `reports/phase2_error_analysis.md` and report |
| **8. GitHub Code Quality** | All Members | Clean commit logs, modular `src/aml/`, unit tests |
| **9. Project Presentation** | Vipul Sharma | `reports/phase2_presentation.pptx` |
| **10. Viva Voce Preparation** | All Members | Common Q&A prep guide in `phase2_plan.md` |
| **11. Project Report** | Utkarsh Jain & Team | `report/main.tex` (IEEE Two-Column format) |

---

## 5. Collaboration & Direct Push Rules

1. **Strict File Isolation:** Only modify and commit files listed in your owned files table.
2. **Pull Before Working:** Always pull the latest `main` before starting:
   ```bash
   git pull origin main
   ```
3. **Pushing Changes:** Once your file/notebook works and passes tests, push directly to `main`:
   ```bash
   git add <your-owned-files>
   git commit -m "feat(<scope>): <short description>"
   git pull --rebase origin main
   git push origin main
   ```
4. **Reproducibility:** Fix random seed to 42 across all components (`torch.manual_seed(42)`, `np.random.seed(42)`).
