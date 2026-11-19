"""
Phase 2 Report Figures — pure matplotlib/numpy (no seaborn required).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import cm

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def row_normalize(mat):
    s = mat.sum(axis=1, keepdims=True)
    return mat / s

def draw_heatmap(ax, data, row_labels, col_labels, cmap='Blues', vmin=0, vmax=1,
                 fmt='.2f', fontsize=7, cbar_kw=None):
    """Pure-matplotlib annotated heatmap."""
    if cbar_kw is None:
        cbar_kw = {}
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = matplotlib.colormaps[cmap]

    im = ax.imshow(data, cmap=cmap_obj, norm=norm, aspect='auto')

    # grid lines
    n = data.shape[0]
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='lightgrey', linewidth=0.4)
    ax.tick_params(which='minor', bottom=False, left=False)

    # major ticks / labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=fontsize)
    ax.set_yticklabels(row_labels, fontsize=fontsize)

    # annotations
    thresh = (vmax + vmin) / 2.0
    for i in range(n):
        for j in range(n):
            val = data[i, j]
            color = 'white' if cmap_obj(norm(val))[0] < 0.4 else 'black'
            # for Blues: dark cell → white text
            rgba = cmap_obj(norm(val))
            lum = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
            text_color = 'white' if lum < 0.45 else 'black'
            ax.text(j, i, format(val, fmt),
                    ha='center', va='center',
                    fontsize=fontsize - 0.5,
                    color=text_color)

    # colorbar
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.8, **cbar_kw)
    cb.ax.tick_params(labelsize=7)
    return im

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 1
# ──────────────────────────────────────────────────────────────────────────────

class_labels = [
    "T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
]

rf_cm = np.array([
    [859,0,12,30,2,1,86,0,10,0],
    [2,961,2,24,4,0,6,0,1,0],
    [11,0,789,11,130,0,54,0,5,0],
    [21,2,8,905,31,0,31,0,2,0],
    [1,1,93,34,811,0,58,0,2,0],
    [0,0,0,1,0,956,0,31,1,11],
    [148,1,120,30,90,0,593,0,18,0],
    [0,0,0,0,0,11,0,952,0,37],
    [0,2,5,2,6,2,5,5,973,0],
    [0,0,0,0,0,10,1,43,2,944],
], dtype=float)

cnn_cm = np.array([
    [873,2,16,25,2,1,76,0,5,0],
    [0,993,0,5,1,0,0,0,1,0],
    [14,1,903,11,29,0,42,0,0,0],
    [12,4,7,940,13,0,23,0,1,0],
    [1,1,49,22,887,0,40,0,0,0],
    [0,0,0,0,0,970,0,23,0,7],
    [105,3,61,25,70,0,732,0,4,0],
    [0,0,0,0,0,7,0,975,0,18],
    [2,1,1,2,1,1,3,2,987,0],
    [0,0,0,0,0,5,0,31,0,964],
], dtype=float)

regnet_cm = np.array([
    [864,1,21,26,2,1,80,0,5,0],
    [1,977,2,13,3,0,2,0,2,0],
    [25,0,885,12,37,0,40,0,1,0],
    [13,10,11,922,16,1,25,0,2,0],
    [2,0,47,28,864,0,58,0,1,0],
    [0,0,0,0,0,969,0,27,0,4],
    [109,1,45,31,72,0,734,0,8,0],
    [0,0,0,0,0,4,0,979,0,17],
    [1,0,1,2,1,2,1,1,991,0],
    [0,0,0,0,0,5,1,33,0,961],
], dtype=float)

matrices = [row_normalize(rf_cm), row_normalize(cnn_cm), row_normalize(regnet_cm)]
titles = [
    "RF Pixels (Phase 1 — AML Baseline)\nMacro F1: 87.31%",
    "DeepCNN (Phase 2 — From Scratch)\nMacro F1: 92.90%",
    "RegNet-Y-400MF (Phase 2 — Transfer Learning)\nMacro F1: 91.42%",
]

fig1, axes = plt.subplots(1, 3, figsize=(22, 7))

for ax, mat, title in zip(axes, matrices, titles):
    draw_heatmap(ax, mat, class_labels, class_labels,
                 cmap='Blues', vmin=0, vmax=1, fmt='.2f', fontsize=7)
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)

    # Red rectangle around Shirt row (index 6, y-axis is inverted in imshow)
    shirt_row = 6
    rect = mpatches.FancyBboxPatch(
        (-0.5, shirt_row - 0.5), 10, 1,
        boxstyle="square,pad=0",
        linewidth=2.5, edgecolor='red', facecolor='none',
        transform=ax.transData, clip_on=False
    )
    ax.add_patch(rect)

fig1.suptitle(
    "AML vs CNN: Normalized Confusion Matrices — Fashion-MNIST Test Set (n=10,000)",
    fontsize=13, fontweight='bold', y=1.01
)
plt.tight_layout()
fig1_path = os.path.join(OUT_DIR, "aml_vs_cnn_confusion.png")
fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
plt.close(fig1)
size1 = os.path.getsize(fig1_path)
print(f"Figure 1 saved: {fig1_path}  ({size1 // 1024} KB)")
assert size1 > 10 * 1024, "Figure 1 is too small!"

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Shirt Recovery Gallery
# ──────────────────────────────────────────────────────────────────────────────

model_names = [
    "Linear SVM (Pixels)",
    "Linear SVM (HOG)",
    "LogReg (Pixels)",
    "LogReg (HOG)",
    "Random Forest (Pixels)",
    "Random Forest (HOG)",
    "k-NN (Pixels)",
    "k-NN (HOG)",
    "DeepCNN — No BatchNorm",
    "DeepCNN — No Augmentation",
    "DeepCNN — Ablation Baseline",
    "SimpleCNN (2-Block)",
    "RegNet-Y-400MF (Transfer)",
    "DeepCNN (Full Model)",
]

recalls = [50.7, 54.6, 57.1, 57.8, 59.3, 59.4, 60.2, 62.6,
           66.8, 68.1, 72.8, 70.9, 73.4, 73.2]

PHASE1_COLOR   = 'steelblue'
ABLATION_COLOR = 'darkorange'
FINAL_COLOR    = 'mediumseagreen'

colors = (
    [PHASE1_COLOR] * 8 +
    [ABLATION_COLOR] * 3 +
    [FINAL_COLOR] * 3
)

fig2, ax2 = plt.subplots(figsize=(14, 8))

y_pos = np.arange(len(model_names))
bars = ax2.barh(y_pos, recalls, color=colors, edgecolor='white', height=0.65)

for bar, val in zip(bars, recalls):
    ax2.text(
        bar.get_width() + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.1f}%",
        va='center', ha='left', fontsize=9, fontweight='bold'
    )

# Reference lines
ax2.axvline(x=80, color='red', linestyle='--', linewidth=1.8)
ax2.axvline(x=62.6, color='gray', linestyle='--', linewidth=1.5)

ax2.text(80.4, len(model_names) - 0.4,
         "Phase 2 Target (≥80%)", color='red', fontsize=8.5, va='top')
ax2.text(63.0, len(model_names) - 0.4,
         "Phase 1 Best (62.6%)", color='gray', fontsize=8.5, va='top')

ax2.set_yticks(y_pos)
ax2.set_yticklabels(model_names, fontsize=9.5)
ax2.set_xlabel("Shirt Recall (%)", fontsize=11)
ax2.set_xlim(40, 108)
ax2.set_title(
    "Shirt (Class 6) Recall: Phase 1 AML vs Phase 2 CNN Models",
    fontsize=13, fontweight='bold', pad=12
)

legend_elements = [
    mpatches.Patch(facecolor=PHASE1_COLOR,   label='Phase 1 (AML)'),
    mpatches.Patch(facecolor=ABLATION_COLOR, label='Phase 2 (Ablation)'),
    mpatches.Patch(facecolor=FINAL_COLOR,    label='Phase 2 (Final)'),
]
ax2.legend(handles=legend_elements, loc='lower right', fontsize=9.5)

ax2.grid(axis='x', linestyle=':', alpha=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
fig2_path = os.path.join(OUT_DIR, "shirt_recovery_gallery.png")
fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close(fig2)
size2 = os.path.getsize(fig2_path)
print(f"Figure 2 saved: {fig2_path}  ({size2 // 1024} KB)")
assert size2 > 10 * 1024, "Figure 2 is too small!"

print("✓ Both figures generated successfully.")
