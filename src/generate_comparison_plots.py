"""
generate_comparison_plots.py
----------------------------
Produces the CNN vs DNABERT-2 comparison figures and benchmark table (Semester 2).

DNABERT-2 was trained and evaluated on Google Colab (GPU A100). Its read-level AUC
and patient-level per-batch scores are stored as constants below. The read-level ROC
curve is reconstructed analytically from the AUC value using the equal-variance normal
model (standard practice when raw probabilities are not stored locally).

CNN is evaluated locally from the saved checkpoint on the same 80/20 stratified split
that was used during training (random_state=42).

Outputs (saved to results/comparison/):
  roc_comparison.png   — 2-panel side-by-side ROC curve (Figure 8.5)
  benchmark_table.csv  — Tables 8.2 / 8.3 / 8.4 data
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

# ── Resolve project root regardless of cwd ────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

BATCH_DIR  = os.path.join(_ROOT, "results", "batches")
MODEL_PATH = os.path.join(_ROOT, "results", "cnn_baseline.keras")
OUT_DIR    = os.path.join(_ROOT, "results", "comparison")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Pre-computed DNABERT-2 results (Colab evaluation) ─────────────────
# Per-batch aggregated probabilities from EvalDNABERT2.ipynb.
# Validation set: 10 batches (4 Normal, 6 Tumor), ~40,000 reads per batch.
# Best checkpoint: Epoch 3 (lowest validation loss = 0.6432).

DB2_NORMAL_SCORES = np.array([0.581810, 0.580134, 0.575185, 0.543895])
DB2_TUMOR_SCORES  = np.array([0.635969, 0.638958, 0.638554, 0.634829, 0.631615, 0.632352])
DB2_OPTIMAL_THR   = 0.6316

DB2_READ_AUC  = 0.6240
DB2_PRECISION = 0.6381
DB2_RECALL    = 0.9517
DB2_F1        = 0.7639

DB2_TRAIN_MIN    = 32.0    # minutes per epoch
DB2_INFER_MS     = 205.9   # ms per batch of 32 sequences
DB2_GPU_MB       = 5816    # peak GPU memory (MB)
DB2_PARAMS       = 117_000_000

CNN_TRAIN_MIN = 5.0        # minutes per epoch (from training logs)
CNN_GPU_MB    = 2000       # <2 GB peak
CNN_PARAMS    = 100_000    # ~100K parameters


# ── STEP 1: Load data ─────────────────────────────────────────────────
print("=" * 60)
print("LOADING DATA")
print("=" * 60)
from data_loader import DataLoader
X, y, run_ids = DataLoader.load_all_batches(BATCH_DIR)

indices = np.arange(len(X))
_, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=y
)
X_test      = X[test_idx]
y_test      = y[test_idx]
run_ids_test = run_ids[test_idx]
print(f"Test set: {len(X_test):,} reads  "
      f"(tumor={int(y_test.sum()):,}, normal={int((y_test==0).sum()):,})")


# ── STEP 2: CNN evaluation ────────────────────────────────────────────
print("\n" + "=" * 60)
print("CNN EVALUATION")
print("=" * 60)

from tensorflow.keras.models import load_model
cnn_model = load_model(MODEL_PATH)

print("Running inference...")
t0 = time.perf_counter()
probs_cnn = cnn_model.predict(X_test, batch_size=2048, verbose=1).flatten()
elapsed = time.perf_counter() - t0
n_batches_cnn = len(X_test) // 2048
cnn_ms_per_batch = (elapsed / n_batches_cnn) * 1000 if n_batches_cnn > 0 else 0.0

# Read-level metrics
fpr_cnn_read, tpr_cnn_read, _ = roc_curve(y_test, probs_cnn)
auc_cnn_read = auc(fpr_cnn_read, tpr_cnn_read)
prec_cnn, rec_cnn, f1_cnn, _ = precision_recall_fscore_support(
    y_test, (probs_cnn >= 0.5).astype(int), average='binary', zero_division=0
)

# Patient-level crowd-voting
df_cnn = pd.DataFrame({"run_id": run_ids_test, "prob": probs_cnn, "label": y_test})
pat_cnn = df_cnn.groupby("run_id").agg(
    patient_prob=("prob", "mean"),
    patient_label=("label", lambda x: int(x.mode()[0]))
).reset_index()
fpr_cnn_pat, tpr_cnn_pat, _ = roc_curve(pat_cnn["patient_label"], pat_cnn["patient_prob"])
auc_cnn_pat = auc(fpr_cnn_pat, tpr_cnn_pat)

print(f"Read-level  AUC : {auc_cnn_read:.4f}")
print(f"Patient-level AUC: {auc_cnn_pat:.4f}")
print(f"Inference       : {cnn_ms_per_batch:.1f} ms/batch (batch_size=2048)")


# ── STEP 3: DNABERT-2 patient-level ROC ───────────────────────────────
labels_db  = np.concatenate([np.zeros(len(DB2_NORMAL_SCORES)),
                              np.ones(len(DB2_TUMOR_SCORES))])
scores_db  = np.concatenate([DB2_NORMAL_SCORES, DB2_TUMOR_SCORES])
fpr_db_pat, tpr_db_pat, _ = roc_curve(labels_db, scores_db)
auc_db_pat = auc(fpr_db_pat, tpr_db_pat)

# Read-level ROC reconstructed from AUC via equal-variance normal model
d_prime     = np.sqrt(2.0) * norm.ppf(DB2_READ_AUC)
fpr_db_read = np.linspace(0, 1, 500)
tpr_db_read = norm.cdf(norm.ppf(np.clip(fpr_db_read, 1e-7, 1 - 1e-7)) + d_prime)
tpr_db_read[0], tpr_db_read[-1] = 0.0, 1.0

print(f"\nDNABERT-2 (pre-computed from Colab):")
print(f"Read-level  AUC : {DB2_READ_AUC:.4f}")
print(f"Patient-level AUC: {auc_db_pat:.4f}")


# ── STEP 4: Side-by-side ROC plot ─────────────────────────────────────
print("\n" + "=" * 60)
print("GENERATING PLOTS")
print("=" * 60)

STYLE = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
plt.style.use(STYLE)

COLOR_CNN = "#34495e"   # dark slate
COLOR_DB  = "#e74c3c"   # coral red
COLOR_RND = "#95a5a6"   # grey dashed

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=200)

# --- Left: Read-Level ROC ---
ax = axes[0]
ax.plot(fpr_cnn_read, tpr_cnn_read,
        color=COLOR_CNN, lw=2.5, label=f"1D-CNN Baseline  (AUC = {auc_cnn_read:.4f})")
ax.plot(fpr_db_read, tpr_db_read,
        color=COLOR_DB,  lw=2.5, label=f"DNABERT-2  (AUC = {DB2_READ_AUC:.4f})")
ax.plot([0, 1], [0, 1], color=COLOR_RND, lw=1.2, linestyle="--", label="Random Chance")
ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.01])
ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold", labelpad=8)
ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold", labelpad=8)
ax.set_title("Read-Level ROC Curve Comparison", fontsize=14, fontweight="bold", pad=12)
ax.legend(loc="lower right", frameon=True, fontsize=10, facecolor="#fdfefe", edgecolor="#bdc3c7")
ax.grid(True, linestyle=":", alpha=0.6)

# --- Right: Patient-Level ROC ---
ax = axes[1]
ax.plot(fpr_cnn_pat, tpr_cnn_pat,
        color=COLOR_CNN, lw=2.5, label=f"1D-CNN  (AUC = {auc_cnn_pat:.4f})")
ax.plot(fpr_db_pat, tpr_db_pat,
        color=COLOR_DB,  lw=2.5, label=f"DNABERT-2  (AUC = {auc_db_pat:.4f})")
ax.plot([0, 1], [0, 1], color=COLOR_RND, lw=1.2, linestyle="--", label="Random Chance")
ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.01])
ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold", labelpad=8)
ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold", labelpad=8)
ax.set_title("Patient-Level ROC Comparison\n(Crowd-Voting Aggregation)",
             fontsize=14, fontweight="bold", pad=12)
ax.legend(loc="lower right", frameon=True, fontsize=10, facecolor="#fdfefe", edgecolor="#bdc3c7")
ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
roc_path = os.path.join(OUT_DIR, "roc_comparison.png")
fig.savefig(roc_path, dpi=300, bbox_inches="tight")
print(f"Saved: {roc_path}")
plt.close(fig)


# ── STEP 5: Benchmark tables ──────────────────────────────────────────
print("\n" + "=" * 70)
print("TABLE 8.2 — READ-LEVEL PERFORMANCE COMPARISON")
print("=" * 70)
read_table = pd.DataFrame({
    "Model":     ["1D-CNN (Baseline)", "DNABERT-2 (fine-tuned)"],
    "AUC-ROC":   [round(auc_cnn_read, 4), DB2_READ_AUC],
    "Precision": [round(prec_cnn, 4),     DB2_PRECISION],
    "Recall":    [round(rec_cnn, 4),      DB2_RECALL],
    "F1 Score":  [round(f1_cnn, 4),       DB2_F1],
}).set_index("Model")
print(read_table.to_string())

print("\n" + "=" * 70)
print("TABLE 8.3 — PATIENT-LEVEL PERFORMANCE (CROWD-VOTING)")
print("=" * 70)

# CNN patient-level at threshold 0.5
pat_cnn["pred"] = (pat_cnn["patient_prob"] >= 0.5).astype(int)
prec_cnn_pat, rec_cnn_pat, f1_cnn_pat, _ = precision_recall_fscore_support(
    pat_cnn["patient_label"], pat_cnn["pred"], average='binary', zero_division=0
)
pat_table = pd.DataFrame({
    "Model":     ["1D-CNN (Baseline)", "DNABERT-2 (fine-tuned)"],
    "AUC-ROC":   [round(auc_cnn_pat, 4), round(auc_db_pat, 4)],
    "Precision": [round(prec_cnn_pat, 4), 1.0],
    "Recall":    [round(rec_cnn_pat, 4),  1.0],
    "F1 Score":  [round(f1_cnn_pat, 4),   1.0],
    "Opt. Threshold": ["0.50", f"{DB2_OPTIMAL_THR:.4f}"],
}).set_index("Model")
print(pat_table.to_string())

print("\n" + "=" * 70)
print("TABLE 8.4 — COMPUTATIONAL PERFORMANCE COMPARISON")
print("=" * 70)
comp_table = pd.DataFrame({
    "Model":                ["1D-CNN (Baseline)", "DNABERT-2 (fine-tuned)"],
    "Train Time/Epoch":     [f"~{CNN_TRAIN_MIN:.0f} min", f"~{DB2_TRAIN_MIN:.0f} min"],
    "Inference (ms/batch)": [f"{cnn_ms_per_batch:.1f}", f"{DB2_INFER_MS:.1f}"],
    "GPU Peak Memory":      [f"<{CNN_GPU_MB//1000} GB", f"{DB2_GPU_MB:,} MB"],
    "Parameters":           [f"~{CNN_PARAMS//1000}K", f"{DB2_PARAMS//1_000_000}M"],
}).set_index("Model")
print(comp_table.to_string())

# Save CSV
csv_path = os.path.join(OUT_DIR, "benchmark_table.csv")
full_table = pd.DataFrame({
    "Model":                   ["1D-CNN (Baseline)", "DNABERT-2 (fine-tuned)"],
    "Read AUC":                [round(auc_cnn_read, 4), DB2_READ_AUC],
    "Read Precision":          [round(prec_cnn, 4),     DB2_PRECISION],
    "Read Recall":             [round(rec_cnn, 4),      DB2_RECALL],
    "Read F1":                 [round(f1_cnn, 4),       DB2_F1],
    "Patient AUC":             [round(auc_cnn_pat, 4),  round(auc_db_pat, 4)],
    "Patient Precision":       [round(prec_cnn_pat, 4), 1.0],
    "Patient Recall":          [round(rec_cnn_pat, 4),  1.0],
    "Patient F1":              [round(f1_cnn_pat, 4),   1.0],
    "Train Time/Epoch (min)":  [CNN_TRAIN_MIN,           DB2_TRAIN_MIN],
    "Inference (ms/batch)":    [round(cnn_ms_per_batch, 1), DB2_INFER_MS],
    "GPU Memory (MB)":         [CNN_GPU_MB,               DB2_GPU_MB],
    "Parameters":              [CNN_PARAMS,               DB2_PARAMS],
})
full_table.to_csv(csv_path, index=False)
print(f"\nBenchmark table saved: {csv_path}")
print(f"ROC comparison plot saved: {roc_path}")
