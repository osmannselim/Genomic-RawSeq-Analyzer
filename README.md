# 🧬 Genomic-RawSeq-Analyzer

> **Deep Learning & LLM-Driven Cancer Detection from Raw Genomic Sequencing Data (FASTQ) — Without Alignment**

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)

**Boğaziçi University — CMPE 492 Senior Project**

| Role | Name | Student ID |
|------|------|------------|
| Mixed | Nuri Başar | 2021400129 |
| Mixed | Osman Selim Yüksel | 2021400105 |
| **Advisor** | Assoc. Prof. Mehmet Turan | — |

</div>

---

## 📖 Overview

Traditional cancer genomics pipelines require expensive and computationally intensive alignment steps (BWA, STAR, GATK) before any meaningful analysis can begin. This project bypasses that entire process.

We treat DNA as a **language** and apply deep learning directly to raw FASTQ sequencing reads, extracting cancer signals without ever aligning to a reference genome. The pipeline ingests raw `.fastq.gz` files from public repositories (SRA/GEO), encodes nucleotide sequences numerically, and feeds them into neural networks for tumor vs. normal classification.

The long-term goal is a complete **FASTQ → Deep Learning → LLM Report** pipeline that transforms raw sequencing data into structured, human-readable clinical insights.

---

## 🏗️ Pipeline Architecture

```
Raw FASTQ Files (SRA/GEO)
        │
        ▼
┌───────────────────┐
│  Data Ingestion   │  Streaming download via ENA FTP
│  & Preprocessing  │  Integer encoding (A=1, C=2, G=3, T=4, N=5)
│                   │  Fixed-length truncation/padding → 80bp
└────────┬──────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐  ┌──────────────┐
│ CNN   │  │ LSTM         │   ← Semester 1 (Baseline)
│ 1D    │  │ Autoencoder  │
└───┬───┘  └──────┬───────┘
    │              │
    ▼              ▼
┌─────────────────────────┐
│  DNABERT-2 Fine-tuning  │   ← Semester 2 (In Progress)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Patient-Level          │
│  Aggregation            │   Crowd-voting across 50k+ reads
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  LLAMA-3 Report         │   ← Semester 2 (Planned)
│  Generation             │   Auto-generated clinical summaries
└─────────────────────────┘
```

---

## 📊 Results (Semester 1 — Baseline)

| Model | Approach | AUC-ROC | Notes |
|-------|----------|---------|-------|
| **1D-CNN** | Supervised | **0.615** | Successful weak learner signal |
| **LSTM Autoencoder** | Unsupervised | 0.509 | Failed — cancer is not a statistical anomaly |

**Key Finding:** Cancer cannot be detected as a reconstruction anomaly in raw reads. Tumor and normal genomes are >99.9% identical at the sequence level, making supervised learning with explicit labels essential.

**Dataset:** 1.8M reads — 61.1% Tumor (~1.1M) / 38.9% Normal (~0.7M), Whole Exome Sequencing (WXS), Homo sapiens from SRA/GEO.

---

## 📁 Repository Structure

```
Genomic-RawSeq-Analyzer/
│
├── notebooks/                          # Colab notebooks (Semester 1)
│   ├── fastq_anomaly_detection_baseline_CNN.ipynb
│   └── fastq_anomaly_detection_ae.ipynb
│
├── src/                                # Modular Python scripts (Semester 2)
│   ├── data_loader.py                  # FASTQ streaming, encoding, batch processing
│   ├── models.py                       # CNN, Autoencoder, DNABERT-2 definitions
│   ├── train.py                        # Training pipeline (CNN / Autoencoder)
│   ├── evaluate.py                     # AUC metrics, patient-level evaluation
│   ├── patient_aggregation.py          # Crowd-voting aggregation pipeline (NEW)
│   └── explainability.py               # Occlusion sensitivity analysis
│
├── results/                            # Saved outputs
│   ├── roc_curves/
│   ├── saliency_maps/
│   └── reports/                        # LLM-generated clinical reports
│
├── requirements.txt
└── README.md
```

> `train.py` automatically runs both read-level and patient-level evaluation when `run_ids` are present in the batch data.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Google Colab Pro (recommended for GPU access) or local NVIDIA GPU (T4/A100)
- Google Drive (~100GB free space for FASTQ batches)

### Installation

```bash
# Clone the repository
git clone https://github.com/osmannselim/Genomic-RawSeq-Analyzer.git
cd Genomic-RawSeq-Analyzer

# Install dependencies
pip install -r requirements.txt
```

### Quick Start (Colab)

1. Open the notebook in Google Colab:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1x6uoOA_ERnFuZJUICg-SP5k7MDVwk-xm?usp=sharing)

2. Mount your Google Drive when prompted (for persistent batch storage).

3. Run cells sequentially — the pipeline automatically checks for existing downloads before re-fetching.

### Running on Your Own FASTQ Data

```python
# 1. Upload your .fastq.gz files to Google Drive
# 2. Update the base directory in the notebook
BASE_DIR = "/content/drive/MyDrive/your-fastq-folder/"

# 3. Run the Inference Only section
# Output: CSV with Cancer Probability Score per sample
```

---

## 🔬 Model Details

### Supervised: 1D-Convolutional Neural Network

Designed to scan DNA sequences for cancer-specific nucleotide motifs (mutation signatures), similar to how image CNNs detect edges.

```
Input (80bp integer-encoded) 
  → Embedding (6 tokens → 16-dim vectors)
  → Conv1D (64 filters, kernel=5, ReLU)   # detects 5-mer motifs
  → MaxPooling1D
  → Conv1D (128 filters, kernel=3, ReLU)  # higher-level features
  → GlobalMaxPooling1D
  → Dense (64, ReLU) + Dropout (0.5)
  → Output (Sigmoid) → Cancer Probability [0,1]
```

**Training:** Binary Cross-Entropy loss, Adam optimizer, early stopping (patience=3).

### Unsupervised: LSTM Autoencoder

Trained exclusively on Normal samples to learn the "grammar" of healthy DNA. Cancer sequences should yield higher reconstruction error.

```
Input (80bp) 
  → LSTM Encoder (64 → 32 units, tanh)   # compressed latent representation
  → RepeatVector
  → LSTM Decoder (32 → 64 units, tanh)
  → TimeDistributed Dense
  → Reconstruction → MSE as Anomaly Score
```

**Result:** Failed to distinguish cancer (AUC ≈ 0.50) — see [Section 8.3 of the final report](#) for analysis.

### Patient-Level Aggregation (Crowd Voting)

Individual reads are weak predictors. We aggregate across thousands of reads per patient:

```
Patient Score = mean(P(Cancer | Read_i)) for all reads i in patient sample

If Patient Score > threshold → Cancer
```

This transforms a weak per-read AUC of 0.615 into a strong patient-level diagnosis.

**Standalone usage:**

```bash
# Run patient-level aggregation with saved model and batches
python src/patient_aggregation.py \
    --model_path results/cnn_baseline.keras \
    --batch_dir data/batches/ \
    --threshold 0.5 \
    --save_dir results/patient_aggregation/
```

**Programmatic usage:**

```python
from patient_aggregation import CrowdVotingAggregator

agg = CrowdVotingAggregator(probs, y_test, run_ids_test)

# Full report: per-patient table, box plot, ROC, threshold sweep
result = agg.full_report(read_auc=0.615, save_dir="results/")

# Or step by step:
result = agg.run(threshold=0.5)         # metrics dict
agg.comparison_table(read_auc=0.615)    # read vs patient AUC
agg.plot_boxplot()                       # Cancer vs Normal distributions
agg.plot_threshold_sweep()               # find optimal threshold
```

> **Note:** Batch files must include `run_ids` (re-run `data_loader.py` with the updated version if using legacy batches).

---

## 🗓️ Semester 2 Roadmap

| Issue | Task | Assignee | Status |
|-------|------|----------|--------|
| [#1](https://github.com/osmannselim/Genomic-RawSeq-Analyzer/issues/1) | Refactor notebooks → modular Python scripts | Both | ✅ Done |
| [#2](https://github.com/osmannselim/Genomic-RawSeq-Analyzer/issues/2) | Patient-level aggregation pipeline | Osman | ✅ Done |
| [#3](https://github.com/osmannselim/Genomic-RawSeq-Analyzer/issues/3) | DNABERT-2 integration | Nuri | 📋 Planned |
| [#4](https://github.com/osmannselim/Genomic-RawSeq-Analyzer/issues/4) | DNABERT-2 vs CNN benchmark | Both | 📋 Planned |
| [#5](https://github.com/osmannselim/Genomic-RawSeq-Analyzer/issues/5) | LLAMA-3 report generation module | Nuri | 📋 Planned |
| [#6](https://github.com/osmannselim/Genomic-RawSeq-Analyzer/issues/6) | Multi-cancer dataset expansion (BRCA, LUAD) | Osman | 📋 Planned |
| [#7](https://github.com/osmannselim/Genomic-RawSeq-Analyzer/issues/7) | Aggregate occlusion sensitivity analysis | Nuri | 📋 Planned |
| [#8](https://github.com/osmannselim/Genomic-RawSeq-Analyzer/issues/8) | Final report & presentation | Both | 📋 Planned |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10 |
| **Deep Learning** | TensorFlow 2.15 / Keras |
| **Transformers** | HuggingFace Transformers (DNABERT-2, LLAMA-3) |
| **Bioinformatics** | BioPython 1.81 |
| **Data** | NumPy 1.26, Pandas 2.1 |
| **Evaluation** | Scikit-learn, Matplotlib |
| **Data Sources** | NCBI SRA, GEO, ENA FTP mirrors |
| **Compute** | Google Colab Pro (NVIDIA T4 / A100) |
| **Storage** | Google Drive (~500GB for processed batches) |

---

## 📚 Key References

- **DNABERT-2** — Zhou et al., NeurIPS 2023 — [GitHub](https://github.com/Zhihan1996/DNABERT_2)
- **DNABERT** — Ji et al., Bioinformatics 2021 — [GitHub](https://github.com/jerryji1993/DNABERT)
- **DeepSEA** — Zhou & Troyanskaya, Nature Methods 2015
- **DeepVariant** — Poplin et al., Nature Biotechnology 2018
- **XGBoost** — Chen & Guestrin, KDD 2016

---

## ⚠️ Ethical Considerations

This project uses **only publicly available, pre-anonymized datasets** from NCBI SRA and GEO. No personal health information (PHI) or clinical identifiers are used at any stage. All model outputs are strictly for **research purposes** and do not constitute clinical diagnostic recommendations. See Section 1.2 of the project report for full ethical guidelines.

---

<div align="center">
  <sub>Boğaziçi University, Department of Computer Engineering — CMPE 492, 2025–2026</sub>
</div>
