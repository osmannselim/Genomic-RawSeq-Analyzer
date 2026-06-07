# 🧬 Genomic-RawSeq-Analyzer

> **Deep Learning & LLM-Driven Cancer Detection from Raw Genomic Sequencing Data (FASTQ) — Without Alignment**

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![Status](https://img.shields.io/badge/Status-Semester%202%20Complete-brightgreen)

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

We treat DNA as a **language** and apply deep learning directly to raw FASTQ sequencing reads, extracting cancer signals without ever aligning to a reference genome. The pipeline ingests raw `.fastq.gz` files from public repositories (SRA/GEO/ENA), encodes nucleotide sequences numerically, and feeds them into neural networks for tumor vs. normal classification.

By the end of Semester 2, the full **FASTQ → Deep Learning → Explainability → LLM Report** pipeline is implemented end-to-end: raw reads are classified, model decisions are explained via occlusion/saliency analysis, and a LLAMA-3-based module turns the results into structured, human-readable clinical summary reports.

> 📂 **All project artifacts** (trained models, FASTQ batches, evaluation plots, generated reports, DNABERT-2 checkpoints) are available on Google Drive:
> **[DNA_Anomaly_Detection — Google Drive folder](https://drive.google.com/drive/folders/1dd3XQjEpEytE-6YFzRvpFFuVQE4HMBfj?usp=sharing)**
> (this repo contains code and notebooks; the Drive folder contains everything else — models, data batches, and results too large for git)

---

## 🏗️ Pipeline Architecture

```
Raw FASTQ Files (SRA / GEO / ENA mirrors)
        │
        ▼
┌───────────────────┐
│  Data Ingestion   │  Direct HTTPS streaming/range-fetch from ENA
│  & Preprocessing  │  Integer encoding (A=1, C=2, G=3, T=4, N=5)
│                   │  Fixed-length truncation/padding → 80bp
└────────┬──────────┘
         │
    ┌────┴─────────┐
    │              │
    ▼              ▼
┌───────┐  ┌──────────────┐  ┌─────────────────────────┐
│ CNN   │  │ LSTM         │  │  DNABERT-2 Fine-tuning  │
│ 1D    │  │ Autoencoder  │  │  (transformer encoder)  │  ← Semester 2
└───┬───┘  └──────┬───────┘  └────────────┬────────────┘
    │  ↑ Semester 1 (Baseline)             │
    └──────────────┬───────────────────────┘
                   ▼
┌─────────────────────────┐      ┌─────────────────────────┐
│  Patient-Level          │      │  Explainability         │
│  Aggregation            │      │  Occlusion sensitivity  │
│  Crowd-voting (50k+ rds)│      │  + saliency maps        │
└────────────┬────────────┘      │  + aggregated motifs    │
             │                   └────────────┬────────────┘
             └────────────┬───────────────────┘
                          ▼
              ┌─────────────────────────┐
              │  LLAMA-3 Report         │
              │  Generation             │   Auto-generated clinical summaries
              └─────────────────────────┘
```

**Cross-cancer generalization:** the Semester 1 CNN (trained on a single breast-cancer cohort) is also evaluated **zero-shot** on two new cohorts — BRCA and LUAD — downloaded fresh from ENA, to test whether learned features transfer across cancer types.

---

## 📊 Results (Semester 1 — Baseline)

| Model | Approach | AUC-ROC | Notes |
|-------|----------|---------|-------|
| **1D-CNN** | Supervised | **0.615** | Successful weak learner signal |
| **LSTM Autoencoder** | Unsupervised | 0.509 | Failed — cancer is not a statistical anomaly |

**Key Finding:** Cancer cannot be detected as a reconstruction anomaly in raw reads. Tumor and normal genomes are >99.9% identical at the sequence level, making supervised learning with explicit labels essential.

**Dataset:** 1.8M reads — 61.1% Tumor (~1.1M) / 38.9% Normal (~0.7M), Whole Exome Sequencing (WXS), Homo sapiens from SRA/GEO.

---

## 📊 Results (Semester 2 — Extensions)

### DNABERT-2 vs. 1D-CNN Benchmark ([`BenchmarkComparison.ipynb`](notebooks/BenchmarkComparison.ipynb))

| Model | Read AUC | Patient AUC | Train (min/epoch) | Inference (ms/batch) | GPU (MB) |
|-------|----------|-------------|-------------------|----------------------|----------|
| **1D-CNN (baseline)** | 0.6155 | 0.6155 | 5.0 | 22.6 | 2,000 |
| **DNABERT-2 (fine-tuned)** | 0.6240 | 1.0000 | 32.0 | 205.9 | 5,816 |

DNABERT-2's transformer-based encoder edges out the CNN on raw read-level AUC and dramatically improves patient-level separation, at the cost of ~9x slower inference and ~3x more GPU memory — a meaningful accuracy/efficiency trade-off discussed in the final report.

### Cross-Cancer Zero-Shot Transfer ([`MultiCancerData.ipynb`](notebooks/MultiCancerData.ipynb))

The Semester 1 breast-cancer CNN was evaluated **without retraining** on two freshly-downloaded cohorts (BRCA, LUAD — fetched directly from ENA via partial HTTPS range-requests, 50k reads/run):

| Cohort | Read AUC | Patient AUC | Interpretation |
|--------|----------|-------------|----------------|
| **LUAD** (Lung Adenocarcinoma) | 0.5053 | 0.5000 | ≈ random — different cancer type, no transfer |
| **BRCA** (Breast — same cancer type, new samples) | — | — | full plots/batches in `results/multi_cancer/brca/` (see Drive) |

**Key takeaway:** the CNN's learned features do **not** generalize zero-shot to a different cancer type (LUAD AUC ≈ 0.50, ≈ random chance), confirming that somatic mutation signatures are largely cancer-type-specific rather than universal — motivating per-cancer-type fine-tuning as future work.

### Explainability — Aggregated Occlusion Analysis ([`OcclusionAnalysis.ipynb`](notebooks/OcclusionAnalysis.ipynb))

Occlusion sensitivity was run across 100+ high-confidence cancer reads and averaged position-by-position to surface consistently important k-mer motifs (top hit: `AGGCT` at position 28, score 0.55). Outputs include a per-read saliency map gallery and an aggregated importance heatmap — see `results/occlusion/` (`motifs.json`, `aggregated_importance.png`, `occlusion_heatmap.png`, `saliency_maps/`).

### LLM Clinical Report Generation ([`LLMReports.ipynb`](notebooks/LLMReports.ipynb))

A LLAMA-3-8B-Instruct (4-bit quantized) module converts per-patient model outputs — cancer probability, top occlusion motifs, cancer type — into structured clinical-style summary reports. Sample reports for 5 test patients are saved under `results/reports/` (e.g. a 0.97 cancer-probability read is summarized as *"high risk... whole-exome sequencing data has been analyzed using an alignment-free deep learning approach..."*).

---

## 📁 Repository Structure

```
Genomic-RawSeq-Analyzer/
│
├── notebooks/
│   ├── fastq_anomaly_detection_baseline_CNN.ipynb   # Semester 1 — CNN baseline
│   ├── fastq_anomaly_detection_ae.ipynb             # Semester 1 — LSTM Autoencoder
│   ├── reverse_encoder.ipynb                        # Semester 1 — sequence decoding utilities
│   ├── TrainDNABERT_2.ipynb / dnabert2_finetuning.ipynb  # Semester 2 — DNABERT-2 fine-tuning
│   ├── EvalDNABERT2.ipynb                           # Semester 2 — DNABERT-2 evaluation
│   ├── BenchmarkComparison.ipynb                    # Semester 2 — DNABERT-2 vs CNN benchmark
│   ├── MultiCancerData.ipynb                        # Semester 2 — BRCA/LUAD zero-shot transfer
│   ├── OcclusionAnalysis.ipynb                      # Semester 2 — aggregated occlusion/saliency
│   └── LLMReports.ipynb                             # Semester 2 — LLAMA-3 clinical report generation
│
├── src/
│   ├── data_loader.py                  # FASTQ streaming, encoding, batch processing
│   ├── models.py                       # CNN, Autoencoder model definitions
│   ├── dnabert2_model.py               # DNABERT-2 wrapper / fine-tuning utilities
│   ├── train.py                        # Training pipeline (CNN / Autoencoder)
│   ├── evaluate.py                     # AUC metrics, patient-level evaluation
│   ├── patient_aggregation.py          # Crowd-voting aggregation pipeline
│   ├── explainability.py               # Occlusion sensitivity & saliency analysis
│   ├── run_occlusion.py                # Aggregated occlusion analysis runner
│   ├── multi_cancer_loader.py          # BRCA/LUAD ENA download + zero-shot eval helpers
│   ├── report_generator.py             # LLAMA-3 clinical report generation
│   └── generate_comparison_plots.py    # DNABERT-2 vs CNN benchmark plotting
│
├── results/                            # Saved outputs (plots, batches, reports — see Drive for full set)
│   ├── multi_cancer/{brca,luad}/       # Zero-shot eval batches, ROC curves, class balance
│   ├── reports/                        # LLM-generated clinical reports (.txt)
│   ├── occlusion/                      # Saliency maps, aggregated importance heatmap, motifs.json
│   └── comparison/                     # DNABERT-2 vs CNN benchmark table & ROC comparison
│
├── docs/
│   └── data_and_models.md
│
├── requirements.txt
└── README.md
```

> `train.py` automatically runs both read-level and patient-level evaluation when `run_ids` are present in the batch data.
>
> Trained model weights, raw FASTQ batches, and the full set of generated artifacts (DNABERT-2 checkpoints, LLAMA-3 weights, intermediate `.npz`/`.npy` batches) are stored on **[Google Drive](https://drive.google.com/drive/folders/1dd3XQjEpEytE-6YFzRvpFFuVQE4HMBfj?usp=sharing)** rather than in git, due to size.

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

All notebooks under [`notebooks/`](notebooks/) are designed to run standalone on Google Colab (T4 GPU runtime):

1. Upload the notebook of interest to Colab (or open directly from this repo).
2. Mount your Google Drive when prompted — all models, batches, and results persist under `MyDrive/DNA_Anomaly_Detection/` (see the [shared Drive folder](https://drive.google.com/drive/folders/1dd3XQjEpEytE-6YFzRvpFFuVQE4HMBfj?usp=sharing) for the expected layout).
3. Run cells sequentially — each notebook checks for existing downloads/batches/checkpoints before re-fetching, so re-running a session picks up where it left off.

**Suggested order to reproduce the full pipeline:**

| Step | Notebook | What it does |
|------|----------|--------------|
| 1 | `fastq_anomaly_detection_baseline_CNN.ipynb` | Train the Semester 1 CNN baseline |
| 2 | `TrainDNABERT_2.ipynb` / `dnabert2_finetuning.ipynb` | Fine-tune DNABERT-2 on the same data |
| 3 | `BenchmarkComparison.ipynb` | Compare CNN vs. DNABERT-2 (AUC, speed, memory) |
| 4 | `MultiCancerData.ipynb` | Download BRCA/LUAD cohorts from ENA, run zero-shot transfer eval |
| 5 | `OcclusionAnalysis.ipynb` | Aggregate occlusion sensitivity across reads → motif heatmaps |
| 6 | `LLMReports.ipynb` | Generate LLAMA-3 clinical summary reports from model outputs |

> **Note on data ingestion:** `MultiCancerData.ipynb` downloads FASTQ data directly from the **European Nucleotide Archive (ENA)** over plain HTTPS — using partial byte-range requests to fetch just enough compressed data to cover the configured read budget (`MAX_READS`), then parsing the (possibly truncated) gzip stream incrementally. This avoids the SRA-toolkit (`fasterq-dump`) entirely, which proved unreliable in the Colab environment (toolkit configuration, missing CLI flags, and cloud-resolver failures across multiple versions).

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

**Result:** Failed to distinguish cancer (AUC ≈ 0.50) — analyzed in detail in the final project report (see Issue [#8](https://github.com/osmannselim/Genomic-RawSeq-Analyzer/issues/8)).

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

### DNABERT-2 (Fine-tuned Transformer Encoder)

Replaces the CNN's embedding + convolutional stack with a pretrained genomic language model ([Zhou et al., 2023](https://github.com/Zhihan1996/DNABERT_2)), fine-tuned on the same tumor/normal read classification task. See `src/dnabert2_model.py` and [`TrainDNABERT_2.ipynb`](notebooks/TrainDNABERT_2.ipynb) / [`EvalDNABERT2.ipynb`](notebooks/EvalDNABERT2.ipynb). Benchmarked head-to-head against the CNN baseline in [`BenchmarkComparison.ipynb`](notebooks/BenchmarkComparison.ipynb) — see results above.

### Explainability — Occlusion Sensitivity & Saliency Maps

`src/explainability.py` and `src/run_occlusion.py` systematically occlude k-mer windows across reads and measure the resulting drop in cancer-probability score, producing per-read saliency maps. [`OcclusionAnalysis.ipynb`](notebooks/OcclusionAnalysis.ipynb) aggregates these scores **across 100+ high-confidence cancer reads** (`plot_aggregated_importance`) to surface recurring motifs that may correspond to real mutation signatures — cross-referenced against the COSMIC database (see `results/occlusion/motifs.json`).

### LLAMA-3 Clinical Report Generation

`src/report_generator.py` and [`LLMReports.ipynb`](notebooks/LLMReports.ipynb) feed each patient's cancer probability, top occlusion motifs, and cancer-type label into a prompt template for **LLAMA-3-8B-Instruct** (4-bit quantized via `BitsAndBytesConfig`, cached on Drive to avoid re-downloading each Colab session), producing structured clinical-style summary reports saved as `.txt` under `results/reports/`.

---

## 🗓️ Semester 2 Roadmap

| Issue | Task | Assignee | Status |
|-------|------|----------|--------|
| [#1](https://github.com/osmannselim/Genomic-RawSeq-Analyzer/issues/1) | Refactor notebooks → modular Python scripts | Both | ✅ Done |
| [#2](https://github.com/osmannselim/Genomic-RawSeq-Analyzer/issues/2) | Patient-level aggregation pipeline | Osman | ✅ Done |
| [#3](https://github.com/osmannselim/Genomic-RawSeq-Analyzer/issues/3) | DNABERT-2 integration | Nuri | ✅ Done |
| [#4](https://github.com/osmannselim/Genomic-RawSeq-Analyzer/issues/4) | DNABERT-2 vs CNN benchmark | Both | ✅ Done |
| [#5](https://github.com/osmannselim/Genomic-RawSeq-Analyzer/issues/5) | LLAMA-3 report generation module | Nuri | ✅ Done |
| [#6](https://github.com/osmannselim/Genomic-RawSeq-Analyzer/issues/6) | Multi-cancer dataset expansion (BRCA, LUAD) | Osman | ✅ Done |
| [#7](https://github.com/osmannselim/Genomic-RawSeq-Analyzer/issues/7) | Aggregate occlusion sensitivity analysis | Nuri | ✅ Done |
| [#8](https://github.com/osmannselim/Genomic-RawSeq-Analyzer/issues/8) | Final report & presentation | Both | 📋 In Progress |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10 |
| **Deep Learning** | TensorFlow 2.15 / Keras |
| **Transformers** | HuggingFace Transformers (DNABERT-2 fine-tuned, LLAMA-3-8B-Instruct 4-bit) |
| **Bioinformatics** | BioPython 1.81 |
| **Data** | NumPy 1.26, Pandas 2.1 |
| **Evaluation** | Scikit-learn, Matplotlib |
| **Data Sources** | NCBI SRA, GEO, **ENA** (direct HTTPS FASTQ mirror — used for streaming + partial range-fetch downloads, bypassing the SRA-toolkit) |
| **Compute** | Google Colab Pro (NVIDIA T4 / A100) |
| **Storage** | Google Drive — [`DNA_Anomaly_Detection`](https://drive.google.com/drive/folders/1dd3XQjEpEytE-6YFzRvpFFuVQE4HMBfj?usp=sharing) (models, batches, checkpoints, results) |

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
