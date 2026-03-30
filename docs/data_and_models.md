# Data & Model Storage

All large files (raw data batches, trained model checkpoints) are stored on Google Drive
and are **not tracked in this repository** (excluded via `.gitignore`).

## 📁 Google Drive Folder

> **[DNA_Anomaly_Detection — Open in Google Drive](https://drive.google.com/drive/folders/1YgsM7UzP8krtIH3Y6mJD9sEztNuiV1XL?usp=sharing)**

---

## Folder Structure

```
DNA_Anomaly_Detection/
│
├── raw_normal_batches/        # Raw nucleotide string sequences — Normal patients
│                              # (used as input for DNABERT-2 tokenizer)
│
├── raw_cancer_batches/        # Raw nucleotide string sequences — Tumor patients
│                              # (used as input for DNABERT-2 tokenizer)
│
├── normal_batches/            # Integer-encoded sequences — Normal patients
│                              # (used for CNN and LSTM Autoencoder, batch_*.npz format)
│
├── cancer_batches/            # Integer-encoded sequences — Tumor patients
│                              # (used for CNN and LSTM Autoencoder, batch_*.npz format)
│
├── dnabert_checkpoints/
│   └── best_dnabert_model.pth # Best DNABERT-2 checkpoint (saved after each epoch)
│
├── lstm_encoder.h5            # Trained LSTM Encoder (Semester 1)
└── lstm_autoencoder.h5        # Trained LSTM Autoencoder (Semester 1)
```

---

## Dataset Details

| Property | Value |
|----------|-------|
| Source | ENA / SRA (European Nucleotide Archive) |
| Cohort | Breast cancer whole-exome sequencing (WXS) |
| Run accessions | ERR166302 – ERR166337 |
| Total patients | 36 (22 Tumor, 14 Normal) |
| Total reads | ~1,800,000 (50,000 reads × 36 patients) |
| Read length | 80 bp (CNN/AE) · 128 bp (DNABERT-2) |

---

## Model Checkpoints

| File | Description | AUC (read-level) | AUC (patient-level) |
|------|-------------|-----------------|---------------------|
| `results/cnn_baseline.keras` | 1D-CNN baseline (Semester 2 rerun) | 0.6157 | 0.9156 |
| `lstm_autoencoder.h5` | LSTM Autoencoder (Semester 1) | 0.509 | — |
| `dnabert_checkpoints/best_dnabert_model.pth` | DNABERT-2 fine-tuned (3 epochs, A100) | ~0.63 | TBD |

---

## Reproducing Results

### CNN + Patient Aggregation
```bash
# Train
python src/train.py --model cnn --batch_dir <path_to_normal+cancer_batches> --output_dir results --epochs 5

# Evaluate with patient-level aggregation
python src/evaluate.py --model_path results/cnn_baseline.keras --batch_dir <path_to_batches> --save_dir results/evaluation
```

### DNABERT-2
Open `notebooks/TrainDNABERT_2.ipynb` in Google Colab (A100 GPU recommended).
Mount Drive, set paths in **Cell 2**, then run all cells sequentially.
