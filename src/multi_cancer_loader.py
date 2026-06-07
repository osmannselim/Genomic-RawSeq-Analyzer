"""
multi_cancer_loader.py
-----------------------
Dataset expansion: download and preprocess BRCA and LUAD WXS cohorts from NCBI SRA.

Extends the Semester 1 pipeline (single breast-cancer cohort) to support
multi-cancer generalisability testing as described in Chapter 9.3.

Cohorts targeted:
  BRCA — Breast Cancer (WXS)  — GEO: GSE48215  / SRA: SRP028580
  LUAD — Lung Adenocarcinoma  — GEO: GSE40419  / SRA: SRP013469

Both cohorts use the same ERR/SRR accession → Tumor/Normal label convention.
The existing data_loader.py pipeline handles download + encoding unchanged.

Usage:
    # 1. Download SRA run lists (see COHORT_METADATA below for accession links)
    # 2. Run this script to process and save batches per cancer type
    python src/multi_cancer_loader.py \
        --cancer_type brca \
        --output_dir  results/multi_cancer/brca \
        --max_reads_per_sample 50000

    # 3. Zero-shot CNN evaluation on new cohort
    python src/multi_cancer_loader.py \
        --cancer_type luad \
        --output_dir  results/multi_cancer/luad \
        --eval_model  results/cnn_baseline.keras
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from data_loader import DataLoader

# ── Cohort metadata ───────────────────────────────────────────────────────────
# Source: NCBI SRA / GEO
# Each entry: SRA run accession → binary label (1=Tumor, 0=Normal)
#
# To add a new cohort:
#   1. Search https://www.ncbi.nlm.nih.gov/sra with query:
#      "cancer_type[Organism] AND WXS[Strategy] AND Homo sapiens[Organism]"
#   2. Download the SraRunTable.txt for the SRP accession
#   3. Add entries below using assign_labels_from_sra_table()

COHORT_METADATA = {
    # ── BRCA: GSE48215 / SRP028580 ───────────────────────────────────────
    # 25 matched Tumor/Normal pairs, WXS, Illumina HiSeq
    # https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE48215
    "brca": {
        "geo_accession": "GSE48215",
        "sra_project":   "SRP028580",
        "cancer_label":  "Breast Invasive Carcinoma (BRCA)",
        "n_tumor":       25,
        "n_normal":      25,
        # Subset of run accessions (fill from SraRunTable.txt after download)
        # Format: {"SRR_ACCESSION": label}  1=Tumor, 0=Normal
        "runs": {
            # Populate from: https://trace.ncbi.nlm.nih.gov/Traces/sra/?study=SRP028580
            # Example entries — replace with actual accessions from SraRunTable.txt:
            "SRR975960": 1, "SRR975961": 1, "SRR975962": 1, "SRR975963": 1,
            "SRR975964": 1, "SRR975965": 0, "SRR975966": 0, "SRR975967": 0,
        },
    },

    # ── LUAD: GSE40419 / SRP013469 ───────────────────────────────────────
    # 17 Tumor / 13 Normal, WXS, Illumina HiSeq
    # https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40419
    "luad": {
        "geo_accession": "GSE40419",
        "sra_project":   "SRP013469",
        "cancer_label":  "Lung Adenocarcinoma (LUAD)",
        "n_tumor":       17,
        "n_normal":      13,
        "runs": {
            # Populate from SraRunTable.txt after download:
            "SRR521284": 1, "SRR521285": 1, "SRR521286": 1, "SRR521287": 1,
            "SRR521288": 1, "SRR521289": 0, "SRR521290": 0, "SRR521291": 0,
        },
    },
}


def assign_labels_from_sra_table(sra_table_path: str) -> pd.DataFrame:
    """
    Parse a downloaded SraRunTable.txt (from NCBI SRA Run Selector) and
    extract Run accessions + Tumor/Normal labels.

    The SRA Run Selector exports a CSV with columns including:
        Run, source_name (or tissue_type), LibraryStrategy, ...

    Returns DataFrame with columns: Run, Label (1=Tumor, 0=Normal)
    """
    df = pd.read_csv(sra_table_path)

    # Common column names for tumor/normal annotation in SRA tables
    label_col = None
    for col in ["source_name", "tissue_type", "Sample_Name", "phenotype",
                "disease_state", "tumor_tissue_site"]:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        raise ValueError(
            f"Could not find a tissue/disease column in {sra_table_path}. "
            f"Available columns: {list(df.columns)}"
        )

    tumor_keywords  = ["tumor", "tumour", "cancer", "malignant", "primary"]
    normal_keywords = ["normal", "healthy", "adjacent", "non-tumor", "control"]

    def parse_label(val: str) -> int:
        v = str(val).lower()
        if any(k in v for k in tumor_keywords):
            return 1
        if any(k in v for k in normal_keywords):
            return 0
        return -1  # unknown

    df["Label"] = df[label_col].apply(parse_label)
    unknown = (df["Label"] == -1).sum()
    if unknown > 0:
        print(f"  Warning: {unknown} samples with unrecognised tissue label — review manually.")
        df = df[df["Label"] != -1]

    result = df[["Run", "Label"]].copy()
    print(f"  Parsed {len(result)} samples: "
          f"{(result['Label']==1).sum()} tumor, "
          f"{(result['Label']==0).sum()} normal")
    return result


def download_and_process_cohort(
    cancer_type: str,
    output_dir: str,
    max_reads_per_sample: int = 50_000,
    sra_table_path: str = None,
) -> str:
    """
    Download and preprocess one cancer cohort using the existing DataLoader pipeline.

    Parameters
    ----------
    cancer_type : str
        One of "brca", "luad" (must be a key in COHORT_METADATA).
    output_dir : str
        Directory to save .npz batch files.
    max_reads_per_sample : int
        Reads per SRA run to download (matches Semester 1 default of 50,000).
    sra_table_path : str, optional
        Path to a downloaded SraRunTable.txt from NCBI Run Selector.
        If provided, overrides the hardcoded run dict in COHORT_METADATA.

    Returns
    -------
    str : output_dir path
    """
    if cancer_type not in COHORT_METADATA:
        raise ValueError(f"Unknown cancer_type '{cancer_type}'. "
                         f"Choose from: {list(COHORT_METADATA)}")

    meta = COHORT_METADATA[cancer_type]
    print(f"\n{'=' * 60}")
    print(f"COHORT: {meta['cancer_label']}")
    print(f"GEO   : {meta['geo_accession']}  |  SRA: {meta['sra_project']}")
    print(f"{'=' * 60}")

    # Build run DataFrame
    if sra_table_path:
        run_df = assign_labels_from_sra_table(sra_table_path)
    else:
        run_df = pd.DataFrame([
            {"Run": acc, "Label": lbl}
            for acc, lbl in meta["runs"].items()
        ])

    print(f"Runs to process: {len(run_df)}  "
          f"(tumor={int((run_df['Label']==1).sum())}, "
          f"normal={int((run_df['Label']==0).sum())})")

    os.makedirs(output_dir, exist_ok=True)
    loader = DataLoader(output_dir=output_dir, max_reads=max_reads_per_sample)
    loader.process_run_list(run_df)

    return output_dir


def zero_shot_eval(
    model_path: str,
    batch_dir: str,
    cancer_label: str = "Unknown",
    save_dir: str = None,
):
    """
    Evaluate the Semester 1 CNN (trained on WXS breast cancer) on a new cohort
    without any retraining — zero-shot transfer performance.

    Computes read-level and patient-level AUC, prints the benchmark table,
    and saves ROC curves to save_dir.
    """
    import matplotlib
    matplotlib.use("Agg")

    from tensorflow.keras.models import load_model
    from evaluate import Evaluator

    print(f"\n{'=' * 60}")
    print(f"ZERO-SHOT EVALUATION: {cancer_label}")
    print(f"{'=' * 60}")

    print("Loading model...")
    model = load_model(model_path)

    print("Loading cohort batches...")
    X, y, run_ids = DataLoader.load_all_batches(batch_dir)
    print(f"  {len(X):,} reads  |  "
          f"tumor={int(y.sum()):,}  normal={int((y==0).sum()):,}")

    evaluator = Evaluator(model, X, y, run_ids=run_ids)
    evaluator.full_report(save_dir=save_dir)


def class_balance_report(batch_dir: str, cancer_label: str = ""):
    """Print a class balance summary for a cohort batch directory."""
    X, y, run_ids = DataLoader.load_all_batches(batch_dir)
    unique, counts = np.unique(run_ids, return_counts=True)
    patients_df = pd.DataFrame({"run_id": unique, "n_reads": counts})

    # Infer per-patient label from majority vote
    import pandas as _pd
    df = _pd.DataFrame({"run_id": run_ids, "label": y})
    pat_labels = df.groupby("run_id")["label"].apply(
        lambda x: int(x.mode()[0])
    ).reset_index(name="label")
    patients_df = patients_df.merge(pat_labels, on="run_id")

    n_tumor  = (patients_df["label"] == 1).sum()
    n_normal = (patients_df["label"] == 0).sum()

    print(f"\nClass Balance — {cancer_label or batch_dir}")
    print(f"  Total reads   : {len(X):,}")
    print(f"  Total patients: {len(patients_df)}")
    print(f"  Tumor patients: {n_tumor}  ({100*n_tumor/len(patients_df):.1f}%)")
    print(f"  Normal patients:{n_normal}  ({100*n_normal/len(patients_df):.1f}%)")
    print(f"  Reads/patient  : {patients_df['n_reads'].mean():.0f} (mean)")
    print(f"  Class ratio    : {int(y.sum()):,} tumor / {int((y==0).sum()):,} normal "
          f"= {y.mean()*100:.1f}% tumor")
    return patients_df


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Download and process multi-cancer WXS cohorts from NCBI SRA."
    )
    p.add_argument("--cancer_type",  required=True, choices=list(COHORT_METADATA),
                   help="Cancer cohort to process: brca or luad")
    p.add_argument("--output_dir",   default=None,
                   help="Output directory for .npz batches "
                        "(default: results/multi_cancer/<cancer_type>)")
    p.add_argument("--max_reads",    type=int, default=50_000,
                   help="Reads per SRA sample to download")
    p.add_argument("--sra_table",    default=None,
                   help="Path to SraRunTable.txt from NCBI Run Selector "
                        "(overrides hardcoded run list)")
    p.add_argument("--eval_model",   default=None,
                   help="Path to .keras model for zero-shot eval after download")
    p.add_argument("--balance_only", action="store_true",
                   help="Only print class balance for an existing batch_dir "
                        "(skip download)")
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir or os.path.join(
        "results", "multi_cancer", args.cancer_type
    )
    meta = COHORT_METADATA[args.cancer_type]

    if args.balance_only:
        class_balance_report(output_dir, cancer_label=meta["cancer_label"])
        return

    batch_dir = download_and_process_cohort(
        cancer_type=args.cancer_type,
        output_dir=output_dir,
        max_reads_per_sample=args.max_reads,
        sra_table_path=args.sra_table,
    )

    class_balance_report(batch_dir, cancer_label=meta["cancer_label"])

    if args.eval_model:
        eval_dir = os.path.join(output_dir, "zero_shot_eval")
        zero_shot_eval(
            model_path=args.eval_model,
            batch_dir=batch_dir,
            cancer_label=meta["cancer_label"],
            save_dir=eval_dir,
        )


if __name__ == "__main__":
    main()
