"""
multi_cancer_loader.py
----------------------
Helpers for the MultiCancerData notebook (Semester 2).

Supports downloading and evaluating BRCA and LUAD cohorts from NCBI SRA
to test zero-shot transfer of the Semester 1 CNN.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support


# ── Cohort metadata ───────────────────────────────────────────────────

COHORT_METADATA = {
    'brca': {
        'cancer_label': 'Breast Invasive Carcinoma (BRCA)',
        'geo_accession': 'GSE48215',
        'sra_project':   'SRP028580',
        'n_tumor':  25,
        'n_normal': 25,
        'runs': {
            # 8-sample hardcoded fallback (4 tumor, 4 normal)
            'SRR949537': 1, 'SRR949538': 1, 'SRR949539': 1, 'SRR949540': 1,
            'SRR949541': 0, 'SRR949542': 0, 'SRR949543': 0, 'SRR949544': 0,
        },
    },
    'luad': {
        'cancer_label': 'Lung Adenocarcinoma (LUAD)',
        'geo_accession': 'GSE40419',
        'sra_project':   'SRP013469',
        'n_tumor':  17,
        'n_normal': 13,
        'runs': {
            # 8-sample hardcoded fallback (4 tumor, 4 normal)
            'SRR521456': 1, 'SRR521457': 1, 'SRR521458': 1, 'SRR521459': 1,
            'SRR521460': 0, 'SRR521461': 0, 'SRR521462': 0, 'SRR521463': 0,
        },
    },
}


# ── SraRunTable parser ────────────────────────────────────────────────

def assign_labels_from_sra_table(table_path: str) -> pd.DataFrame:
    """
    Parse an NCBI SraRunTable.txt and assign binary labels.

    Heuristic: rows where sample_type / tissue_type / source_name contains
    'tumor' or 'cancer' → label 1; 'normal' or 'healthy' → label 0.
    Rows that cannot be classified are dropped with a warning.

    Returns
    -------
    pd.DataFrame with columns: Run, Label
    """
    df = pd.read_csv(table_path, sep=None, engine='python')
    df.columns = [c.strip() for c in df.columns]

    run_col = next((c for c in df.columns if c.lower() in ('run', 'run_id', 'sra_id')), None)
    if run_col is None:
        raise ValueError("SraRunTable has no 'Run' column.")

    text_cols = [c for c in df.columns if c.lower() in
                 ('sample_type', 'tissue_type', 'source_name', 'disease',
                  'tumor_normal', 'sample_description')]

    rows = []
    for _, row in df.iterrows():
        combined = ' '.join(str(row.get(c, '')) for c in text_cols).lower()
        if any(t in combined for t in ('tumor', 'cancer', 'malignant')):
            rows.append({'Run': row[run_col], 'Label': 1})
        elif any(t in combined for t in ('normal', 'healthy', 'adjacent')):
            rows.append({'Run': row[run_col], 'Label': 0})
        else:
            print(f"  WARNING: could not classify {row[run_col]} — skipped. Text: '{combined[:80]}'")

    result = pd.DataFrame(rows)
    print(f"Parsed {len(result)} runs  (tumor={result['Label'].sum()}, "
          f"normal={(result['Label']==0).sum()})")
    return result


# ── Class balance report ──────────────────────────────────────────────

def class_balance_report(batch_dir: str, cancer_label: str = 'Cancer') -> None:
    """Print and plot class balance for all batches in batch_dir."""
    from data_loader import DataLoader
    X, y, _ = DataLoader.load_all_batches(batch_dir)

    n_tumor  = int(y.sum())
    n_normal = int((y == 0).sum())
    total    = len(y)
    ratio    = n_tumor / total if total else 0

    print(f'\n{"="*50}')
    print(f'CLASS BALANCE — {cancer_label}')
    print(f'{"="*50}')
    print(f'  Total reads : {total:,}')
    print(f'  Tumor       : {n_tumor:,}  ({ratio:.1%})')
    print(f'  Normal      : {n_normal:,}  ({1-ratio:.1%})')
    print(f'{"="*50}')

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(['Normal', 'Tumor'], [n_normal, n_tumor], color=['#2ecc71', '#e74c3c'])
    ax.set_title(f'Class Balance\n{cancer_label}', fontsize=12)
    ax.set_ylabel('Read Count')
    for i, v in enumerate([n_normal, n_tumor]):
        ax.text(i, v + total * 0.005, f'{v:,}', ha='center', fontsize=10)
    plt.tight_layout()
    save_path = os.path.join(batch_dir, 'class_balance.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {save_path}')


# ── Zero-shot evaluation ──────────────────────────────────────────────

def zero_shot_eval(
    model_path: str,
    batch_dir: str,
    cancer_label: str = 'Cancer',
    save_dir: str = None,
) -> dict:
    """
    Load the Semester 1 CNN and evaluate it zero-shot on a new cohort.

    Produces read-level and patient-level ROC curves, prints a summary table,
    and saves plots to save_dir.

    Returns
    -------
    dict with keys: read_auc, patient_auc, precision, recall, f1
    """
    from tensorflow.keras.models import load_model
    from data_loader import DataLoader

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print(f'Loading model from {model_path} ...')
    model = load_model(model_path)

    print(f'Loading batches from {batch_dir} ...')
    X, y, run_ids = DataLoader.load_all_batches(batch_dir)

    print('Running inference ...')
    probs = model.predict(X, batch_size=2048, verbose=1).flatten()

    # Read-level
    fpr, tpr, _ = roc_curve(y, probs)
    read_auc = auc(fpr, tpr)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, (probs >= 0.5).astype(int), average='binary', zero_division=0)

    # Patient-level crowd-voting
    pat_auc = None
    if run_ids is not None:
        df = pd.DataFrame({'run_id': run_ids, 'prob': probs, 'label': y})
        pat = df.groupby('run_id').agg(
            patient_prob=('prob', 'mean'),
            patient_label=('label', lambda x: int(x.mode()[0])),
        ).reset_index()
        if len(pat['patient_label'].unique()) > 1:
            fpr_p, tpr_p, _ = roc_curve(pat['patient_label'], pat['patient_prob'])
            pat_auc = auc(fpr_p, tpr_p)

    print(f'\n{"="*55}')
    print(f'ZERO-SHOT EVALUATION — {cancer_label}')
    print(f'{"="*55}')
    print(f'  Read-level AUC    : {read_auc:.4f}')
    if pat_auc is not None:
        print(f'  Patient-level AUC : {pat_auc:.4f}')
    print(f'  Precision         : {prec:.4f}')
    print(f'  Recall            : {rec:.4f}')
    print(f'  F1 Score          : {f1:.4f}')
    print(f'{"="*55}')

    # Plot
    fig, axes = plt.subplots(1, 2 if pat_auc is not None else 1,
                             figsize=(14 if pat_auc else 7, 5), dpi=130)
    if pat_auc is None:
        axes = [axes]

    axes[0].plot(fpr, tpr, color='#e74c3c', lw=2,
                 label=f'AUC = {read_auc:.4f}')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0].set_title(f'Read-Level ROC\n{cancer_label}', fontsize=12)
    axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
    axes[0].legend(loc='lower right')

    if pat_auc is not None:
        axes[1].plot(fpr_p, tpr_p, color='#2980b9', lw=2,
                     label=f'AUC = {pat_auc:.4f}')
        axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[1].set_title(f'Patient-Level ROC (Crowd-Voting)\n{cancer_label}', fontsize=12)
        axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
        axes[1].legend(loc='lower right')

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, 'zero_shot_roc.png')
        plt.savefig(path, dpi=200, bbox_inches='tight')
        print(f'Saved: {path}')
    plt.show()

    return {
        'read_auc': read_auc,
        'patient_auc': pat_auc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
    }
