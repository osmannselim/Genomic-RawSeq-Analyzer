"""
evaluate.py
-----------
Evaluation utilities for the Genomic-RawSeq-Analyzer pipeline.

Provides:
  - read_level_auc()         : ROC-AUC on individual sequencing reads
  - patient_level_auc()      : Aggregated (crowd-voting) patient-level AUC
  - prediction_distribution(): Visualise model confidence distributions
  - benchmark_comparison()   : Side-by-side metrics table for model comparison

Patient-level aggregation hypothesis:
    Individual reads are weak predictors (AUC ≈ 0.615).
    Aggregating predictions across all reads from the same patient
    (SRA run = proxy for patient) should yield a much stronger signal.

    Patient cancer probability = mean(P(cancer | read_i)) for all reads i

Usage:
    from evaluate import Evaluator

    evaluator = Evaluator(model, X_test, y_test, run_ids_test)
    evaluator.full_report()
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score,
    precision_recall_fscore_support, classification_report,
)


class Evaluator:
    """
    Comprehensive evaluation for binary cancer classification.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model to evaluate.
    X_test : np.ndarray, shape (N, max_len)
        Test sequences (integer-encoded).
    y_test : np.ndarray, shape (N,)
        Ground-truth binary labels (0=Normal, 1=Tumor).
    run_ids : np.ndarray or list, optional
        SRA run accession per read (for patient-level aggregation).
        Shape (N,). If None, patient-level evaluation is skipped.
    """

    def __init__(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        run_ids=None,
    ):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.run_ids = np.array(run_ids) if run_ids is not None else None

        self._probs = None   # cached read-level predictions

    # ── Predictions ───────────────────────────────────────

    @property
    def probs(self) -> np.ndarray:
        """Read-level cancer probabilities (cached after first call)."""
        if self._probs is None:
            print("Running inference...")
            t0 = time.time()
            self._probs = self.model.predict(
                self.X_test, batch_size=2048, verbose=1
            ).flatten()
            print(f"Inference time: {time.time() - t0:.1f}s")
        return self._probs

    # ── Read-level evaluation ─────────────────────────────

    def read_level_auc(self, plot: bool = True, save_path: str = None) -> float:
        """
        Compute and optionally plot the read-level ROC curve.

        Returns
        -------
        float : AUC-ROC score
        """
        fpr, tpr, _ = roc_curve(self.y_test, self.probs)
        roc_auc = auc(fpr, tpr)

        print(f"\nRead-level AUC: {roc_auc:.4f}")

        if plot:
            plt.figure(figsize=(7, 7))
            plt.plot(fpr, tpr, color="darkorange", lw=2,
                     label=f"ROC (AUC = {roc_auc:.4f})")
            plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate", fontsize=12)
            plt.ylabel("True Positive Rate", fontsize=12)
            plt.title("Read-Level ROC Curve", fontsize=14)
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=200, bbox_inches="tight")
                print(f"Saved ROC curve: {save_path}")
            plt.show()

        return roc_auc

    def classification_metrics(self, threshold: float = 0.5) -> dict:
        """
        Return precision, recall, F1, and accuracy at a given threshold.
        """
        y_pred = (self.probs >= threshold).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, average="binary", zero_division=0
        )
        acc = np.mean(y_pred == self.y_test)
        metrics = {
            "threshold": threshold,
            "accuracy": acc,
            "precision": p,
            "recall": r,
            "f1": f1,
            "auc": roc_auc_score(self.y_test, self.probs),
        }
        print(f"\nClassification Metrics (threshold={threshold}):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        return metrics

    # ── Patient-level aggregation ─────────────────────────

    def patient_level_auc(
        self,
        threshold: float = 0.5,
        plot: bool = True,
        save_path: str = None,
    ) -> dict:
        """
        Crowd-voting: aggregate read-level probabilities per patient (run_id),
        then evaluate patient-level AUC and classification metrics.

        Each patient's cancer probability = mean(read probabilities).
        The ground-truth patient label is the majority label of its reads.

        Parameters
        ----------
        threshold : float
            Decision threshold for patient-level binary classification.
        plot : bool
            If True, generate ROC curve and box plot.
        save_path : str
            Base path for saving plots (without extension).

        Returns
        -------
        dict : patient-level metrics including AUC, precision, recall, F1,
               and the per-patient DataFrame for downstream analysis.
               Returns None if run_ids not provided.
        """
        if self.run_ids is None:
            print("run_ids not provided — skipping patient-level evaluation.")
            return None

        # ── Build per-patient aggregation ──
        df = pd.DataFrame({
            "run_id": self.run_ids,
            "prob": self.probs,
            "label": self.y_test,
        })

        patient_df = df.groupby("run_id").agg(
            patient_prob=("prob", "mean"),
            patient_label=("label", lambda x: int(x.mode()[0])),
            n_reads=("prob", "count"),
            prob_std=("prob", "std"),
            prob_median=("prob", "median"),
        ).reset_index()

        # ── Patient-level AUC ──
        patient_auc = roc_auc_score(
            patient_df["patient_label"], patient_df["patient_prob"]
        )

        # ── Patient-level classification at threshold ──
        patient_df["predicted"] = (patient_df["patient_prob"] >= threshold).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            patient_df["patient_label"], patient_df["predicted"],
            average="binary", zero_division=0,
        )
        acc = np.mean(
            patient_df["predicted"] == patient_df["patient_label"]
        )

        print(f"\nPatient-level aggregation (crowd-voting):")
        print(f"  Patients evaluated : {len(patient_df)}")
        print(f"    Normal patients  : {sum(patient_df['patient_label'] == 0)}")
        print(f"    Tumor patients   : {sum(patient_df['patient_label'] == 1)}")
        print(f"  Reads per patient  : {patient_df['n_reads'].mean():.0f} (mean)")
        print(f"  Patient-level AUC  : {patient_auc:.4f}")
        print(f"  Threshold          : {threshold}")
        print(f"  Accuracy           : {acc:.4f}")
        print(f"  Precision          : {p:.4f}")
        print(f"  Recall             : {r:.4f}")
        print(f"  F1 Score           : {f1:.4f}")

        # ── Per-patient detail table ──
        print(f"\n  Per-patient breakdown:")
        for _, row in patient_df.sort_values("patient_prob", ascending=False).iterrows():
            true_label = "TUMOR " if row["patient_label"] == 1 else "NORMAL"
            pred_label = "TUMOR " if row["predicted"] == 1 else "NORMAL"
            correct = "✓" if row["patient_label"] == row["predicted"] else "✗"
            print(f"    {row['run_id']}  "
                  f"P(cancer)={row['patient_prob']:.4f}  "
                  f"true={true_label}  pred={pred_label}  "
                  f"reads={row['n_reads']:,}  {correct}")

        if plot:
            self._plot_patient_roc(patient_df, patient_auc, save_path)
            self._plot_patient_boxplot(patient_df, threshold, save_path)

        return {
            "auc": patient_auc,
            "accuracy": acc,
            "precision": p,
            "recall": r,
            "f1": f1,
            "threshold": threshold,
            "n_patients": len(patient_df),
            "patient_df": patient_df,
        }

    # ── Patient-level plots ──────────────────────────────

    @staticmethod
    def _plot_patient_roc(patient_df, patient_auc, save_path=None):
        """Plot patient-level ROC curve."""
        fpr, tpr, _ = roc_curve(
            patient_df["patient_label"], patient_df["patient_prob"]
        )
        plt.figure(figsize=(7, 7))
        plt.plot(fpr, tpr, color="green", lw=2,
                 label=f"Patient ROC (AUC = {patient_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("Patient-Level ROC Curve (Crowd Voting)", fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            path = f"{save_path}_roc.png"
            plt.savefig(path, dpi=200, bbox_inches="tight")
            print(f"Saved: {path}")
        plt.show()

    @staticmethod
    def _plot_patient_boxplot(patient_df, threshold=0.5, save_path=None):
        """
        Box plot of patient cancer scores: Cancer vs Normal distributions.
        Visualises the separation achieved by crowd-voting aggregation.
        """
        normal_scores = patient_df.loc[
            patient_df["patient_label"] == 0, "patient_prob"
        ].values
        cancer_scores = patient_df.loc[
            patient_df["patient_label"] == 1, "patient_prob"
        ].values

        fig, ax = plt.subplots(figsize=(8, 6))

        bp = ax.boxplot(
            [normal_scores, cancer_scores],
            labels=["Normal", "Cancer"],
            patch_artist=True,
            widths=0.5,
            showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="gold", markersize=8),
        )

        bp["boxes"][0].set_facecolor("steelblue")
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor("crimson")
        bp["boxes"][1].set_alpha(0.6)

        # Overlay individual patient points (strip/swarm style)
        np.random.seed(42)
        jitter = 0.08
        ax.scatter(
            np.ones(len(normal_scores)) + np.random.uniform(-jitter, jitter, len(normal_scores)),
            normal_scores, color="steelblue", alpha=0.7, s=40, zorder=3, edgecolors="white",
        )
        ax.scatter(
            2 * np.ones(len(cancer_scores)) + np.random.uniform(-jitter, jitter, len(cancer_scores)),
            cancer_scores, color="crimson", alpha=0.7, s=40, zorder=3, edgecolors="white",
        )

        # Threshold line
        ax.axhline(y=threshold, color="gray", linestyle="--", lw=1.5,
                    label=f"Threshold = {threshold}")

        ax.set_ylabel("Aggregated Cancer Probability\n(mean across reads)", fontsize=11)
        ax.set_title("Patient-Level Score Distribution\n(Crowd-Voting Aggregation)", fontsize=13)
        ax.legend(loc="upper left")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        if save_path:
            path = f"{save_path}_boxplot.png"
            plt.savefig(path, dpi=200, bbox_inches="tight")
            print(f"Saved: {path}")
        plt.show()

    # ── Distribution plot ─────────────────────────────────

    def prediction_distribution(self, save_path: str = None):
        """
        Plot the distribution of cancer probabilities split by true label.
        Shows the separation gap between tumor and normal predictions.
        """
        cancer_p = self.probs[self.y_test == 1]
        normal_p = self.probs[self.y_test == 0]

        gap = np.mean(cancer_p) - np.mean(normal_p)

        print(f"\nPrediction distributions:")
        print(f"  Mean P(cancer | Normal reads): {np.mean(normal_p):.4f}")
        print(f"  Mean P(cancer | Tumor  reads): {np.mean(cancer_p):.4f}")
        print(f"  Separation gap:                {gap:.4f}")
        print(f"  → {'Signal DETECTED — aggregation will work!' if gap > 0.01 else 'No clear signal — model struggling.'}")

        plt.figure(figsize=(10, 5))
        plt.hist(normal_p, bins=50, alpha=0.55, density=True, color="steelblue", label="Normal reads")
        plt.hist(cancer_p, bins=50, alpha=0.55, density=True, color="crimson",   label="Tumor reads")
        plt.axvline(np.mean(normal_p), color="steelblue", linestyle="--", lw=1.5)
        plt.axvline(np.mean(cancer_p), color="crimson",   linestyle="--", lw=1.5)
        plt.title("Prediction Distribution: Tumor vs Normal Reads", fontsize=13)
        plt.xlabel("Predicted Cancer Probability", fontsize=11)
        plt.ylabel("Density", fontsize=11)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()

    # ── Full report ───────────────────────────────────────

    def full_report(self, save_dir: str = None):
        """
        Run the complete evaluation suite and print a summary table.
        Optionally save all plots to save_dir.
        """
        import os
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        _p = lambda name: os.path.join(save_dir, name) if save_dir else None

        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)

        read_auc  = self.read_level_auc(save_path=_p("roc_read_level.png"))
        self.prediction_distribution(save_path=_p("prediction_distribution.png"))
        self.classification_metrics()

        patient_result = None
        if self.run_ids is not None:
            patient_result = self.patient_level_auc(
                save_path=_p("patient_level") if save_dir else None,
            )

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Read-level AUC:    {read_auc:.4f}")
        if patient_result is not None:
            print(f"  Patient-level AUC: {patient_result['auc']:.4f}")
            boost = patient_result["auc"] - read_auc
            print(f"  AUC improvement:   {'+' if boost >= 0 else ''}{boost:.4f}"
                  f"  ({'↑ crowd-voting helps!' if boost > 0 else '↓ investigate'})")
        print("=" * 60)


# ─────────────────────────────────────────────────────────
# Benchmark comparison table (Semester 2: CNN vs DNABERT-2)
# ─────────────────────────────────────────────────────────

def benchmark_comparison(results: list) -> pd.DataFrame:
    """
    Print a formatted comparison table for multiple models.

    Parameters
    ----------
    results : list of dict, each with keys:
        model_name, auc, precision, recall, f1,
        train_time_min, inference_ms_per_batch, gpu_memory_gb

    Returns
    -------
    pd.DataFrame

    Example
    -------
    from evaluate import benchmark_comparison
    benchmark_comparison([
        {"model_name": "1D-CNN (Baseline)", "auc": 0.615, ...},
        {"model_name": "DNABERT-2",         "auc": ???,   ...},
    ])
    """
    df = pd.DataFrame(results)
    df = df.set_index("model_name")

    print("\n" + "=" * 70)
    print("MODEL BENCHMARK COMPARISON")
    print("=" * 70)
    print(df.to_string(float_format="{:.4f}".format))
    print("=" * 70)
    return df


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from tensorflow.keras.models import load_model
    from sklearn.model_selection import train_test_split
    from data_loader import DataLoader

    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--batch_dir",  required=True)
    parser.add_argument("--save_dir",   default=None)
    parser.add_argument("--test_size",  type=float, default=0.2)
    args = parser.parse_args()

    print("Loading model and data...")
    model = load_model(args.model_path)
    X, y, run_ids = DataLoader.load_all_batches(args.batch_dir)

    indices = np.arange(len(X))
    _, test_idx = train_test_split(
        indices, test_size=args.test_size, random_state=42, stratify=y
    )
    X_test, y_test = X[test_idx], y[test_idx]
    run_ids_test = run_ids[test_idx] if run_ids is not None else None

    evaluator = Evaluator(model, X_test, y_test, run_ids=run_ids_test)
    evaluator.full_report(save_dir=args.save_dir)
