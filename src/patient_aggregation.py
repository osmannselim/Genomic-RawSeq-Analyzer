"""
patient_aggregation.py
----------------------
Patient-level crowd-voting aggregation pipeline.

Core hypothesis: individual reads are weak classifiers (AUC ≈ 0.615),
but aggregating thousands of read predictions per patient via
mean-probability voting can yield a much stronger patient-level signal.

Pipeline:
    1. Group read-level predictions by run_accession (patient ID)
    2. Compute mean cancer probability per patient
    3. Apply threshold → binary patient-level label
    4. Evaluate patient-level AUC vs. read-level AUC
    5. Visualise Cancer vs Normal score distributions

Usage:
    from patient_aggregation import CrowdVotingAggregator

    agg = CrowdVotingAggregator(probs, y_true, run_ids)
    results = agg.run(threshold=0.5)
    agg.comparison_table(read_auc=0.615)

CLI:
    python patient_aggregation.py \\
        --model_path cnn_baseline.keras \\
        --batch_dir /path/to/batches \\
        --save_dir /path/to/results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score,
    precision_recall_fscore_support, confusion_matrix,
)


class CrowdVotingAggregator:
    """
    Aggregate read-level cancer probabilities into patient-level diagnoses.

    Parameters
    ----------
    probs : np.ndarray, shape (N,)
        Read-level cancer probabilities from the model.
    y_true : np.ndarray, shape (N,)
        Ground-truth binary labels (0=Normal, 1=Tumor) per read.
    run_ids : np.ndarray, shape (N,)
        SRA run accession IDs (proxy for patient identity) per read.
    """

    def __init__(
        self,
        probs: np.ndarray,
        y_true: np.ndarray,
        run_ids: np.ndarray,
    ):
        if len(probs) != len(y_true) or len(probs) != len(run_ids):
            raise ValueError(
                f"Length mismatch: probs={len(probs)}, "
                f"y_true={len(y_true)}, run_ids={len(run_ids)}"
            )

        self.probs = np.asarray(probs).flatten()
        self.y_true = np.asarray(y_true).flatten()
        self.run_ids = np.asarray(run_ids).flatten()

        self._patient_df = None  # lazily built

    # ── Core aggregation ──────────────────────────────────

    @property
    def patient_df(self) -> pd.DataFrame:
        """Per-patient aggregation DataFrame (built once, cached)."""
        if self._patient_df is None:
            self._patient_df = self._aggregate()
        return self._patient_df

    def _aggregate(self) -> pd.DataFrame:
        """
        Group reads by run_id and compute patient-level statistics.

        For each patient:
            - patient_prob   : mean cancer probability across all reads
            - patient_label  : ground-truth (majority label of its reads)
            - n_reads        : number of reads for this patient
            - prob_std       : standard deviation of read probabilities
            - prob_median    : median cancer probability
            - prob_q25, q75  : interquartile range of read probabilities
        """
        df = pd.DataFrame({
            "run_id": self.run_ids,
            "prob": self.probs,
            "label": self.y_true,
        })

        patient_df = df.groupby("run_id").agg(
            patient_prob=("prob", "mean"),
            patient_label=("label", lambda x: int(x.mode()[0])),
            n_reads=("prob", "count"),
            prob_std=("prob", "std"),
            prob_median=("prob", "median"),
            prob_q25=("prob", lambda x: np.percentile(x, 25)),
            prob_q75=("prob", lambda x: np.percentile(x, 75)),
        ).reset_index()

        return patient_df

    # ── Evaluation ────────────────────────────────────────

    def run(self, threshold: float = 0.5) -> dict:
        """
        Run the full patient-level evaluation pipeline.

        Parameters
        ----------
        threshold : float
            Decision threshold for converting mean probability to binary label.

        Returns
        -------
        dict with keys: auc, accuracy, precision, recall, f1, threshold,
                        n_patients, confusion_matrix, patient_df
        """
        pdf = self.patient_df.copy()

        # AUC
        patient_auc = roc_auc_score(pdf["patient_label"], pdf["patient_prob"])

        # Threshold → binary predictions
        pdf["predicted"] = (pdf["patient_prob"] >= threshold).astype(int)

        # Classification metrics
        p, r, f1, _ = precision_recall_fscore_support(
            pdf["patient_label"], pdf["predicted"],
            average="binary", zero_division=0,
        )
        acc = np.mean(pdf["predicted"] == pdf["patient_label"])
        cm = confusion_matrix(pdf["patient_label"], pdf["predicted"])

        return {
            "auc": patient_auc,
            "accuracy": acc,
            "precision": p,
            "recall": r,
            "f1": f1,
            "threshold": threshold,
            "n_patients": len(pdf),
            "confusion_matrix": cm,
            "patient_df": pdf,
        }

    def threshold_sweep(
        self,
        thresholds: np.ndarray = None,
    ) -> pd.DataFrame:
        """
        Evaluate patient-level metrics across a range of thresholds.

        Useful for finding the optimal decision boundary and for
        plotting threshold vs. F1/precision/recall curves.

        Returns
        -------
        pd.DataFrame with columns: threshold, accuracy, precision, recall, f1
        """
        if thresholds is None:
            thresholds = np.arange(0.30, 0.71, 0.02)

        rows = []
        for t in thresholds:
            result = self.run(threshold=t)
            rows.append({
                "threshold": t,
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
            })
        return pd.DataFrame(rows)

    # ── Comparison table ──────────────────────────────────

    def comparison_table(self, read_auc: float) -> pd.DataFrame:
        """
        Print the key comparison table: read-level vs patient-level AUC.

        Parameters
        ----------
        read_auc : float
            The read-level AUC to compare against.

        Returns
        -------
        pd.DataFrame
        """
        patient_auc = roc_auc_score(
            self.patient_df["patient_label"],
            self.patient_df["patient_prob"],
        )

        data = {
            "Level": ["Read-level", "Patient-level (aggregated)"],
            "AUC": [read_auc, patient_auc],
        }
        df = pd.DataFrame(data)

        print("\n" + "=" * 50)
        print("  READ-LEVEL vs PATIENT-LEVEL COMPARISON")
        print("=" * 50)
        print(df.to_string(index=False, float_format="{:.4f}".format))
        boost = patient_auc - read_auc
        print(f"\n  Improvement: {'+' if boost >= 0 else ''}{boost:.4f}")
        print("=" * 50)

        return df

    # ── Visualisation ─────────────────────────────────────

    def plot_boxplot(self, threshold: float = 0.5, save_path: str = None):
        """
        Box plot of patient cancer scores: Cancer vs Normal distributions.
        Each dot is one patient. Threshold line shown for reference.
        """
        pdf = self.patient_df
        normal = pdf.loc[pdf["patient_label"] == 0, "patient_prob"].values
        cancer = pdf.loc[pdf["patient_label"] == 1, "patient_prob"].values

        fig, ax = plt.subplots(figsize=(8, 6))

        bp = ax.boxplot(
            [normal, cancer],
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

        # Overlay individual patient points
        np.random.seed(42)
        jitter = 0.08
        ax.scatter(
            np.ones(len(normal)) + np.random.uniform(-jitter, jitter, len(normal)),
            normal, color="steelblue", alpha=0.7, s=50, zorder=3, edgecolors="white",
        )
        ax.scatter(
            2 * np.ones(len(cancer)) + np.random.uniform(-jitter, jitter, len(cancer)),
            cancer, color="crimson", alpha=0.7, s=50, zorder=3, edgecolors="white",
        )

        ax.axhline(y=threshold, color="gray", linestyle="--", lw=1.5,
                    label=f"Threshold = {threshold}")
        ax.set_ylabel("Aggregated Cancer Probability\n(mean across reads)", fontsize=11)
        ax.set_title("Patient-Level Score Distribution\n(Crowd-Voting Aggregation)", fontsize=13)
        ax.legend(loc="upper left")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()

    def plot_roc(self, read_auc: float = None, save_path: str = None):
        """
        Plot patient-level ROC curve.
        Optionally overlay read-level AUC as a reference point.
        """
        pdf = self.patient_df
        fpr, tpr, _ = roc_curve(pdf["patient_label"], pdf["patient_prob"])
        patient_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 7))
        plt.plot(fpr, tpr, color="green", lw=2,
                 label=f"Patient-level (AUC = {patient_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")

        if read_auc is not None:
            plt.axhline(y=read_auc, color="orange", linestyle=":", alpha=0.6,
                         label=f"Read-level AUC = {read_auc:.4f}")

        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("Patient-Level ROC (Crowd Voting)", fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()

    def plot_threshold_sweep(self, save_path: str = None):
        """
        Plot precision, recall, and F1 across thresholds.
        Helps identify the optimal operating point.
        """
        sweep_df = self.threshold_sweep()

        plt.figure(figsize=(9, 5))
        plt.plot(sweep_df["threshold"], sweep_df["precision"],
                 "b-o", markersize=4, label="Precision")
        plt.plot(sweep_df["threshold"], sweep_df["recall"],
                 "r-s", markersize=4, label="Recall")
        plt.plot(sweep_df["threshold"], sweep_df["f1"],
                 "g-^", markersize=5, lw=2, label="F1 Score")
        plt.xlabel("Decision Threshold", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.title("Patient-Level Threshold Sweep", fontsize=13)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()

    # ── Full pipeline ─────────────────────────────────────

    def full_report(
        self,
        read_auc: float = None,
        threshold: float = 0.5,
        save_dir: str = None,
    ) -> dict:
        """
        Run the complete patient-level aggregation report.

        Produces:
            - Per-patient summary table
            - Comparison table (read-level vs patient-level AUC)
            - Box plot (Cancer vs Normal score distributions)
            - ROC curve
            - Threshold sweep plot
        """
        import os
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        _p = lambda name: os.path.join(save_dir, name) if save_dir else None

        print("\n" + "=" * 60)
        print("PATIENT-LEVEL AGGREGATION REPORT")
        print("=" * 60)

        result = self.run(threshold=threshold)

        # Print per-patient details
        pdf = result["patient_df"]
        print(f"\nPer-patient breakdown:")
        for _, row in pdf.sort_values("patient_prob", ascending=False).iterrows():
            true_lbl = "TUMOR " if row["patient_label"] == 1 else "NORMAL"
            pred_lbl = "TUMOR " if row["predicted"] == 1 else "NORMAL"
            correct  = "✓" if row["patient_label"] == row["predicted"] else "✗"
            print(f"  {row['run_id']}  "
                  f"P(cancer)={row['patient_prob']:.4f}  "
                  f"true={true_lbl}  pred={pred_lbl}  "
                  f"reads={row['n_reads']:,}  {correct}")

        print(f"\nConfusion Matrix (threshold={threshold}):")
        cm = result["confusion_matrix"]
        print(f"  TN={cm[0, 0]}  FP={cm[0, 1]}")
        print(f"  FN={cm[1, 0]}  TP={cm[1, 1]}")

        if read_auc is not None:
            self.comparison_table(read_auc)

        self.plot_boxplot(threshold=threshold, save_path=_p("patient_boxplot.png"))
        self.plot_roc(read_auc=read_auc, save_path=_p("patient_roc.png"))
        self.plot_threshold_sweep(save_path=_p("threshold_sweep.png"))

        return result


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run patient-level crowd-voting aggregation."
    )
    parser.add_argument("--model_path", required=True,
                        help="Path to trained .keras model")
    parser.add_argument("--batch_dir", required=True,
                        help="Directory with batch_*.npz files (must include run_ids)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold (default: 0.5)")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test split fraction (default: 0.2)")
    parser.add_argument("--save_dir", default=None,
                        help="Directory to save plots and results")
    args = parser.parse_args()

    from tensorflow.keras.models import load_model
    from sklearn.model_selection import train_test_split
    from data_loader import DataLoader

    print("Loading model...")
    model = load_model(args.model_path)

    print("Loading data...")
    X, y, run_ids = DataLoader.load_all_batches(args.batch_dir)

    if run_ids is None:
        raise RuntimeError(
            "Batch files do not contain run_ids. "
            "Re-run data_loader.py with the updated version to include run_ids."
        )

    # Stratified split preserving run_ids alignment
    indices = np.arange(len(X))
    _, test_idx = train_test_split(
        indices, test_size=args.test_size, random_state=42, stratify=y
    )
    X_test, y_test = X[test_idx], y[test_idx]
    run_ids_test = run_ids[test_idx]

    # Get predictions
    print("Running inference...")
    probs = model.predict(X_test, batch_size=2048, verbose=1).flatten()

    # Read-level AUC for comparison
    from sklearn.metrics import roc_auc_score
    read_auc = roc_auc_score(y_test, probs)
    print(f"Read-level AUC: {read_auc:.4f}")

    # Patient-level aggregation
    aggregator = CrowdVotingAggregator(probs, y_test, run_ids_test)
    aggregator.full_report(
        read_auc=read_auc,
        threshold=args.threshold,
        save_dir=args.save_dir,
    )
