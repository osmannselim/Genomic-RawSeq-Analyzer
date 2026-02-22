"""
explainability.py
-----------------
Explainability tools for the Genomic-RawSeq-Analyzer CNN model.

Implements Occlusion Sensitivity (position-masking) to identify which
nucleotide positions contribute most to the model's cancer prediction.
Gradient-based methods (GradCAM) are not straightforward here because the
input is a discrete integer sequence, so occlusion is the natural alternative.

Occlusion logic:
    For each position i in a read:
        mask position i with 0 (padding token)
        importance[i] = original_prob - masked_prob

    High importance = removing that nucleotide significantly reduces
    the model's confidence that the sequence is cancerous.
    These positions likely correspond to known mutation signatures.

Usage:
    from explainability import OcclusionAnalyzer
    analyzer = OcclusionAnalyzer(model, max_len=80)

    # Single read
    importance, prob = analyzer.explain_read(sequence)

    # Top-N most confident cancer reads from a pool
    analyzer.plot_top_cancer_reads(X_cancer, top_n=3)

    # Population-level: average importance across many reads
    avg_importance = analyzer.aggregate_importance(X_cancer, n_samples=500)
    analyzer.plot_aggregated_importance(avg_importance)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from data_loader import BASE_DECODING   # {1:'A', 2:'C', 3:'G', 4:'T', 5:'N', 0:'_'}


class OcclusionAnalyzer:
    """
    Position-masking explainability for the 1D-CNN cancer classifier.

    Parameters
    ----------
    model : tf.keras.Model
        Trained CNN model. Must accept input shape (N, max_len).
    max_len : int
        Fixed sequence length (default 80).
    """

    def __init__(self, model, max_len: int = 80):
        self.model = model
        self.max_len = max_len

    # ── Core ──────────────────────────────────────────────

    def explain_read(self, sequence: np.ndarray) -> tuple:
        """
        Compute per-position importance scores for a single read.

        Strategy: create a batch of seq_len + 1 sequences:
            - sequence[0]   = original (no mask)
            - sequence[i+1] = original with position i zeroed out
        Predict the whole batch in one forward pass (fast).

        Parameters
        ----------
        sequence : np.ndarray, shape (max_len,)
            Integer-encoded DNA sequence.

        Returns
        -------
        importance : np.ndarray, shape (max_len,)
            Per-position importance scores (clipped at 0, normalised 0–1).
        original_prob : float
            Model's cancer probability for the unmasked read.
        """
        seq_len = len(sequence)
        batch = np.zeros((seq_len + 1, seq_len), dtype=np.int8)

        # Row 0: original
        batch[0] = sequence

        # Rows 1..seq_len: one masked position each
        for i in range(seq_len):
            masked = sequence.copy()
            masked[i] = 0       # mask = padding token
            batch[i + 1] = masked

        preds = self.model.predict(batch, verbose=0).flatten()

        original_prob = float(preds[0])
        masked_probs = preds[1:]

        # Importance = drop in confidence when position is removed
        importance = original_prob - masked_probs
        importance = np.maximum(importance, 0)      # only positive contributions

        # Normalise for visualisation
        if importance.max() > 0:
            importance = importance / importance.max()

        return importance, original_prob

    # ── Single-read plot ──────────────────────────────────

    def plot_read(
        self,
        sequence: np.ndarray,
        read_id: int = 0,
        ax=None,
        save_path: str = None,
    ):
        """
        Plot occlusion sensitivity map for one read.

        Bar height = importance of each nucleotide position.
        X-axis labels = actual DNA letters at each position.
        """
        importance, prob = self.explain_read(sequence)
        seq_chars = [BASE_DECODING.get(int(x), "?") for x in sequence]

        colors = cm.Reds(importance)

        show = ax is None
        if ax is None:
            fig, ax = plt.subplots(figsize=(18, 4))

        ax.bar(range(len(importance)), importance, color=colors, alpha=0.85)
        ax.set_xticks(range(len(seq_chars)))
        ax.set_xticklabels(seq_chars, fontsize=8, fontfamily="monospace")
        ax.set_xlim(-1, len(seq_chars))
        ax.set_title(
            f"Occlusion Sensitivity Map  |  Read #{read_id}\n"
            f"Cancer Confidence: {prob:.1%}",
            fontsize=13,
        )
        ax.set_xlabel("Position  (red bars = mutation-associated motifs)", fontsize=11)
        ax.set_ylabel("Importance Score", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved: {save_path}")

        if show:
            plt.show()

    # ── Top-N plot ────────────────────────────────────────

    def plot_top_cancer_reads(
        self,
        X_cancer: np.ndarray,
        top_n: int = 3,
        sample_size: int = 2000,
        save_dir: str = None,
    ):
        """
        Find the top_n reads the model is most confident are cancerous
        and plot their occlusion sensitivity maps.

        Parameters
        ----------
        X_cancer : np.ndarray, shape (N, max_len)
        top_n : int
            Number of reads to visualise.
        sample_size : int
            Subsample this many reads before ranking (for speed).
        save_dir : str, optional
            If set, save each plot as a PNG here.
        """
        sample = X_cancer[:sample_size]
        preds = self.model.predict(sample, batch_size=512, verbose=0).flatten()
        top_indices = preds.argsort()[-top_n:][::-1]

        print(f"Top {top_n} most confident cancer reads (from {sample_size} samples):")
        for rank, idx in enumerate(top_indices, 1):
            save_path = None
            if save_dir:
                import os
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"saliency_rank{rank}_read{idx}.png")
            print(f"  Rank {rank}: Read #{idx}  confidence={preds[idx]:.2%}")
            self.plot_read(sample[idx], read_id=idx, save_path=save_path)

    # ── Population-level aggregation ─────────────────────

    def aggregate_importance(
        self,
        X: np.ndarray,
        n_samples: int = 500,
        confidence_threshold: float = 0.6,
    ) -> np.ndarray:
        """
        Compute average per-position importance across multiple reads.

        Only reads where model confidence > confidence_threshold are included,
        ensuring we average over reads the model genuinely identifies as cancer.

        This aggregated map reveals consistently mutated positions across
        patients — potentially corresponding to COSMIC mutation signatures.

        Parameters
        ----------
        X : np.ndarray, shape (N, max_len)
        n_samples : int
            Maximum reads to analyse (batch-predicted first for speed).
        confidence_threshold : float
            Minimum cancer probability to include a read.

        Returns
        -------
        avg_importance : np.ndarray, shape (max_len,)
            Mean normalised importance per position.
        """
        sample = X[:n_samples]
        probs = self.model.predict(sample, batch_size=512, verbose=0).flatten()

        high_conf_idx = np.where(probs >= confidence_threshold)[0]
        print(f"  {len(high_conf_idx)}/{n_samples} reads exceed "
              f"confidence threshold {confidence_threshold:.0%}")

        if len(high_conf_idx) == 0:
            print("  Warning: No reads above threshold. Lowering to top-10% percentile.")
            high_conf_idx = probs.argsort()[-max(1, n_samples // 10):][::-1]

        all_importance = []
        for idx in high_conf_idx:
            imp, _ = self.explain_read(sample[idx])
            all_importance.append(imp)

        avg_importance = np.mean(all_importance, axis=0)

        # Normalise
        if avg_importance.max() > 0:
            avg_importance = avg_importance / avg_importance.max()

        return avg_importance

    def plot_aggregated_importance(
        self,
        avg_importance: np.ndarray,
        title: str = "Aggregated Occlusion Sensitivity (Population Average)",
        save_path: str = None,
    ):
        """
        Plot the position-averaged importance heatmap.
        Highlights consistently important motif positions across patients.
        """
        plt.figure(figsize=(18, 4))
        colors = cm.Reds(avg_importance)
        plt.bar(range(len(avg_importance)), avg_importance, color=colors, alpha=0.85)
        plt.title(title, fontsize=13)
        plt.xlabel("Position in 80-bp Read", fontsize=11)
        plt.ylabel("Mean Importance Score", fontsize=11)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved: {save_path}")

        plt.show()


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from tensorflow.keras.models import load_model
    from data_loader import DataLoader

    parser = argparse.ArgumentParser(description="Run occlusion sensitivity analysis.")
    parser.add_argument("--model_path", required=True, help="Path to .keras model file")
    parser.add_argument("--batch_dir",  required=True, help="Directory with batch_*.npz files")
    parser.add_argument("--top_n",   type=int, default=3)
    parser.add_argument("--n_agg",   type=int, default=500, help="Reads for aggregation")
    parser.add_argument("--save_dir", default=None, help="Directory to save plots")
    args = parser.parse_args()

    print("Loading model...")
    model = load_model(args.model_path)

    print("Loading data...")
    X, y = DataLoader.load_all_batches(args.batch_dir)
    X_cancer = X[y == 1]
    print(f"Cancer reads: {len(X_cancer):,}")

    analyzer = OcclusionAnalyzer(model)

    print(f"\nPlotting top {args.top_n} cancer reads...")
    analyzer.plot_top_cancer_reads(X_cancer, top_n=args.top_n, save_dir=args.save_dir)

    print(f"\nComputing aggregated importance ({args.n_agg} reads)...")
    avg_imp = analyzer.aggregate_importance(X_cancer, n_samples=args.n_agg)
    save_path = f"{args.save_dir}/aggregated_importance.png" if args.save_dir else None
    analyzer.plot_aggregated_importance(avg_imp, save_path=save_path)
