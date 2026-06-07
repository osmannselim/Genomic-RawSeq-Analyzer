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

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
from data_loader import BASE_DECODING   # {1:'A', 2:'C', 3:'G', 4:'T', 5:'N', 0:'_'}

# ── COSMIC SBS signature patterns (trinucleotide contexts) ─────────────────────
# Each entry: (signature_id, description, list_of_trinucleotide_patterns)
# Patterns are 3-mers of the form X[REF>ALT]X (reference context).
# We simplify to the reference trinucleotide since we work with reference sequence.
COSMIC_SIGNATURES = [
    ("SBS1",  "Clock-like / CpG methylation (aging)",
     ["CGA", "CGC", "CGG", "CGT"]),
    ("SBS2",  "APOBEC cytidine deaminase",
     ["TCA", "TCG", "TCT", "TCC"]),
    ("SBS4",  "Tobacco carcinogen (C>A at CCN)",
     ["CCA", "CCG", "CCT", "CCC"]),
    ("SBS6",  "Mismatch repair deficiency",
     ["CAA", "CAT", "CTG", "CTA"]),
    ("SBS13", "APOBEC (C>G at TCN)",
     ["TCA", "TCT", "TCC", "TCG"]),
    ("SBS17", "Unknown / oxidative damage (T>G at GTT)",
     ["GTT", "GTC", "GTA", "GTG"]),
    ("SBS3",  "Homologous recombination deficiency (BRCA1/2)",
     ["ACT", "ACC", "ACA", "ACG"]),
]


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
        """Bar chart of position-averaged importance (quick overview)."""
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

    def plot_heatmap(
        self,
        importance_matrix: np.ndarray,
        save_path: str = None,
        title: str = "Per-Read Occlusion Importance Heatmap",
    ):
        """
        2D seaborn heatmap: rows = individual reads, columns = positions.
        Useful for spotting consistently important regions across reads.

        Parameters
        ----------
        importance_matrix : np.ndarray, shape (n_reads, max_len)
        """
        fig, axes = plt.subplots(2, 1, figsize=(20, 8),
                                 gridspec_kw={"height_ratios": [4, 1]})

        # Top panel: per-read heatmap
        sns.heatmap(
            importance_matrix,
            ax=axes[0],
            cmap="Reds",
            xticklabels=10,
            yticklabels=False,
            cbar_kws={"label": "Importance"},
        )
        axes[0].set_title(title, fontsize=13)
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Reads (high confidence cancer)", fontsize=10)

        # Bottom panel: population average
        avg = importance_matrix.mean(axis=0)
        axes[1].fill_between(range(len(avg)), avg, color="crimson", alpha=0.7)
        axes[1].set_xlim(0, len(avg))
        axes[1].set_ylim(0, avg.max() * 1.2 if avg.max() > 0 else 1)
        axes[1].set_xlabel("Position in 80-bp Read", fontsize=11)
        axes[1].set_ylabel("Mean", fontsize=9)
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved heatmap: {save_path}")
        plt.show()
        plt.close(fig)

    # ── K-mer extraction ──────────────────────────────────

    def extract_top_kmers(
        self,
        importance: np.ndarray,
        sequence: np.ndarray,
        k: int = 5,
        top_n: int = 3,
    ) -> list:
        """
        Identify the top_n highest-importance k-mers in a single read.

        Uses a sliding window: importance of a k-mer = mean importance of
        its constituent positions. Skips windows containing padding ('_').

        Returns
        -------
        list of dict: [{"position": int, "kmer": str, "score": float}, ...]
        sorted descending by score, length top_n.
        """
        seq_chars = [BASE_DECODING.get(int(x), "N") for x in sequence]
        results = []
        for i in range(len(importance) - k + 1):
            window_chars = seq_chars[i: i + k]
            if "_" in window_chars:
                continue
            kmer  = "".join(window_chars)
            score = float(np.mean(importance[i: i + k]))
            results.append({"position": i, "kmer": kmer, "score": round(score, 6)})
        results.sort(key=lambda x: -x["score"])
        return results[:top_n]

    def extract_population_kmers(
        self,
        avg_importance: np.ndarray,
        consensus_sequence: np.ndarray,
        k: int = 5,
        top_n: int = 5,
    ) -> list:
        """
        Extract top k-mers from the population-averaged importance vector.
        Uses the most common base at each position as the consensus sequence.

        Parameters
        ----------
        avg_importance : np.ndarray, shape (max_len,)
        consensus_sequence : np.ndarray, shape (max_len,)
            Integer-encoded consensus (e.g., most frequent base per position).

        Returns
        -------
        list of dict with keys: position, kmer, score, cosmic_hits
        """
        kmers = self.extract_top_kmers(avg_importance, consensus_sequence, k=k, top_n=top_n)
        for entry in kmers:
            entry["cosmic_hits"] = self.cosmic_lookup(entry["kmer"])
        return kmers

    # ── COSMIC cross-reference ────────────────────────────

    @staticmethod
    def cosmic_lookup(kmer: str) -> list:
        """
        Check whether a k-mer contains any COSMIC SBS trinucleotide context.

        Scans every 3-mer sub-window in `kmer` against the COSMIC_SIGNATURES
        table. Returns a list of matching signature dicts.

        Returns
        -------
        list of dict: [{"signature": str, "description": str, "context": str}, ...]
        Empty list if no match.
        """
        hits = []
        kmer_upper = kmer.upper()
        for sig_id, sig_desc, patterns in COSMIC_SIGNATURES:
            for pattern in patterns:
                if pattern in kmer_upper:
                    hits.append({
                        "signature": sig_id,
                        "description": sig_desc,
                        "context": pattern,
                    })
                    break  # one hit per signature is enough
        return hits

    # ── Full analysis pipeline ────────────────────────────

    def run_full_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int = 500,
        k: int = 5,
        top_kmers: int = 5,
        confidence_threshold: float = 0.6,
        save_dir: str = None,
    ) -> dict:
        """
        End-to-end occlusion sensitivity analysis:
          1. Aggregate importance across n_samples high-confidence cancer reads
          2. Build importance matrix for heatmap
          3. Extract top k-mers from population average
          4. Cross-reference with COSMIC signatures
          5. Save heatmap + motifs.json

        Returns
        -------
        dict with keys:
          avg_importance  : np.ndarray
          importance_matrix : np.ndarray
          top_motifs      : list of dicts (position, kmer, score, cosmic_hits)
          n_reads_used    : int
        """
        import os

        X_cancer = X[y == 1]

        print(f"Cancer reads available : {len(X_cancer):,}")
        print(f"Analysing top {n_samples} reads (confidence ≥ {confidence_threshold:.0%})...")

        # ── Step 1: predict probabilities on candidate pool ──
        pool = X_cancer[:n_samples]
        probs = self.model.predict(pool, batch_size=512, verbose=0).flatten()

        high_idx = np.where(probs >= confidence_threshold)[0]
        if len(high_idx) == 0:
            print("  No reads above threshold — falling back to top-10%")
            high_idx = probs.argsort()[-max(1, n_samples // 10):][::-1]

        print(f"  {len(high_idx)} reads qualify for analysis")

        # ── Step 2: compute importance per read ──
        importance_rows = []
        for idx in high_idx:
            imp, _ = self.explain_read(pool[idx])
            importance_rows.append(imp)

        importance_matrix = np.array(importance_rows)   # (n_reads, max_len)
        avg_importance    = importance_matrix.mean(axis=0)
        if avg_importance.max() > 0:
            avg_importance = avg_importance / avg_importance.max()

        # ── Step 3: consensus sequence (most common base per position) ──
        consensus = np.array([
            np.bincount(pool[high_idx, pos].astype(int),
                        minlength=6).argmax()
            for pos in range(pool.shape[1])
        ])

        # ── Step 4: extract k-mers + COSMIC ──
        motifs = self.extract_population_kmers(
            avg_importance, consensus, k=k, top_n=top_kmers
        )

        # ── Step 5: save outputs ──
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.plot_heatmap(
                importance_matrix,
                save_path=os.path.join(save_dir, "occlusion_heatmap.png"),
            )
            self.plot_aggregated_importance(
                avg_importance,
                save_path=os.path.join(save_dir, "aggregated_importance.png"),
            )
            motifs_path = os.path.join(save_dir, "motifs.json")
            with open(motifs_path, "w") as f:
                json.dump({"n_reads_analysed": len(high_idx),
                           "k": k,
                           "top_motifs": motifs}, f, indent=2)
            print(f"Saved motifs: {motifs_path}")

        print("\nTop motifs:")
        for m in motifs:
            cosmic_str = ", ".join(h["signature"] for h in m["cosmic_hits"]) or "none"
            print(f"  pos {m['position']:>3}  {m['kmer']}  "
                  f"score={m['score']:.4f}  COSMIC={cosmic_str}")

        return {
            "avg_importance":    avg_importance,
            "importance_matrix": importance_matrix,
            "top_motifs":        motifs,
            "n_reads_used":      len(high_idx),
        }


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
    X, y, _run_ids = DataLoader.load_all_batches(args.batch_dir)
    X_cancer = X[y == 1]
    print(f"Cancer reads: {len(X_cancer):,}")

    analyzer = OcclusionAnalyzer(model)

    print(f"\nPlotting top {args.top_n} cancer reads...")
    analyzer.plot_top_cancer_reads(X_cancer, top_n=args.top_n, save_dir=args.save_dir)

    print(f"\nComputing aggregated importance ({args.n_agg} reads)...")
    avg_imp = analyzer.aggregate_importance(X_cancer, n_samples=args.n_agg)
    save_path = f"{args.save_dir}/aggregated_importance.png" if args.save_dir else None
    analyzer.plot_aggregated_importance(avg_imp, save_path=save_path)
