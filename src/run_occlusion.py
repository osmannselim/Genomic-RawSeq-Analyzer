"""
run_occlusion.py
----------------
CLI wrapper for aggregated occlusion sensitivity analysis.

Runs the full OcclusionAnalyzer pipeline on high-confidence cancer reads,
produces the population heatmap, and saves motifs.json for downstream
use by the LLM report generator.

Usage:
    python src/run_occlusion.py \
        --model_path results/cnn_baseline.keras \
        --batch_dir  results/batches \
        --save_dir   results/occlusion \
        --n_samples  500 \
        --k          5 \
        --top_kmers  5
"""

import os
import sys
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from tensorflow.keras.models import load_model
from data_loader import DataLoader
from explainability import OcclusionAnalyzer


def parse_args():
    p = argparse.ArgumentParser(description="Aggregated occlusion sensitivity analysis.")
    p.add_argument("--model_path", default="results/cnn_baseline.keras")
    p.add_argument("--batch_dir",  default="results/batches")
    p.add_argument("--save_dir",   default="results/occlusion")
    p.add_argument("--n_samples",  type=int,   default=500,
                   help="Cancer reads to analyse (first N from pool)")
    p.add_argument("--k",          type=int,   default=5,
                   help="K-mer length for motif extraction")
    p.add_argument("--top_kmers",  type=int,   default=5,
                   help="Number of top motifs to report")
    p.add_argument("--threshold",  type=float, default=0.6,
                   help="Minimum cancer confidence to include a read")
    p.add_argument("--top_n_reads", type=int,  default=3,
                   help="Number of individual saliency maps to save")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("OCCLUSION SENSITIVITY ANALYSIS")
    print("=" * 60)

    print(f"\nLoading model: {args.model_path}")
    model = load_model(args.model_path)

    print("Loading data...")
    X, y, run_ids = DataLoader.load_all_batches(args.batch_dir)
    X_cancer = X[y == 1]
    print(f"Total cancer reads: {len(X_cancer):,}")

    analyzer = OcclusionAnalyzer(model)

    # ── Individual saliency maps for top-N most confident reads ──
    print(f"\nGenerating top-{args.top_n_reads} individual saliency maps...")
    analyzer.plot_top_cancer_reads(
        X_cancer,
        top_n=args.top_n_reads,
        sample_size=2000,
        save_dir=os.path.join(args.save_dir, "saliency_maps"),
    )

    # ── Population-level aggregation + heatmap + motifs.json ──
    print(f"\nRunning population-level analysis ({args.n_samples} reads)...")
    results = analyzer.run_full_analysis(
        X, y,
        n_samples=args.n_samples,
        k=args.k,
        top_kmers=args.top_kmers,
        confidence_threshold=args.threshold,
        save_dir=args.save_dir,
    )

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Reads analysed    : {results['n_reads_used']}")
    print(f"K-mer length      : {args.k}")
    print(f"Outputs saved to  : {args.save_dir}/")
    print(f"  occlusion_heatmap.png")
    print(f"  aggregated_importance.png")
    print(f"  motifs.json")
    print(f"\nTop motifs (for LLM report):")
    for m in results["top_motifs"]:
        cosmic = ", ".join(h["signature"] for h in m["cosmic_hits"]) or "none"
        print(f"  pos {m['position']:>3}–{m['position']+args.k-1:<3}  "
              f"{m['kmer']}  score={m['score']:.4f}  COSMIC={cosmic}")


if __name__ == "__main__":
    main()
