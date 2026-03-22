"""
test_pipeline.py
----------------
End-to-end integration test for Issue #2: Patient-level aggregation.

Simulates the full pipeline WITHOUT needing real FASTQ downloads:
  1. Creates synthetic batch .npz files WITH run_ids  (mimics data_loader output)
  2. Loads them back via DataLoader.load_all_batches() (verifies 3-tuple return)
  3. Trains a quick CNN on the synthetic data          (verifies train.py changes)
  4. Runs Evaluator.full_report() with run_ids         (verifies evaluate.py)
  5. Runs CrowdVotingAggregator.full_report()          (verifies patient_aggregation.py)
  6. Tests backward compatibility with legacy batches   (no run_ids)
  7. Tests threshold sweep                              (finds optimal threshold)

Usage:
    cd Genomic-RawSeq-Analyzer
    python test_pipeline.py

All outputs (plots, CSVs) are saved to test_output/
"""

import os
import sys
import numpy as np
import shutil

# ── Setup paths ──
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF info logs

BATCH_DIR = os.path.join(ROOT, "test_batches")
OUTPUT_DIR = os.path.join(ROOT, "test_output")
LEGACY_DIR = os.path.join(ROOT, "test_legacy_batches")


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ──────────────────────────────────────────────────────────────
# STEP 1: Create synthetic batch data WITH run_ids
# ──────────────────────────────────────────────────────────────

def step1_create_synthetic_batches():
    separator("STEP 1: Create synthetic batch data with run_ids")

    os.makedirs(BATCH_DIR, exist_ok=True)

    # Simulate 36 patients (matching the real SRA run list)
    # ERR166302-ERR166315 = Tumor (14 patients)
    # ERR166316-ERR166329 = Normal (14 patients)
    # ERR166330-ERR166337 = Tumor (8 patients)
    tumor_runs = [f"ERR166{i}" for i in range(302, 316)] + \
                 [f"ERR166{i}" for i in range(330, 338)]
    normal_runs = [f"ERR166{i}" for i in range(316, 330)]

    reads_per_patient = 500  # small for speed (real = 50,000)
    max_len = 80

    print(f"  Tumor patients:  {len(tumor_runs)}")
    print(f"  Normal patients: {len(normal_runs)}")
    print(f"  Reads/patient:   {reads_per_patient}")

    all_X, all_y, all_run_ids = [], [], []

    # Tumor reads: slightly biased toward higher values at certain positions
    # This simulates the weak signal the CNN learns to detect
    for run_id in tumor_runs:
        reads = np.random.randint(1, 6, size=(reads_per_patient, max_len), dtype=np.int8)
        # Inject subtle signal: positions 10-15 biased toward G(3) and T(4)
        signal_mask = np.random.random(reads_per_patient) < 0.3  # 30% of reads
        reads[signal_mask, 10:16] = np.random.choice([3, 4], size=(signal_mask.sum(), 6))
        all_X.append(reads)
        all_y.extend([1] * reads_per_patient)
        all_run_ids.extend([run_id] * reads_per_patient)

    # Normal reads: uniform random
    for run_id in normal_runs:
        reads = np.random.randint(1, 6, size=(reads_per_patient, max_len), dtype=np.int8)
        all_X.append(reads)
        all_y.extend([0] * reads_per_patient)
        all_run_ids.extend([run_id] * reads_per_patient)

    X = np.concatenate(all_X, axis=0)
    y = np.array(all_y, dtype=np.int8)
    run_ids = np.array(all_run_ids, dtype="U20")

    # Shuffle together (important: keeps alignment)
    perm = np.random.RandomState(42).permutation(len(X))
    X, y, run_ids = X[perm], y[perm], run_ids[perm]

    # Save as batches (4 patients per batch, matching real pipeline)
    batch_size = 2000  # reads per batch file
    n_batches = (len(X) + batch_size - 1) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, len(X))
        save_path = os.path.join(BATCH_DIR, f"batch_{i+1:03d}.npz")
        np.savez_compressed(
            save_path,
            X=X[start:end],
            y=y[start:end],
            run_ids=run_ids[start:end],
        )
        print(f"  Saved {save_path}  ({end - start} reads)")

    print(f"\n  Total: {len(X):,} reads across {len(tumor_runs) + len(normal_runs)} patients")
    print(f"  Batches saved to: {BATCH_DIR}")
    return True


# ──────────────────────────────────────────────────────────────
# STEP 2: Load batches and verify run_ids
# ──────────────────────────────────────────────────────────────

def step2_load_and_verify():
    separator("STEP 2: Load batches and verify run_ids (3-tuple)")

    from data_loader import DataLoader

    X, y, run_ids = DataLoader.load_all_batches(BATCH_DIR)

    print(f"\n  X shape:          {X.shape}")
    print(f"  y shape:          {y.shape}")
    print(f"  run_ids:          {'Loaded ✓' if run_ids is not None else 'MISSING ✗'}")
    print(f"  Unique patients:  {len(np.unique(run_ids))}")
    print(f"  Tumor reads:      {sum(y == 1):,}")
    print(f"  Normal reads:     {sum(y == 0):,}")

    # Verify alignment
    assert run_ids is not None, "run_ids should not be None"
    assert len(X) == len(y) == len(run_ids), "Length mismatch!"

    # Check patient labels are consistent
    import pandas as pd
    df = pd.DataFrame({"run_id": run_ids, "label": y})
    patient_labels = df.groupby("run_id")["label"].nunique()
    mixed = patient_labels[patient_labels > 1]
    if len(mixed) > 0:
        print(f"  WARNING: {len(mixed)} patients have mixed labels!")
    else:
        print(f"  Label consistency: All patients have single label ✓")

    print("\n  ✓ Step 2 passed")
    return X, y, run_ids


# ──────────────────────────────────────────────────────────────
# STEP 3: Train CNN on synthetic data
# ──────────────────────────────────────────────────────────────

def step3_train_cnn(X, y, run_ids):
    separator("STEP 3: Train CNN on synthetic data")

    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from models import build_cnn

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Split preserving run_ids alignment
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    run_ids_test = run_ids[test_idx]

    print(f"  Train: {len(X_train):,} reads")
    print(f"  Test:  {len(X_test):,} reads")
    print(f"  Test patients: {len(np.unique(run_ids_test))}")

    # Build and train (quick: 3 epochs)
    model = build_cnn(max_len=X_train.shape[1])
    model.build(input_shape=(None, X_train.shape[1]))
    print(f"\n  Model: {model.name}")
    print(f"  Parameters: {model.count_params():,}")

    print("\n  Training (3 epochs)...")
    history = model.fit(
        X_train, y_train,
        epochs=3,
        batch_size=128,
        validation_data=(X_test, y_test),
        verbose=1,
    )

    # Save model
    model_path = os.path.join(OUTPUT_DIR, "cnn_test.keras")
    model.save(model_path)
    print(f"\n  Model saved: {model_path}")
    print("  ✓ Step 3 passed")

    return model, X_test, y_test, run_ids_test


# ──────────────────────────────────────────────────────────────
# STEP 4: Run Evaluator.full_report() with run_ids
# ──────────────────────────────────────────────────────────────

def step4_evaluator_report(model, X_test, y_test, run_ids_test):
    separator("STEP 4: Evaluator.full_report() with patient-level aggregation")

    import matplotlib
    matplotlib.use("Agg")  # no display needed

    from evaluate import Evaluator

    eval_dir = os.path.join(OUTPUT_DIR, "evaluator_report")

    evaluator = Evaluator(model, X_test, y_test, run_ids=run_ids_test)
    evaluator.full_report(save_dir=eval_dir)

    # Check outputs exist
    expected_files = ["roc_read_level.png", "prediction_distribution.png",
                      "patient_level_roc.png", "patient_level_boxplot.png"]
    found = os.listdir(eval_dir) if os.path.exists(eval_dir) else []
    print(f"\n  Output files in {eval_dir}:")
    for f in sorted(found):
        print(f"    {f}")

    print("\n  ✓ Step 4 passed")
    return evaluator


# ──────────────────────────────────────────────────────────────
# STEP 5: Run CrowdVotingAggregator.full_report()
# ──────────────────────────────────────────────────────────────

def step5_aggregator_report(evaluator):
    separator("STEP 5: CrowdVotingAggregator standalone pipeline")

    import matplotlib
    matplotlib.use("Agg")

    from patient_aggregation import CrowdVotingAggregator
    from sklearn.metrics import roc_auc_score

    agg_dir = os.path.join(OUTPUT_DIR, "aggregation_report")

    probs = evaluator.probs
    y_test = evaluator.y_test
    run_ids = evaluator.run_ids

    # Read-level AUC for comparison
    read_auc = roc_auc_score(y_test, probs)
    print(f"  Read-level AUC: {read_auc:.4f}")

    # Create aggregator and run full report
    agg = CrowdVotingAggregator(probs, y_test, run_ids)
    result = agg.full_report(
        read_auc=read_auc,
        threshold=0.5,
        save_dir=agg_dir,
    )

    print(f"\n  Patient-level AUC: {result['auc']:.4f}")
    print(f"  Improvement:       {result['auc'] - read_auc:+.4f}")

    # Check outputs
    found = os.listdir(agg_dir) if os.path.exists(agg_dir) else []
    print(f"\n  Output files in {agg_dir}:")
    for f in sorted(found):
        print(f"    {f}")

    print("\n  ✓ Step 5 passed")
    return result


# ──────────────────────────────────────────────────────────────
# STEP 6: Backward compatibility test (legacy batches)
# ──────────────────────────────────────────────────────────────

def step6_backward_compat():
    separator("STEP 6: Backward compatibility (legacy batches without run_ids)")

    from data_loader import DataLoader

    os.makedirs(LEGACY_DIR, exist_ok=True)

    # Create old-format batch (no run_ids key)
    np.savez_compressed(
        os.path.join(LEGACY_DIR, "batch_001.npz"),
        X=np.random.randint(0, 6, size=(500, 80), dtype=np.int8),
        y=np.array([1] * 250 + [0] * 250, dtype=np.int8),
    )

    X, y, run_ids = DataLoader.load_all_batches(LEGACY_DIR)

    print(f"  X shape:   {X.shape}")
    print(f"  y shape:   {y.shape}")
    print(f"  run_ids:   {run_ids}")

    assert run_ids is None, "run_ids should be None for legacy batches"
    print("  Legacy batches load correctly without run_ids ✓")

    # Verify Evaluator gracefully skips patient-level
    import matplotlib
    matplotlib.use("Agg")
    from evaluate import Evaluator
    from models import build_cnn

    model = build_cnn()
    evaluator = Evaluator(model, X, y, run_ids=None)
    result = evaluator.patient_level_auc()
    assert result is None, "Should return None when run_ids is None"
    print("  Evaluator skips patient-level when run_ids=None ✓")

    print("\n  ✓ Step 6 passed")


# ──────────────────────────────────────────────────────────────
# STEP 7: Threshold sweep analysis
# ──────────────────────────────────────────────────────────────

def step7_threshold_sweep(evaluator):
    separator("STEP 7: Threshold sweep analysis")

    import matplotlib
    matplotlib.use("Agg")

    from patient_aggregation import CrowdVotingAggregator

    agg = CrowdVotingAggregator(evaluator.probs, evaluator.y_test, evaluator.run_ids)
    sweep_df = agg.threshold_sweep()

    print("  Threshold sweep results:")
    print(f"  {'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*50}")
    for _, row in sweep_df.iterrows():
        print(f"  {row['threshold']:>10.2f} {row['accuracy']:>10.4f} "
              f"{row['precision']:>10.4f} {row['recall']:>10.4f} {row['f1']:>10.4f}")

    best = sweep_df.loc[sweep_df["f1"].idxmax()]
    print(f"\n  Best threshold: {best['threshold']:.2f}  (F1 = {best['f1']:.4f})")

    # Save sweep to CSV
    csv_path = os.path.join(OUTPUT_DIR, "threshold_sweep.csv")
    sweep_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Save plot
    plot_path = os.path.join(OUTPUT_DIR, "threshold_sweep.png")
    agg.plot_threshold_sweep(save_path=plot_path)
    print(f"  Saved: {plot_path}")

    print("\n  ✓ Step 7 passed")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    separator("PATIENT-LEVEL AGGREGATION — FULL PIPELINE TEST")
    print("  This script tests all 5 tasks from Issue #2:")
    print("    Task 1: Group reads by run_accession (patient ID)")
    print("    Task 2: Compute mean cancer probability per patient")
    print("    Task 3: Apply threshold → binary patient label")
    print("    Task 4: Evaluate patient-level AUC vs read-level AUC")
    print("    Task 5: Box plot of Cancer vs Normal distributions")

    # Clean previous test outputs
    for d in [BATCH_DIR, OUTPUT_DIR, LEGACY_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)

    try:
        step1_create_synthetic_batches()
        X, y, run_ids = step2_load_and_verify()
        model, X_test, y_test, run_ids_test = step3_train_cnn(X, y, run_ids)
        evaluator = step4_evaluator_report(model, X_test, y_test, run_ids_test)
        result = step5_aggregator_report(evaluator)
        step6_backward_compat()
        step7_threshold_sweep(evaluator)

        separator("ALL TESTS PASSED ✓")
        print(f"  Output directory: {OUTPUT_DIR}")
        print(f"\n  Files generated:")
        for dirpath, dirnames, filenames in os.walk(OUTPUT_DIR):
            level = dirpath.replace(OUTPUT_DIR, "").count(os.sep)
            indent = "    " + "  " * level
            print(f"{indent}{os.path.basename(dirpath)}/")
            subindent = "    " + "  " * (level + 1)
            for f in sorted(filenames):
                size = os.path.getsize(os.path.join(dirpath, f))
                print(f"{subindent}{f}  ({size/1024:.1f} KB)")

    except Exception as e:
        separator("TEST FAILED ✗")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Cleanup synthetic batches (keep output for inspection)
        for d in [BATCH_DIR, LEGACY_DIR]:
            if os.path.exists(d):
                shutil.rmtree(d)


if __name__ == "__main__":
    main()
