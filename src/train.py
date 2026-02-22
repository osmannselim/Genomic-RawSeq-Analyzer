"""
train.py
--------
Main training script for the Genomic-RawSeq-Analyzer pipeline.

Trains either the supervised 1D-CNN or the unsupervised LSTM Autoencoder,
then runs full evaluation and saves the model to disk.

Usage
-----
# Train CNN (supervised):
python train.py --model cnn --batch_dir /path/to/batches --output_dir /path/to/output

# Train Autoencoder (unsupervised):
python train.py --model autoencoder --normal_dir /path/normal --cancer_dir /path/cancer --output_dir /path/to/output
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from data_loader import DataLoader
from models import build_cnn, build_lstm_autoencoder
from evaluate import Evaluator


# ─────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────

def train_cnn(batch_dir: str, output_dir: str, epochs: int = 5):
    """Train the supervised 1D-CNN."""
    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ──
    print("Loading batch data...")
    X, y = DataLoader.load_all_batches(batch_dir)

    print(f"\nDataset stats:")
    print(f"  Total reads : {len(X):,}")
    print(f"  Tumor       : {sum(y == 1):,}  ({sum(y == 1) / len(y) * 100:.1f}%)")
    print(f"  Normal      : {sum(y == 0):,}  ({sum(y == 0) / len(y) * 100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    del X, y   # free RAM
    import gc; gc.collect()

    # ── Build & train ──
    print("\nBuilding model...")
    model = build_cnn(max_len=X_train.shape[1])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, "cnn_checkpoint.keras"),
            save_best_only=True, monitor="val_loss", verbose=1
        ),
    ]

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=128,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
    )

    # ── Save model ──
    model_path = os.path.join(output_dir, "cnn_baseline.keras")
    model.save(model_path)
    print(f"\nModel saved: {model_path}")

    # ── Plot training history ──
    _plot_history(history, output_dir)

    # ── Evaluate ──
    evaluator = Evaluator(model, X_test, y_test)
    evaluator.full_report(save_dir=os.path.join(output_dir, "evaluation"))

    return model


def train_autoencoder(
    normal_dir: str,
    cancer_dir: str,
    output_dir: str,
    epochs: int = 10,
):
    """Train the unsupervised LSTM Autoencoder on Normal sequences only."""
    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ──
    print("Loading data...")
    X_normal, X_cancer = DataLoader.load_unsupervised(normal_dir, cancer_dir)

    X_train, X_val = train_test_split(X_normal, test_size=0.15, random_state=42)

    # ── Build & train ──
    print("\nBuilding LSTM Autoencoder...")
    autoencoder, encoder = build_lstm_autoencoder(seq_len=X_train.shape[1])
    autoencoder.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, "ae_checkpoint.keras"),
            save_best_only=True, monitor="val_loss", verbose=1
        ),
    ]

    print("\nTraining (Normal data only)...")
    history = autoencoder.fit(
        X_train, X_train,    # Input = Target (reconstruction)
        epochs=epochs,
        batch_size=128,
        validation_data=(X_val, X_val),
        callbacks=callbacks,
        shuffle=True,
    )

    # ── Save ──
    ae_path  = os.path.join(output_dir, "lstm_autoencoder.keras")
    enc_path = os.path.join(output_dir, "lstm_encoder.keras")
    autoencoder.save(ae_path)
    encoder.save(enc_path)
    print(f"Saved: {ae_path}")
    print(f"Saved: {enc_path}")

    # ── Plot training history ──
    _plot_history(history, output_dir)

    # ── Evaluate anomaly detection ──
    _evaluate_autoencoder(autoencoder, X_val, X_cancer, output_dir)

    return autoencoder, encoder


# ─────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────

def _plot_history(history, output_dir: str):
    """Save training/validation loss (and accuracy for CNN) plots."""
    metrics = list(history.history.keys())
    train_metrics = [m for m in metrics if not m.startswith("val_")]

    n = len(train_metrics)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, train_metrics):
        ax.plot(history.history[metric],     label="Train")
        ax.plot(history.history[f"val_{metric}"], label="Val")
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved training history: {path}")
    plt.show()


def _evaluate_autoencoder(autoencoder, X_normal_val, X_cancer, output_dir: str):
    """Compute reconstruction errors and ROC-AUC for the autoencoder."""
    from sklearn.metrics import roc_curve, auc as sk_auc

    print("\nRunning autoencoder evaluation...")

    pred_normal = autoencoder.predict(X_normal_val, batch_size=512, verbose=1)
    pred_cancer = autoencoder.predict(X_cancer,     batch_size=512, verbose=1)

    mse_normal = np.mean(np.square(X_normal_val - pred_normal), axis=(1, 2))
    mse_cancer = np.mean(np.square(X_cancer     - pred_cancer), axis=(1, 2))

    print(f"  Mean MSE (Normal): {np.mean(mse_normal):.5f}")
    print(f"  Mean MSE (Cancer): {np.mean(mse_cancer):.5f}")

    y_true   = np.concatenate([np.zeros(len(mse_normal)), np.ones(len(mse_cancer))])
    y_scores = np.concatenate([mse_normal, mse_cancer])

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc     = sk_auc(fpr, tpr)
    print(f"  Anomaly Detection AUC: {roc_auc:.4f}")

    # Distribution plot
    plt.figure(figsize=(9, 5))
    plt.hist(mse_normal, bins=80, alpha=0.55, density=True, color="steelblue", label="Normal")
    plt.hist(mse_cancer, bins=80, alpha=0.55, density=True, color="crimson",   label="Cancer")
    plt.title(f"Reconstruction Error Distribution  (AUC={roc_auc:.4f})")
    plt.xlabel("MSE")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ae_reconstruction_error.png"), dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train genomic cancer detection model.")

    parser.add_argument(
        "--model", required=True, choices=["cnn", "autoencoder"],
        help="Which model to train",
    )
    parser.add_argument("--output_dir", required=True, help="Where to save model + results")
    parser.add_argument("--epochs", type=int, default=5)

    # CNN args
    parser.add_argument("--batch_dir",  default=None, help="[CNN] Directory with batch_*.npz files")

    # Autoencoder args
    parser.add_argument("--normal_dir", default=None, help="[AE] Directory with normal .npy files")
    parser.add_argument("--cancer_dir", default=None, help="[AE] Directory with cancer .npy files")

    args = parser.parse_args()

    if args.model == "cnn":
        if not args.batch_dir:
            parser.error("--batch_dir required for CNN training")
        train_cnn(args.batch_dir, args.output_dir, epochs=args.epochs)

    elif args.model == "autoencoder":
        if not args.normal_dir or not args.cancer_dir:
            parser.error("--normal_dir and --cancer_dir required for autoencoder training")
        train_autoencoder(args.normal_dir, args.cancer_dir, args.output_dir, epochs=args.epochs)
