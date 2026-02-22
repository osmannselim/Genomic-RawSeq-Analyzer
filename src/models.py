"""
models.py
---------
Model definitions for the Genomic-RawSeq-Analyzer pipeline.

Contains:
  - build_cnn()              : Supervised 1D-CNN for tumor vs normal classification
  - build_lstm_autoencoder() : Unsupervised LSTM Autoencoder for anomaly detection

Semester 1 results:
  - CNN AUC:               0.615  (read-level)
  - LSTM Autoencoder AUC:  0.509  (failed — cancer ≠ statistical anomaly)

Semester 2 plan:
  - Replace CNN with DNABERT-2 fine-tuned on our dataset (see Issue #3)

Usage:
    from models import build_cnn, build_lstm_autoencoder

    model = build_cnn(max_len=80)
    autoencoder, encoder = build_lstm_autoencoder(seq_len=80)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D,
    Dense, Dropout, LSTM, RepeatVector, TimeDistributed,
)


# ─────────────────────────────────────────────────────────
# Model 1: Supervised 1D-CNN
# ─────────────────────────────────────────────────────────

def build_cnn(
    max_len: int = 80,
    vocab_size: int = 6,       # 0=pad, 1=A, 2=C, 3=G, 4=T, 5=N
    embed_dim: int = 16,
    filters_1: int = 64,
    filters_2: int = 128,
    kernel_1: int = 5,
    kernel_2: int = 3,
    dense_units: int = 64,
    dropout_rate: float = 0.5,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Build and compile the baseline 1D-CNN model.

    Architecture
    ------------
    Embedding (integer DNA → dense vectors)
    → Conv1D (detect 5-mer motifs)  → MaxPool
    → Conv1D (higher-level features) → GlobalMaxPool
    → Dense + Dropout
    → Sigmoid output (cancer probability)

    The Embedding layer treats each nucleotide (0–5) as a token and maps
    it to a learnable 16-dimensional vector, similar to word embeddings in NLP.
    Conv1D then scans for mutation-signature k-mer patterns across the sequence.

    Parameters
    ----------
    max_len : int
        Input sequence length (80 bp in Semester 1).
    vocab_size : int
        Number of distinct integer tokens (6: pad + 5 bases).
    embed_dim : int
        Dimensionality of nucleotide embeddings.
    filters_1, filters_2 : int
        Number of convolutional filters per layer.
    kernel_1, kernel_2 : int
        Convolutional kernel (motif window) sizes.
    dense_units : int
        Units in the fully-connected layer.
    dropout_rate : float
        Dropout probability for regularisation.
    learning_rate : float
        Adam optimizer learning rate.

    Returns
    -------
    model : tf.keras.Model (compiled, ready to train)
    """
    model = models.Sequential(
        [
            # Integer DNA → dense embedding vectors
            Embedding(
                input_dim=vocab_size,
                output_dim=embed_dim,
                input_length=max_len,
                name="nucleotide_embedding",
            ),
            # 1st conv: detect 5-mer motifs
            Conv1D(filters=filters_1, kernel_size=kernel_1, activation="relu", name="conv1_motif"),
            MaxPooling1D(pool_size=2, name="maxpool1"),
            # 2nd conv: compose higher-level features
            Conv1D(filters=filters_2, kernel_size=kernel_2, activation="relu", name="conv2_features"),
            GlobalMaxPooling1D(name="global_maxpool"),
            # Classification head
            Dense(dense_units, activation="relu", name="dense"),
            Dropout(dropout_rate, name="dropout"),
            Dense(1, activation="sigmoid", name="output"),
        ],
        name="1D_CNN_Baseline",
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ─────────────────────────────────────────────────────────
# Model 2: Unsupervised LSTM Autoencoder
# ─────────────────────────────────────────────────────────

def build_lstm_autoencoder(
    seq_len: int = 80,
    encoder_units: tuple = (64, 32),
    decoder_units: tuple = (32, 64),
    learning_rate: float = 1e-3,
    clip_norm: float = 1.0,
) -> tuple:
    """
    Build and compile an LSTM Autoencoder for unsupervised anomaly detection.

    Training scheme: train exclusively on Normal sequences.
    At inference, cancer reads are expected to have higher reconstruction error
    (MSE) because the encoder has never seen their mutation patterns.

    NOTE: Semester 1 result — AUC 0.509 (failed).
    Cancer and normal genomes are >99.9% identical at read level,
    so the model cannot learn a meaningful difference in reconstruction error.
    This architecture is retained for documentation purposes.

    Architecture
    ------------
    Input (seq_len, 1)
    → LSTM Encoder  [64 → 32 units, tanh]   (bottleneck z)
    → RepeatVector  (tile z seq_len times)
    → LSTM Decoder  [32 → 64 units, tanh]
    → TimeDistributed Dense (1)  → Reconstruction

    tanh activation is used throughout (not ReLU) to avoid NaN loss
    from unbounded activations over integer sequences.
    clipnorm=1.0 added as a safety measure against gradient explosion.

    Parameters
    ----------
    seq_len : int
        Sequence length (must match X.shape[1]).
    encoder_units : tuple
        Number of units in each LSTM encoder layer.
    decoder_units : tuple
        Number of units in each LSTM decoder layer.
    learning_rate : float
        Adam optimizer learning rate.
    clip_norm : float
        Gradient clipping norm.

    Returns
    -------
    autoencoder : tf.keras.Model (compiled, input → reconstruction)
    encoder     : tf.keras.Model (input → bottleneck embedding)
    """
    inputs = Input(shape=(seq_len, 1), name="sequence_input")

    # ── Encoder ──
    x = LSTM(encoder_units[0], activation="tanh", return_sequences=True, name="enc_lstm_1")(inputs)
    encoded = LSTM(encoder_units[1], activation="tanh", return_sequences=False, name="enc_lstm_2")(x)

    # ── Bottleneck → Repeat ──
    repeated = RepeatVector(seq_len, name="repeat")(encoded)

    # ── Decoder ──
    x = LSTM(decoder_units[0], activation="tanh", return_sequences=True, name="dec_lstm_1")(repeated)
    x = LSTM(decoder_units[1], activation="tanh", return_sequences=True, name="dec_lstm_2")(x)
    decoded = TimeDistributed(Dense(1), name="reconstruction")(x)

    # ── Models ──
    autoencoder = Model(inputs, decoded, name="LSTM_Autoencoder")
    encoder = Model(inputs, encoded, name="LSTM_Encoder")

    autoencoder.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate, clipnorm=clip_norm),
        loss="mse",
    )

    return autoencoder, encoder


# ─────────────────────────────────────────────────────────
# Convenience: model summary
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Model 1: 1D-CNN (Supervised)")
    print("=" * 60)
    cnn = build_cnn()
    cnn.summary()

    print("\n" + "=" * 60)
    print("Model 2: LSTM Autoencoder (Unsupervised)")
    print("=" * 60)
    ae, enc = build_lstm_autoencoder()
    ae.summary()
    print(f"\nEncoder output shape: {enc.output_shape}")
