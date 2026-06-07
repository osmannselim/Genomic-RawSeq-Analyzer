"""
dnabert2_model.py
-----------------
DNABERT-2 fine-tuning module for Tumor/Normal read classification.

Replaces the 1D-CNN + integer encoding baseline with:
  - BPE tokenization via DNABERT-2's AutoTokenizer
  - DNABERT-2 (zhihan1996/DNABERT-2-117M) as the sequence encoder
  - Linear classification head on the [CLS] token output

Requirements:
    pip install transformers>=4.40 torch>=2.2 accelerate>=0.28

Usage:
    from dnabert2_model import DNABERT2Classifier, DNADataset, decode_batch_to_sequences
    from data_loader import DataLoader

    # Load existing integer-encoded batches and decode to raw strings
    X_int, y, run_ids = DataLoader.load_all_batches("results/batches")
    sequences = decode_batch_to_sequences(X_int)

    # Build model + tokenize
    model = DNABERT2Classifier()
    dataset = DNADataset(sequences, y, model.tokenizer, max_length=512)
"""

import os
import logging
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_NAME = "zhihan1996/DNABERT-2-117M"

# Reverse of the integer encoding in data_loader.py
_INT_TO_BASE = {0: "", 1: "A", 2: "C", 3: "G", 4: "T", 5: "N"}


# ── Data utilities ─────────────────────────────────────────────────────────────

def decode_batch_to_sequences(X_int: np.ndarray) -> list:
    """
    Decode integer-encoded reads (from existing .npz batches) back to raw
    DNA strings for DNABERT-2 tokenization.

    Parameters
    ----------
    X_int : np.ndarray, shape (N, read_len)
        Integer-encoded reads (A=1, C=2, G=3, T=4, N=5, padding=0).

    Returns
    -------
    list of str, length N
        Raw nucleotide strings with padding stripped.
    """
    sequences = []
    for row in X_int:
        seq = "".join(_INT_TO_BASE[int(b)] for b in row)
        sequences.append(seq)
    return sequences


class DNADataset(Dataset):
    """
    PyTorch Dataset that tokenizes raw DNA strings with the DNABERT-2 tokenizer.

    Parameters
    ----------
    sequences : list of str
        Raw nucleotide strings (e.g. "ACGTNNACGT...").
    labels : array-like of int
        Binary labels (0 = Normal, 1 = Tumor).
    tokenizer : AutoTokenizer
        Pre-loaded DNABERT-2 tokenizer.
    max_length : int
        Maximum token length (BPE tokens, not base-pairs). Default: 512.
    """

    def __init__(
        self,
        sequences: list,
        labels,
        tokenizer,
        max_length: int = 512,
    ):
        self.sequences = sequences
        self.labels = np.array(labels, dtype=np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            seq,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float32),
        }


# ── Model ──────────────────────────────────────────────────────────────────────

class DNABERT2Classifier(nn.Module):
    """
    Binary classifier built on top of DNABERT-2.

    Architecture:
        DNABERT-2 encoder (frozen or fine-tuned)
            ↓
        [CLS] token hidden state  (hidden_dim = 768)
            ↓
        Dropout(dropout_rate)
            ↓
        Linear(768 → 1)
            ↓
        Sigmoid → cancer probability

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Default: "zhihan1996/DNABERT-2-117M".
    dropout_rate : float
        Dropout before classification head. Default: 0.1.
    freeze_encoder : bool
        If True, freeze all DNABERT-2 weights and only train the head.
        Useful for very small datasets or quick probing experiments.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        dropout_rate: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        logger.info(f"Loading DNABERT-2 tokenizer from {model_name} …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        logger.info(f"Loading DNABERT-2 encoder from {model_name} …")
        self.encoder = AutoModel.from_pretrained(
            model_name, trust_remote_code=True
        )

        hidden_size = self.encoder.config.hidden_size  # 768 for DNABERT-2-117M

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, 1)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder weights frozen — only training the classification head.")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        input_ids : Tensor, shape (batch, seq_len)
        attention_mask : Tensor, shape (batch, seq_len)

        Returns
        -------
        Tensor, shape (batch,)
            Cancer probability per read (sigmoid output).
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # outputs.last_hidden_state: (batch, seq_len, hidden_size)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output).squeeze(-1)  # (batch,)
        return torch.sigmoid(logits)

    def save(self, save_dir: str) -> None:
        """Save model weights and tokenizer to save_dir."""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, "dnabert2_classifier.pt"))
        self.tokenizer.save_pretrained(save_dir)
        self.encoder.config.save_pretrained(save_dir)
        logger.info(f"Model saved to {save_dir}")

    @classmethod
    def load(cls, save_dir: str, **kwargs) -> "DNABERT2Classifier":
        """Load a previously saved classifier."""
        model = cls(**kwargs)
        weights_path = os.path.join(save_dir, "dnabert2_classifier.pt")
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        logger.info(f"Weights loaded from {weights_path}")
        return model


# ── Training ───────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: DNABERT2Classifier,
    loader: TorchDataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> dict:
    """
    Run one full training epoch.

    Returns
    -------
    dict with keys: loss, accuracy
    """
    model.train()
    criterion = nn.BCELoss()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                preds = model(input_ids, attention_mask)
                loss = criterion(preds, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(input_ids, attention_mask)
            loss = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += ((preds >= 0.5).float() == labels).sum().item()
        total += len(labels)

    return {"loss": total_loss / total, "accuracy": correct / total}


@torch.no_grad()
def evaluate_epoch(
    model: DNABERT2Classifier,
    loader: TorchDataLoader,
    device: torch.device,
) -> dict:
    """
    Run evaluation and return loss, accuracy, and raw predictions.

    Returns
    -------
    dict with keys: loss, accuracy, y_true (np.ndarray), y_pred (np.ndarray)
    """
    from sklearn.metrics import roc_auc_score

    model.eval()
    criterion = nn.BCELoss()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        preds = model(input_ids, attention_mask)
        loss = criterion(preds, labels)

        total_loss += loss.item() * len(labels)
        correct += ((preds >= 0.5).float() == labels).sum().item()
        total += len(labels)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan")

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "auc": auc,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def fine_tune(
    model: DNABERT2Classifier,
    train_loader: TorchDataLoader,
    val_loader: TorchDataLoader,
    epochs: int = 5,
    lr: float = 2e-5,
    warmup_steps: int = 100,
    checkpoint_dir: str = "results/dnabert2",
    device: Optional[torch.device] = None,
    use_amp: bool = True,
) -> dict:
    """
    Fine-tune DNABERT2Classifier with AdamW optimizer + linear warmup.

    Parameters
    ----------
    model : DNABERT2Classifier
    train_loader, val_loader : TorchDataLoader
    epochs : int
    lr : float
        Peak learning rate (AdamW). Typical range: 1e-5 – 5e-5.
    warmup_steps : int
        Linear warmup for scheduler.
    checkpoint_dir : str
        Directory to save best model checkpoint.
    device : torch.device or None
        Defaults to CUDA if available, else CPU.
    use_amp : bool
        Enable automatic mixed precision (recommended for A100).

    Returns
    -------
    dict with training history (lists of per-epoch metrics).
    """
    from transformers import get_linear_schedule_with_warmup

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Training on device: {device}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    os.makedirs(checkpoint_dir, exist_ok=True)
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_auc": [],
    }
    best_val_auc = 0.0

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, scaler)
        scheduler.step()
        val_metrics = evaluate_epoch(model, val_loader, device)

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_auc"].append(val_metrics["auc"])

        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f}  train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['accuracy']:.4f}  "
            f"val_auc={val_metrics['auc']:.4f}"
        )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_path = os.path.join(checkpoint_dir, "best_checkpoint")
            model.save(best_path)
            logger.info(f"  ✓ New best AUC={best_val_auc:.4f} — checkpoint saved.")

    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_checkpoint")
    model.save(final_path)
    logger.info(f"Final model saved to {final_path}")
    history["best_val_auc"] = best_val_auc

    return history
