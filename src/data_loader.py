"""
data_loader.py
--------------
FASTQ data ingestion, integer encoding, and batch processing pipeline.

Handles downloading from ENA FTP, parsing FASTQ files via BioPython,
integer-encoding sequences, and saving compressed batches to disk.

Usage:
    from data_loader import DataLoader, assign_labels
    loader = DataLoader(output_dir="/path/to/batches")
    loader.process_run_list(run_df)          # run_df has columns: Run, Label
    X, y = DataLoader.load_all_batches(dir)  # load saved batches
"""

import os
import gzip
import glob
import logging
import numpy as np
import pandas as pd
from Bio import SeqIO

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────

BASE_FTP_URL = "http://ftp.sra.ebi.ac.uk/vol1/fastq"

# ERR166316–ERR166329 are Normal; all others in the Semester 1 dataset are Tumor
NORMAL_RUN_RANGE = (166316, 166329)

# Full SRA run list used in Semester 1 (breast cancer WXS cohort)
DEFAULT_RUN_IDS = [
    "ERR166302", "ERR166303", "ERR166304", "ERR166305", "ERR166306",
    "ERR166307", "ERR166308", "ERR166309", "ERR166310", "ERR166311",
    "ERR166312", "ERR166313", "ERR166314", "ERR166315",
    "ERR166316", "ERR166317", "ERR166318", "ERR166319", "ERR166320",
    "ERR166321", "ERR166322", "ERR166323", "ERR166324", "ERR166325",
    "ERR166326", "ERR166327", "ERR166328", "ERR166329",
    "ERR166330", "ERR166331", "ERR166332", "ERR166333", "ERR166334",
    "ERR166335", "ERR166336", "ERR166337",
]

BASE_ENCODING = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
BASE_DECODING = {v: k for k, v in BASE_ENCODING.items()}
BASE_DECODING[0] = "_"   # padding symbol


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def assign_labels(run_ids: list) -> pd.DataFrame:
    """
    Assign binary labels to SRA run accession IDs.

    Label convention (Semester 1 dataset):
        ERR166316–ERR166329 → 0  (Normal / Healthy)
        All others           → 1  (Tumor / Cancer)

    Returns
    -------
    pd.DataFrame with columns: Run, Label
    """
    labels = []
    for run in run_ids:
        num = int(run[3:])
        label = 0 if NORMAL_RUN_RANGE[0] <= num <= NORMAL_RUN_RANGE[1] else 1
        labels.append(label)
    return pd.DataFrame({"Run": run_ids, "Label": labels})


def seq_to_int(seq_str: str, max_len: int = 80) -> list:
    """
    Convert a nucleotide string to a fixed-length integer vector.

    Encoding:  A=1, C=2, G=3, T=4, N (ambiguous)=5, Padding=0
    Sequences longer than max_len are truncated right.
    Sequences shorter than max_len are right-padded with 0.
    """
    nums = [BASE_ENCODING.get(base, 5) for base in seq_str[:max_len]]
    if len(nums) < max_len:
        nums += [0] * (max_len - len(nums))
    return nums


def int_to_seq(vec: np.ndarray) -> str:
    """Reverse of seq_to_int. Convert integer vector back to nucleotide string."""
    return "".join(BASE_DECODING.get(int(x), "?") for x in vec)


def _build_ftp_url(run_id: str) -> str:
    """Construct the ENA FTP URL for a given SRA run accession."""
    if len(run_id) == 9:
        sub_dir = run_id[:6]
    elif len(run_id) == 10:
        sub_dir = f"00{run_id[-1]}"
    else:
        sub_dir = run_id[:6]
    return f"{BASE_FTP_URL}/{sub_dir}/{run_id}/{run_id}_1.fastq.gz"


# ─────────────────────────────────────────────────────────
# DataLoader
# ─────────────────────────────────────────────────────────

class DataLoader:
    """
    End-to-end pipeline: download FASTQ from ENA → encode → save batches.

    Parameters
    ----------
    output_dir : str
        Root directory where .npz batch files will be saved.
    max_len : int
        Fixed read length after truncation/padding. Default: 80.
    reads_per_file : int
        Maximum reads extracted per SRA run. Default: 50,000.
    batch_size : int
        Number of SRA runs per .npz checkpoint file. Default: 4.
    """

    def __init__(
        self,
        output_dir: str,
        max_len: int = 80,
        reads_per_file: int = 50_000,
        batch_size: int = 4,
    ):
        self.output_dir = output_dir
        self.max_len = max_len
        self.reads_per_file = reads_per_file
        self.batch_size = batch_size
        os.makedirs(output_dir, exist_ok=True)

    # ── Public ─────────────────────────────────────────────

    def process_run_list(self, run_df: pd.DataFrame) -> None:
        """
        Process a DataFrame of SRA runs (columns: Run, Label).

        Downloads each FASTQ, encodes reads, and saves compressed
        .npz checkpoint files every `batch_size` runs.
        Resume-safe: existing batch files are not overwritten.
        """
        current_X, current_y = [], []
        total = len(run_df)

        for idx, row in run_df.iterrows():
            run_id, label = row["Run"], int(row["Label"])
            logger.info(f"[{idx + 1}/{total}] Processing {run_id}  label={label}")

            reads = self._download_and_parse(run_id)
            if reads:
                current_X.extend(reads)
                current_y.extend([label] * len(reads))
                logger.info(f"  → {len(reads):,} reads added.")
            else:
                logger.warning(f"  → Skipped {run_id} (no data).")

            files_done = idx + 1
            if files_done % self.batch_size == 0 or files_done == total:
                batch_num = (files_done + self.batch_size - 1) // self.batch_size
                self._save_batch(current_X, current_y, batch_num)
                current_X, current_y = [], []

    # ── Static loaders ─────────────────────────────────────

    @staticmethod
    def load_all_batches(batch_dir: str) -> tuple:
        """
        Load and concatenate all .npz batch files from a directory.

        Returns
        -------
        X : np.ndarray, shape (N, max_len)   — integer-encoded reads
        y : np.ndarray, shape (N,)            — binary labels (0=Normal, 1=Tumor)
        """
        files = sorted(glob.glob(os.path.join(batch_dir, "batch_*.npz")))
        if not files:
            raise FileNotFoundError(f"No batch files found in: {batch_dir}")

        X_parts, y_parts = [], []
        for f in files:
            with np.load(f) as data:
                X_parts.append(data["X"])
                y_parts.append(data["y"])
            logger.info(f"  Loaded {os.path.basename(f)}")

        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        logger.info(f"Total: X={X.shape}  y={y.shape}  "
                    f"(Normal={sum(y == 0):,}  Tumor={sum(y == 1):,})")
        return X, y

    @staticmethod
    def load_unsupervised(
        normal_dir: str,
        cancer_dir: str,
        max_len: int = 80,
    ) -> tuple:
        """
        Load Normal and Cancer data separately for LSTM Autoencoder training.
        Files should be individual .npy arrays (labels are implicit via folder).

        Returns
        -------
        X_normal : np.ndarray, shape (N, max_len, 1)
        X_cancer : np.ndarray, shape (M, max_len, 1)
        """

        def _load_dir(path: str) -> np.ndarray:
            files = sorted(glob.glob(os.path.join(path, "*.npy")))
            if not files:
                raise FileNotFoundError(f"No .npy files in: {path}")
            return np.concatenate([np.load(f) for f in files], axis=0)

        X_normal = _load_dir(normal_dir).reshape(-1, max_len, 1)
        X_cancer = _load_dir(cancer_dir).reshape(-1, max_len, 1)
        logger.info(f"Unsupervised: normal={X_normal.shape}  cancer={X_cancer.shape}")
        return X_normal, X_cancer

    # ── Private ─────────────────────────────────────────────

    def _download_and_parse(self, run_id: str) -> list:
        """Download one SRA run, parse FASTQ, return list of encoded reads."""
        url = _build_ftp_url(run_id)
        local_path = f"{run_id}_1.fastq.gz"
        reads = []

        try:
            exit_code = os.system(f"wget -q '{url}' -O '{local_path}'")
            if (
                exit_code != 0
                or not os.path.exists(local_path)
                or os.path.getsize(local_path) < 1000
            ):
                logger.warning(f"  Download failed for {run_id}.")
                return []

            with gzip.open(local_path, "rt") as handle:
                for record in SeqIO.parse(handle, "fastq"):
                    reads.append(seq_to_int(str(record.seq), self.max_len))
                    if len(reads) >= self.reads_per_file:
                        break

        except Exception as e:
            logger.error(f"  Error on {run_id}: {e}")

        finally:
            if os.path.exists(local_path):
                os.remove(local_path)

        return reads

    def _save_batch(self, X: list, y: list, batch_num: int) -> None:
        """Compress and save current batch as .npz."""
        if not X:
            logger.warning("  Empty batch — nothing saved.")
            return

        save_path = os.path.join(self.output_dir, f"batch_{batch_num:03d}.npz")
        np.savez_compressed(
            save_path,
            X=np.array(X, dtype=np.int8),
            y=np.array(y, dtype=np.int8),
        )
        logger.info(f"  ── Saved {save_path}  shape=({len(X)}, {self.max_len})")


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and encode SRA FASTQ runs.")
    parser.add_argument("--output_dir", required=True, help="Directory for batch .npz files")
    parser.add_argument("--reads_per_file", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    run_df = assign_labels(DEFAULT_RUN_IDS)
    print(f"Runs: {len(run_df)}  |  Normal: {sum(run_df.Label == 0)}  "
          f"Tumor: {sum(run_df.Label == 1)}")

    loader = DataLoader(
        output_dir=args.output_dir,
        reads_per_file=args.reads_per_file,
        batch_size=args.batch_size,
    )
    loader.process_run_list(run_df)
