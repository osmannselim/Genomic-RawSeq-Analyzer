"""
report_generator.py
--------------------
FASTQ → LLM Clinical Report — final stage of the Genomic-RawSeq-Analyzer pipeline.

Takes per-patient model outputs (cancer probability, top occlusion motifs) and
generates a human-readable clinical-style summary using LLAMA-3-8B-Instruct.

Backends (tried in order):
  1. Ollama local inference  — `ollama run llama3` (no GPU required, recommended)
  2. HuggingFace Inference API — requires HF_TOKEN env var
  3. HuggingFace transformers local — requires ~16 GB RAM, slow on CPU

Usage:
    # Generate reports for patients using motifs from run_occlusion.py:
    python src/report_generator.py \
        --batch_dir   results/batches \
        --model_path  results/cnn_baseline.keras \
        --motifs_path results/occlusion/motifs.json \
        --output_dir  results/reports \
        --n_patients  5

    # Or import and call directly:
    from report_generator import ReportGenerator
    gen = ReportGenerator(backend="ollama")
    report = gen.generate(patient_data)
    gen.save(report, patient_id, output_dir="results/reports")
"""

import os
import sys
import json
import argparse
import textwrap
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# ── Risk thresholds ──────────────────────────────────────────────────────
RISK_THRESHOLDS = {
    "high":     0.65,
    "moderate": 0.55,
}

def risk_label(prob: float) -> str:
    if prob >= RISK_THRESHOLDS["high"]:
        return "HIGH RISK"
    if prob >= RISK_THRESHOLDS["moderate"]:
        return "MODERATE RISK"
    return "LOW RISK"


# ── Prompt template ──────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a clinical genomics assistant helping researchers interpret
    alignment-free whole-exome sequencing results. Your reports are
    informational summaries for research purposes only — not clinical diagnoses.
    Write clearly, concisely, and in plain language that a clinician could read.
    Use a structured format with short sections. Avoid speculation beyond the data.
""")

def build_prompt(patient: dict) -> str:
    """
    Construct the LLM prompt from a patient data dict.

    Expected keys:
        patient_id      : str  (e.g. "ERR166302")
        cancer_prob     : float (0–1)
        n_reads         : int
        cancer_type     : str  (e.g. "Tumor (WXS, breast)")
        top_motifs      : list of dicts (position, kmer, score, cosmic_hits)
    """
    pid   = patient["patient_id"]
    prob  = patient["cancer_prob"]
    label = risk_label(prob)
    reads = patient.get("n_reads", "N/A")
    ctype = patient.get("cancer_type", "Whole Exome Sequencing")

    motif_lines = []
    for m in patient.get("top_motifs", []):
        pos   = m["position"]
        kmer  = m["kmer"]
        score = m["score"]
        cosmic_hits = m.get("cosmic_hits", [])
        if cosmic_hits:
            sigs = ", ".join(f"{h['signature']} ({h['description']})" for h in cosmic_hits)
            motif_lines.append(
                f"  - Positions {pos}–{pos+len(kmer)-1}: {kmer} "
                f"(importance={score:.3f}) → matches {sigs}"
            )
        else:
            motif_lines.append(
                f"  - Positions {pos}–{pos+len(kmer)-1}: {kmer} "
                f"(importance={score:.3f})"
            )
    motif_block = "\n".join(motif_lines) if motif_lines else "  - No high-importance motifs detected."

    return textwrap.dedent(f"""\
        PATIENT GENOMIC ANALYSIS REPORT
        ================================
        Patient ID        : {pid}
        Sequencing Type   : {ctype}
        Reads Analysed    : {reads:,} reads
        Cancer Probability: {prob:.4f}  →  {label}

        Top Occlusion-Sensitivity Motifs (k=5):
        {motif_block}

        ---
        Based on the above alignment-free deep learning analysis, please write
        a structured clinical summary report with the following sections:

        1. PATIENT SUMMARY — one paragraph describing the overall finding
        2. MODEL EVIDENCE — what the cancer probability score means and how
           it was computed (crowd-voting aggregation over sequencing reads)
        3. MOTIF ANALYSIS — interpret the detected k-mer motifs; mention any
           COSMIC mutation signature associations found
        4. LIMITATIONS — key caveats (read length, cohort size, research-only)
        5. RECOMMENDATION — suggested next steps for clinical follow-up

        Keep each section to 2–4 sentences. Do not invent data not provided above.
    """)


# ── Backend implementations ──────────────────────────────────────────────

def _call_ollama(prompt: str, model: str = "llama3") -> str:
    """Call Ollama local inference via its Python client or HTTP API."""
    try:
        import ollama
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": prompt},
            ],
        )
        return response["message"]["content"]
    except ImportError:
        pass

    # Fallback: raw HTTP if ollama Python package not installed
    import urllib.request
    import json as _json
    payload = _json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = _json.loads(resp.read())
    return data["message"]["content"]


def _call_hf_api(prompt: str, model: str = "meta-llama/Meta-Llama-3-8B-Instruct") -> str:
    """Call HuggingFace Inference API (requires HF_TOKEN env var)."""
    import urllib.request
    import json as _json

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN environment variable not set. "
            "Export it with: export HF_TOKEN=hf_..."
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    payload = _json.dumps({
        "inputs": f"<|system|>{SYSTEM_PROMPT}<|user|>{prompt}<|assistant|>",
        "parameters": {"max_new_tokens": 800, "temperature": 0.3},
    }).encode()

    url = f"https://api-inference.huggingface.co/models/{model}"
    req = urllib.request.Request(
        url, data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = _json.loads(resp.read())

    if isinstance(data, list):
        return data[0].get("generated_text", str(data))
    return str(data)


def _call_hf_local(prompt: str, model: str = "meta-llama/Meta-Llama-3-8B-Instruct") -> str:
    """Run LLAMA-3 locally via transformers pipeline (CPU-only, slow)."""
    from transformers import pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        max_new_tokens=800,
        temperature=0.3,
        do_sample=True,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    result = pipe(messages)
    return result[0]["generated_text"][-1]["content"]


# ── ReportGenerator class ────────────────────────────────────────────────

class ReportGenerator:
    """
    Generate clinical-style reports from patient model outputs using LLAMA-3.

    Parameters
    ----------
    backend : str
        "ollama"    — local Ollama server (default, recommended)
        "hf_api"    — HuggingFace Inference API (needs HF_TOKEN)
        "hf_local"  — HuggingFace transformers local (needs ~16 GB RAM)
    model : str
        Model identifier (Ollama model name or HF model ID).
    """

    BACKENDS = {
        "ollama":    _call_ollama,
        "hf_api":    _call_hf_api,
        "hf_local":  _call_hf_local,
    }

    def __init__(self, backend: str = "ollama", model: str = None):
        if backend not in self.BACKENDS:
            raise ValueError(f"backend must be one of {list(self.BACKENDS)}")
        self.backend = backend
        self.model   = model or ("llama3" if backend == "ollama"
                                 else "meta-llama/Meta-Llama-3-8B-Instruct")
        self._call   = self.BACKENDS[backend]

    def generate(self, patient: dict) -> str:
        """
        Generate a clinical report for one patient.

        Parameters
        ----------
        patient : dict — see build_prompt() for expected keys.

        Returns
        -------
        str : formatted report text
        """
        prompt = build_prompt(patient)
        print(f"  Querying {self.backend} ({self.model})...")
        raw = self._call(prompt, self.model)
        return self._format_report(patient, raw)

    def generate_batch(self, patients: list, output_dir: str) -> list:
        """
        Generate and save reports for a list of patients.

        Returns list of (patient_id, report_path) tuples.
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []
        for i, p in enumerate(patients, 1):
            pid = p["patient_id"]
            print(f"\n[{i}/{len(patients)}] Generating report for {pid}...")
            try:
                report = self.generate(p)
                path   = self.save(report, pid, output_dir)
                results.append((pid, path))
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append((pid, None))
        return results

    def save(self, report: str, patient_id: str, output_dir: str) -> str:
        """Save report to <output_dir>/<patient_id>_report.txt. Returns path."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{patient_id}_report.txt")
        with open(path, "w") as f:
            f.write(report)
        print(f"  Saved: {path}")
        return path

    @staticmethod
    def _format_report(patient: dict, llm_text: str) -> str:
        """Wrap LLM output in a consistent header/footer."""
        pid   = patient["patient_id"]
        prob  = patient["cancer_prob"]
        label = risk_label(prob)
        ts    = datetime.now().strftime("%Y-%m-%d %H:%M")
        sep   = "=" * 70

        header = (
            f"{sep}\n"
            f"GENOMIC-RAWSEQ-ANALYZER — CLINICAL SUMMARY REPORT\n"
            f"{sep}\n"
            f"Patient ID      : {pid}\n"
            f"Cancer Prob.    : {prob:.4f}  [{label}]\n"
            f"Generated       : {ts}\n"
            f"Model           : 1D-CNN + Crowd-Voting Aggregation\n"
            f"DISCLAIMER      : Research use only. Not a clinical diagnosis.\n"
            f"{sep}\n\n"
        )
        footer = (
            f"\n{sep}\n"
            f"END OF REPORT — {pid}\n"
            f"{sep}\n"
        )
        return header + llm_text.strip() + footer


# ── Helpers to build patient dicts from existing results ─────────────────

def build_patient_list_from_results(
    model_path: str,
    batch_dir:  str,
    motifs_path: str,
    n_patients: int = 5,
) -> list:
    """
    Load the CNN model + batch data, run inference, and assemble patient dicts
    with the top-scoring patients (highest cancer probability).

    Motifs are shared population-level motifs from motifs.json (run_occlusion.py
    output) — the same top-5 motifs are assigned to every patient since occlusion
    aggregation is done at population level, not per-patient.
    """
    import numpy as np
    import pandas as pd
    from tensorflow.keras.models import load_model
    from data_loader import DataLoader

    print("Loading model and data...")
    cnn    = load_model(model_path)
    X, y, run_ids = DataLoader.load_all_batches(batch_dir)

    print("Running inference...")
    probs = cnn.predict(X, batch_size=2048, verbose=1).flatten()

    df = pd.DataFrame({"run_id": run_ids, "prob": probs, "label": y})
    pat = df.groupby("run_id").agg(
        cancer_prob=("prob", "mean"),
        patient_label=("label", lambda x: int(x.mode()[0])),
        n_reads=("prob", "count"),
    ).reset_index().sort_values("cancer_prob", ascending=False)

    # Load population motifs (same for all patients)
    motifs = []
    if motifs_path and os.path.exists(motifs_path):
        with open(motifs_path) as f:
            motifs = json.load(f).get("top_motifs", [])
    else:
        print(f"  Warning: motifs.json not found at {motifs_path}. "
              "Run run_occlusion.py first for real motifs.")

    patients = []
    for _, row in pat.head(n_patients).iterrows():
        ctype = "Tumor (WXS)" if row["patient_label"] == 1 else "Normal (WXS)"
        patients.append({
            "patient_id":  row["run_id"],
            "cancer_prob": round(float(row["cancer_prob"]), 4),
            "n_reads":     int(row["n_reads"]),
            "cancer_type": ctype,
            "top_motifs":  motifs,
        })
    return patients


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate LLM clinical reports for patients.")
    p.add_argument("--batch_dir",   default="results/batches")
    p.add_argument("--model_path",  default="results/cnn_baseline.keras")
    p.add_argument("--motifs_path", default="results/occlusion/motifs.json")
    p.add_argument("--output_dir",  default="results/reports")
    p.add_argument("--n_patients",  type=int, default=5)
    p.add_argument("--backend",     default="ollama",
                   choices=["ollama", "hf_api", "hf_local"])
    p.add_argument("--llm_model",   default=None,
                   help="Override model name (e.g. llama3.1, llama3:70b)")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("LLM CLINICAL REPORT GENERATION")
    print("=" * 60)

    patients = build_patient_list_from_results(
        model_path=args.model_path,
        batch_dir=args.batch_dir,
        motifs_path=args.motifs_path,
        n_patients=args.n_patients,
    )

    print(f"\nBackend : {args.backend}")
    print(f"Patients: {len(patients)}")
    for p in patients:
        print(f"  {p['patient_id']}  P(cancer)={p['cancer_prob']:.4f}  "
              f"[{risk_label(p['cancer_prob'])}]  {p['cancer_type']}")

    gen     = ReportGenerator(backend=args.backend, model=args.llm_model)
    results = gen.generate_batch(patients, output_dir=args.output_dir)

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")
    for pid, path in results:
        status = path if path else "FAILED"
        print(f"  {pid}  →  {status}")


if __name__ == "__main__":
    main()
