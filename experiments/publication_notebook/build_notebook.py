#!/usr/bin/env python3
"""Assemble the Kaggle and the Windows notebooks (experiments/final_baseline_run*.ipynb) from the cell files in ./cells.
Run:  python3 experiments/publication_notebook/build_notebook.py
Also writes final_baseline_run*_script.py (all code cells in order) for local testing."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS = HERE / "cells"

MD = {}
MD["intro"] = r"""
# BPD screening: fine-tuning vs. training-free semantic search (VBRI). Complete experiment, A to Z

This notebook produces **every number, table and figure** of the paper in one run on one machine (Kaggle, Tesla T4):
data checks and data figures, fine-tuning of 7 encoders x 3 seeds with an honest training time, inference benchmark,
pooled dense and lexical retrieval baselines, VBRI, calibration, C-SSRS rule, ablation, k-sweep, statistics,
participant-level analysis, vignettes, all paper figures (PDF), LaTeX tables and `numbers.json`.

**Datasets** (attach the Kaggle dataset that contains both folders)
* `imbalance_free_dataset`: train 4,050 (1,350 per class, 150 per class per question), validation 333, test 333. **All models are trained on / store this set.**
* `augmented_dataset`: train 4,279 (unbalanced), same validation and test. Used **only** for the "unbalanced" rows of the ablation and to rebuild the 3,258 original answers for the data figures.
* Both are checked (row counts, labels, fingerprints) and the notebook stops if the main dataset is not the verified one.

**No test-set tuning.** k, temperature, margin and threshold are chosen on the validation set. The best fine-tuned model is chosen by validation macro-F1.

**Running it from VS Code with a Kaggle remote kernel**
1. On Kaggle, open a notebook session: Accelerator *GPU T4*, Internet *On*, Persistence *Files only* (or more), and attach the dataset.
2. In VS Code, open this notebook (it stays on your computer), select the Kaggle kernel, and press **Run All**. Compute and files live on Kaggle (`/kaggle/working/final_baseline_run`).
3. When it finishes, the last cell prints a results bundle into the notebook. **Save the notebook (Ctrl+S)** and run on your computer:
   `python experiments/publication_notebook/extract_bundle.py`. This copies all figures and tables into `VBRI_Publication_Paper/`.
   (Alternative: download `/kaggle/working/final_baseline_run_results.zip` and run the script with `--zip <file>`.)

**If the session or connection breaks:** reconnect and press **Run All** again. Finished training runs are skipped, an interrupted run resumes from its last checkpoint, and the training time stays correct (see section 3). Every later stage reads its inputs from disk, so it can also be re-run alone.

**Time:** about 1.5 h per seed for fine-tuning (3 seeds about 4 to 5 h), retrieval about 20 to 30 min, the rest a few minutes.
"""
MD_WIN = {}
MD_WIN["intro"] = r"""
# BPD screening: fine-tuning vs. training-free semantic search (VBRI). Complete experiment, A to Z (Windows workstation)

Same experiment as the Kaggle notebook, set up for **one local NVIDIA GPU on Windows** (for example an RTX 5090, 32 GB).
Everything (checkpoints, results, figures, tables) is saved next to this notebook in `final_baseline_run_out/`, and the final
figures and tables are copied into `..\VBRI_Publication_Paper\figures\generated` and `..\VBRI_Publication_Paper\tables`.

**What is different from the Kaggle version**
* Every model trains with batch 16 in one step (no gradient accumulation). The effective batch is still 16, so the training recipe is unchanged.
* bf16 mixed precision and TF32 (Ampere or newer). If DeBERTa-v3 collapses under bf16 it is retried once in fp32.
* Larger evaluation (64) and encoding (256) batches. Data loading runs in the main process (safe in Windows notebooks).

**One-time setup** (PowerShell, in the `BPD` folder)
```
python -m venv .venv-bpd
.venv-bpd\Scripts\activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r experiments\publication_notebook\requirements_windows.txt
```
Use 64-bit Python 3.10 to 3.12. RTX 50-series GPUs need a CUDA 12.8 build of PyTorch (the command above); the notebook stops with a clear message if the build does not support the GPU.

**Run**
1. Close every other program that uses the GPU (other notebooks, training jobs, games). The notebook warns if less than 12 GB of GPU memory is free. Training times are only clean if nothing else runs on the GPU.
2. Turn off Windows sleep for the duration of the run.
3. In VS Code open `experiments\final_baseline_run_windows.ipynb`, select the `.venv-bpd` kernel and press **Run All**. About 25 GB of free disk is needed.
4. If anything stops the run (kernel restart, reboot, power cut): press **Run All** again. Finished runs are skipped, an interrupted run resumes from its last checkpoint and the training time stays correct (section 3).

**No test-set tuning.** k, temperature, margin and threshold are chosen on the validation set; the best fine-tuned model is chosen by validation macro-F1.
"""
MD_WIN["export"] = "## 17. Export\nCopies the PDF figures, LaTeX tables and `numbers.json` into the paper folder and writes `final_baseline_run_results.zip` (no weights, no embeddings)."

MD["setup"] = "## 1. Setup, settings and environment\nAll switches are in the first cell. Leave `FORCE_FRESH = False` when you resume."
MD["helpers"] = "## 2. Helpers: JSON, metrics, statistics and figure style\nMetrics are computed with one function (`summarize`) for every model, so all numbers are comparable."
MD["data"] = "## 4. Datasets and their checks"
MD["normalizer"] = "Text cleaning for the fine-tuned models: the same `TextNormalizer` as the earlier baseline notebook (lower case, special characters, contractions, slang). Retrieval uses the raw answer text, as in the original VBRI experiments."
MD["datafigs"] = "## 5. Data figures\nFigure: class distribution and label mix per question (original 3,258 answers). Figure: the training views and how much of the balanced set is augmented text."
MD["training"] = "## 6. Fine-tuning: model settings, trainer and training function\nHyper-parameters are the same as in the earlier baseline notebook (effective batch size 16). Only the validation loss selects the checkpoint; the test set is scored once, after training."
MD["smoke"] = """## 7. Smoke test: does the timer survive a crash?
DistilBERT on a small subset, three times: (A) killed mid-epoch *without clean-up* (like Kaggle dying), (B) resumed, (C) uninterrupted reference.
B must resume, show wasted time, and have an effective time close to C. Otherwise the notebook stops before the long run. Skipped once passed."""
MD["main"] = """## 8. Fine-tuning: seeds x models
Seed 42 for all models first, then 43 and 44. If DeBERTa-v3 collapses to one class under fp16, it is retried once in fp32 (`DeBERTa-fp32`, both rows kept).
Run this cell again after any failure: finished runs are skipped and an interrupted run resumes."""
MD["summary"] = "## 9. Fine-tuning summary (mean ± sd over seeds)"
MD["inference"] = """## 10. Inference benchmark of the fine-tuned models (same session and GPU)
Per-answer latency including tokenisation, with warm-up and `torch.cuda.synchronize()`: GPU batch 1, GPU batch 32, CPU batch 1 (first 100 test answers). fp32 weights."""
MD["engine"] = """## 11. Retrieval engine
Dense k-NN over normalised embeddings (pooled index or nine isolated per-criterion databases), similarity-weighted vote, BM25 and TF-IDF baselines, temperature-softmax calibration with margin and BPD threshold, and the C-SSRS rule (Q5 only, patterns unchanged). Embeddings are cached in `retrieval/emb`."""
MD["experiments"] = """## 12. Retrieval experiments
Pooled baselines (4 encoders x 2 text formats, BM25, TF-IDF), VBRI raw, calibrated, + C-SSRS, ablation (isolation x balance, with and without class weights), k-sweep and per-criterion analysis.
Every k and every calibration setting is chosen on the **validation** set. A reproduction check compares k = 5 with the earlier VBRI notebook."""
MD["timing"] = "## 13. VBRI timing (same session and hardware as fine-tuning)\nIndex build (3 runs), per-answer latency text -> prediction on GPU and CPU, index size. The encoder load time is reported but not counted."
MD["analysis"] = "## 14. Statistics, participant level, error analysis and vignettes\nPaired bootstrap 95% CIs, McNemar tests, ROC/AUROC and specificity at 85% sensitivity, participant-level agreement, negation errors, and vignettes with the retrieved neighbours."
MD["figures"] = "## 15. Paper figures (PDF + PNG in `paper_assets/figures`)"
MD["tables"] = "## 16. LaTeX tables and `numbers.json` (in `paper_assets/tables`)"
MD["export"] = "## 17. Export\nZip on the Kaggle disk, and the same bundle (PDF figures, tables, results; no weights, no PNG, no embeddings) printed below for the VS Code remote kernel."

ORDER = [
    ("md", "intro"), ("md", "setup"), ("py", "01_setup.py"), ("py", "02_env.py"),
    ("md", "helpers"), ("py", "03_helpers.py"),
    ("mdfile", "04_timer.md"), ("py", "04_timer.py"),
    ("md", "data"), ("py", "05_data.py"), ("md", "normalizer"), ("py", "06_normalizer.py"),
    ("md", "datafigs"), ("py", "07_datafigs.py"),
    ("md", "training"), ("py", "08_training.py"),
    ("md", "smoke"), ("py", "09_smoke.py"),
    ("md", "main"), ("py", "10_main.py"),
    ("md", "summary"), ("py", "11_ft_summary.py"),
    ("md", "inference"), ("py", "12_inference.py"),
    ("md", "engine"), ("py", "13_retrieval_engine.py"),
    ("md", "experiments"), ("py", "14_retrieval_experiments.py"),
    ("md", "timing"), ("py", "15_vbri_timing.py"),
    ("md", "analysis"), ("py", "16_analysis.py"),
    ("md", "figures"), ("py", "17_figures.py"),
    ("md", "tables"), ("py", "18_tables.py"),
    ("md", "export"), ("py", "19_export.py"),
]


def cell(kind, text):
    lines = text.strip("\n").split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    c = {"cell_type": "markdown" if kind != "code" else "code", "metadata": {}, "source": src}
    if kind == "code":
        c["outputs"], c["execution_count"] = [], None
    return c


PROFILES = {
    "kaggle": {"nb": HERE.parent / "final_baseline_run.ipynb", "py": HERE / "final_baseline_run_script.py", "md": {}},
    "windows": {"nb": HERE.parent / "final_baseline_run_windows.ipynb", "py": HERE / "final_baseline_run_windows_script.py", "md": MD_WIN},
}


def build(profile="kaggle"):
    P = PROFILES[profile]
    cells, code = [], []
    for kind, ref in ORDER:
        if kind == "md":
            cells.append(cell("markdown", P["md"].get(ref, MD[ref])))
        elif kind == "mdfile":
            cells.append(cell("markdown", (CELLS / ref).read_text()))
        else:
            src = (CELLS / ref).read_text().replace('PROFILE = "__PROFILE__"', f'PROFILE = "{profile}"')
            cells.append(cell("code", src))
            code.append(f"# {'=' * 30} {ref} {'=' * 30}\n{src}\n")
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    P["nb"].write_text(json.dumps(nb, indent=1), encoding="utf-8")
    P["py"].write_text("\n".join(code), encoding="utf-8")
    print(f"[{profile}] wrote {P['nb']} ({len(cells)} cells) and {P['py']}")


if __name__ == "__main__":
    for prof in PROFILES:
        build(prof)
