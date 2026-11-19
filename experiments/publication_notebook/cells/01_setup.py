PROFILE = "__PROFILE__"   # set by build_notebook.py: "kaggle" or "windows"

import os, sys, json, time, shutil, hashlib, platform, subprocess, random, math, gc, re, csv, statistics, zipfile, inspect, traceback, base64, io
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict

# Use ONE GPU so that timings are not affected by Kaggle's 2xT4 / DataParallel. Must be set before torch is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if PROFILE == "windows":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # Windows without developer mode cannot symlink the HF cache

def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)

for _mod, _pkg in [("contractions", "contractions"), ("rank_bm25", "rank-bm25"),
                   ("sentence_transformers", "sentence-transformers"), ("scipy", "scipy"), ("sklearn", "scikit-learn")]:
    try:
        __import__(_mod)
    except ImportError:
        _pip(_pkg)
import contractions

# ------------------------------ settings ------------------------------------
CODE_VERSION = "final_baseline_v1"
SEEDS = [42, 43, 44]                       # all models for seed 42 first, then 43, then 44
MODELS = ["ELECTRA", "BERT", "DistilBERT", "RoBERTa", "XLNet", "DeBERTa", "ModernBERT"]
MAX_EPOCHS = 15
PATIENCE = 2                               # early stopping on validation loss
TRAIN_SUBSET = None                        # None = full training set (only the local test mode uses a subset)
KEEP_WEIGHTS_SEEDS = [SEEDS[0]]            # weights are kept only for these seeds (needed for the inference benchmark)

# Stages. Every stage saves its results to disk, so any later stage can be re-run on its own after a restart.
RUN_DATA_FIGURES = True
RUN_SMOKE_TEST = True                      # short crash-and-resume check (skipped automatically once it has passed)
RUN_MAIN_TRAINING = True
RUN_INFERENCE_BENCH = True
RUN_RETRIEVAL = True                       # pooled baselines, lexical baselines, VBRI, calibration, ablation, k-sweep
RUN_VBRI_TIMING = True
RUN_ANALYSIS = True                        # statistics, patient level, vignettes, error analysis
RUN_FIGURES = True
RUN_TABLES = True
RUN_EXPORT = True
EMBED_BUNDLE_IN_NOTEBOOK = PROFILE == "kaggle"   # print the results bundle into the notebook (Kaggle remote kernel only)
COPY_TO_PAPER = PROFILE == "windows"       # copy final figures/tables straight into ../VBRI_Publication_Paper (if that folder exists)

# GPU use. "windows" = one big local GPU: every model trains with batch 16 in a single step (same effective batch 16
# as before, no gradient accumulation), larger evaluation and encoding batches, bf16 and TF32 on Ampere or newer.
FULL_BATCH = PROFILE == "windows"
EVAL_BS = 64 if PROFILE == "windows" else None   # None = the per-model value from the earlier baseline
ENCODE_BS = 256 if PROFILE == "windows" else 64

FORCE_FRESH = False                        # DANGER: True deletes everything this notebook produced. Leave False when resuming.
DATASET_DIR_OVERRIDE = None                # e.g. "/kaggle/input/<name>/imbalance_free_dataset"
AUG_DATASET_DIR_OVERRIDE = None            # e.g. "/kaggle/input/<name>/augmented_dataset"
N_BOOT = 10000                             # bootstrap resamples for confidence intervals

# Local test mode (tiny CPU run used to check the notebook before Kaggle). Never set this on Kaggle.
TEST_MODE = os.environ.get("BPD_TEST_MODE") == "1"
if TEST_MODE:
    SEEDS = [42]
    MODELS = ["DistilBERT"]
    MAX_EPOCHS = 2
    TRAIN_SUBSET = {"train": 160, "val": 333}
    KEEP_WEIGHTS_SEEDS = [42]
    N_BOOT = 300
    EMBED_BUNDLE_IN_NOTEBOOK = True
    print("!!! TEST MODE: tiny CPU run for checking the notebook. Results are NOT valid.")

ON_KAGGLE = Path("/kaggle/working").exists()
if PROFILE == "windows":
    # everything (checkpoints, results, figures) is saved next to this notebook
    WORK_DIR = Path(os.environ.get("BPD_WORK_DIR") or (Path.cwd() / "final_baseline_run_out"))
    PAPER_REPO_DIR = Path(os.environ.get("BPD_PAPER_DIR") or (Path.cwd().parent / "VBRI_Publication_Paper"))
else:
    WORK_DIR = Path(os.environ.get("BPD_WORK_DIR") or ("/kaggle/working/final_baseline_run" if ON_KAGGLE else "./final_baseline_run_out"))
    PAPER_REPO_DIR = None
if FORCE_FRESH and WORK_DIR.exists():
    print("!!! FORCE_FRESH=True -> deleting", WORK_DIR)
    shutil.rmtree(WORK_DIR)
RUNS_DIR = WORK_DIR / "runs"
RETR_DIR = WORK_DIR / "retrieval"
PAPER_DIR = WORK_DIR / "paper_assets"
FIG_DIR, TAB_DIR = PAPER_DIR / "figures", PAPER_DIR / "tables"
for _d in (WORK_DIR, RUNS_DIR, RETR_DIR, FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)
print("Work dir:", WORK_DIR.resolve())
print("Free disk (GB):", round(shutil.disk_usage(WORK_DIR).free / 1e9, 1), "(about 25 GB is needed for checkpoints, weights and model downloads)")
if PROFILE == "windows" and len(str(WORK_DIR.resolve())) > 120:
    print("!!! The work folder path is long. Windows limits paths to 260 characters; move the project closer to the drive root "
          "or enable long paths if you see 'file not found' errors when checkpoints are saved.")
