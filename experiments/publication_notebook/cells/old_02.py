import os, sys, json, time, shutil, hashlib, platform, subprocess, random, math, gc, re, csv, statistics, zipfile, inspect, traceback
from pathlib import Path
from datetime import datetime, timezone

# Use ONE GPU so that timings are not affected by Kaggle's 2xT4 / DataParallel. Must be set before torch is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)

try:
    import contractions
except ImportError:
    _pip("contractions")
    import contractions
try:
    import sentence_transformers  # needed only for the VBRI benchmark
except ImportError:
    _pip("sentence-transformers")

# ------------------------------ settings ------------------------------------
CODE_VERSION = "final_baseline_v1"
SEEDS = [42, 43, 44]                       # run order: all models for seed 42 first, then 43, then 44
MODELS = ["ELECTRA", "BERT", "DistilBERT", "RoBERTa", "XLNet", "DeBERTa", "ModernBERT"]
MAX_EPOCHS = 15
PATIENCE = 2                               # early stopping on validation loss
RUN_SMOKE_TEST = True                      # short crash-and-resume check (skipped automatically once it has passed)
RUN_MAIN_TRAINING = True
RUN_INFERENCE_BENCH = True
RUN_VBRI_BENCH = True
KEEP_WEIGHTS_SEEDS = [SEEDS[0]]            # weights are kept only for these seeds (needed for the inference benchmark)
FORCE_FRESH = False                        # DANGER: True deletes everything this notebook produced. Leave False when resuming.
DATASET_DIR_OVERRIDE = None                # e.g. "/kaggle/input/<name>/imbalance_free_dataset"

ON_KAGGLE = Path("/kaggle/working").exists()
WORK_DIR = Path("/kaggle/working/final_baseline_run") if ON_KAGGLE else Path("./final_baseline_run_out")
if FORCE_FRESH and WORK_DIR.exists():
    print("!!! FORCE_FRESH=True -> deleting", WORK_DIR)
    shutil.rmtree(WORK_DIR)
RUNS_DIR = WORK_DIR / "runs"
WORK_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
print("Work dir:", WORK_DIR.resolve())
print("Free disk (GB):", round(shutil.disk_usage(WORK_DIR).free / 1e9, 1))