EXPECTED = {
    "train_rows": 4050, "val_rows": 333, "test_rows": 333,
    "train_per_class": 1350, "test_label_counts": {0: 225, 1: 67, 2: 41},
    "train_fp": "529c5eab3547f1e7f7c4240444adcd6296c377d141ac6e275b488e066e1b477f",
    "test_fp": "b0d1f78b4093c5ef95ad488a014ddf09c5c161e4953f786e79341c7085578941",
}

def find_imbalance_free_dir():
    if DATASET_DIR_OVERRIDE:
        return Path(DATASET_DIR_OVERRIDE), [Path(DATASET_DIR_OVERRIDE)]
    found = []
    for root in [Path("/kaggle/input"), Path("."), Path("experiments/dataset"), Path("../experiments/dataset"), Path("./dataset")]:
        if not root.exists():
            continue
        for dirpath, dirnames, _ in os.walk(root):
            d = Path(dirpath)
            if len(d.relative_to(root).parts) > 5:
                dirnames[:] = []
                continue
            if d.name == "imbalance_free_dataset" and (d / "dataset_dict.json").exists():
                found.append(d)
                dirnames[:] = []
    if not found:
        raise FileNotFoundError("imbalance_free_dataset not found. Attach the Kaggle dataset or set DATASET_DIR_OVERRIDE.")
    return found[0], found

from datasets import load_from_disk, Dataset, DatasetDict

DATA_DIR, _all = find_imbalance_free_dir()
print("Candidates found:", [str(p) for p in _all])
print("USING DATASET   :", DATA_DIR)
RAW = load_from_disk(str(DATA_DIR))

# ---- hard checks: stop if this is not the dataset the paper describes -------------
assert len(RAW["train"]) == EXPECTED["train_rows"], f"train rows {len(RAW['train'])} != {EXPECTED['train_rows']}"
assert len(RAW["validation"]) == EXPECTED["val_rows"] and len(RAW["test"]) == EXPECTED["test_rows"], "val/test size mismatch"
tr_rows = RAW["train"].to_pandas().to_dict("records")
te_rows = RAW["test"].to_pandas().to_dict("records")
from collections import Counter
tr_counts = Counter(int(r["label"]) for r in tr_rows)
assert all(tr_counts[c] == EXPECTED["train_per_class"] for c in (0, 1, 2)), f"train not balanced: {tr_counts}"
q_counts = Counter((int(r["question_id"]), int(r["label"])) for r in tr_rows)
assert all(v == 150 for v in q_counts.values()) and len(q_counts) == 27, "train is not 150 per class per question"
assert dict(Counter(int(r["label"]) for r in te_rows)) == EXPECTED["test_label_counts"], "test label counts differ"
fp_tr, fp_te = fingerprint(tr_rows), fingerprint(te_rows)
assert fp_tr == EXPECTED["train_fp"], "TRAIN fingerprint differs from the verified imbalance_free_dataset"
assert fp_te == EXPECTED["test_fp"], "TEST fingerprint differs: the test set is not the one used for VBRI"
print("OK: train 4050 (1350/class, 150/class/question), validation 333, test 333 (225/67/41). Fingerprints match.")

DATASET_REPORT = {"path": str(DATA_DIR), "train_fp": fp_tr, "test_fp": fp_te,
                  "train": len(RAW["train"]), "validation": len(RAW["validation"]), "test": len(RAW["test"])}
atomic_write_json(WORK_DIR / "dataset_used.json", DATASET_REPORT)

# keep test info aligned with prediction order
TEST_IDS = [int(r["id"]) for r in te_rows]
TEST_QIDS = [int(r["question_id"]) for r in te_rows]
TEST_Y = [int(r["label"]) for r in te_rows]