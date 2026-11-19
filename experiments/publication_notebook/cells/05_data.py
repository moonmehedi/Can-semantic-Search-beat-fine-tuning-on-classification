from datasets import load_from_disk, Dataset, DatasetDict

EXPECTED = {   # imbalance_free_dataset (main dataset: fine-tuning AND VBRI use this training set)
    "train_rows": 4050, "val_rows": 333, "test_rows": 333,
    "train_per_class": 1350, "test_label_counts": {0: 225, 1: 67, 2: 41},
    "train_fp": "529c5eab3547f1e7f7c4240444adcd6296c377d141ac6e275b488e066e1b477f",
    "test_fp": "b0d1f78b4093c5ef95ad488a014ddf09c5c161e4953f786e79341c7085578941",
}
EXPECTED_AUG = {"train_rows": 4279, "label_counts": {0: 1746, 1: 1612, 2: 921}}     # augmented_dataset (ablation only)
EXPECTED_ORIG = {                                                                    # the 3,258 original answers
    "rows": 3258, "participants": 362, "class_counts": {0: 2195, 1: 673, 2: 390},
    "orig_train_counts": {0: 1746, 1: 538, 2: 308},
    "per_question": {1: (198, 118, 46), 2: (244, 71, 47), 3: (283, 51, 28), 4: (240, 33, 89), 5: (325, 22, 15),
                     6: (230, 99, 33), 7: (227, 90, 45), 8: (220, 113, 29), 9: (228, 76, 58)},
}
DSM5 = {1: "Abandonment avoidance", 2: "Unstable relationships", 3: "Identity disturbance", 4: "Impulsive behaviours",
        5: "Suicidal / self-harm", 6: "Affective instability", 7: "Chronic emptiness", 8: "Inappropriate anger",
        9: "Paranoia / dissociation"}

def find_dataset_dirs(name, override=None):
    if override:
        return [Path(override)]
    found = []
    for root in [Path("/kaggle/input"), Path("."), Path("experiments/dataset"), Path("../experiments/dataset"), Path("./dataset")]:
        if not root.exists():
            continue
        for dirpath, dirnames, _ in os.walk(root):
            d = Path(dirpath)
            if len(d.relative_to(root).parts) > 5:
                dirnames[:] = []
                continue
            if d.name == name and (d / "dataset_dict.json").exists():
                found.append(d.resolve())
                dirnames[:] = []
    return list(dict.fromkeys(found))

def rows_of(ds):
    return ds.to_pandas().to_dict("records")

# ------------------------------ main dataset ---------------------------------
_c = find_dataset_dirs("imbalance_free_dataset", DATASET_DIR_OVERRIDE)
if not _c:
    raise FileNotFoundError("imbalance_free_dataset not found. Attach the Kaggle dataset or set DATASET_DIR_OVERRIDE.")
DATA_DIR = _c[0]
print("Candidates:", [str(p) for p in _c])
print("USING MAIN DATASET :", DATA_DIR)
RAW = load_from_disk(str(DATA_DIR))
assert len(RAW["train"]) == EXPECTED["train_rows"], f"train rows {len(RAW['train'])} != {EXPECTED['train_rows']}"
assert len(RAW["validation"]) == EXPECTED["val_rows"] and len(RAW["test"]) == EXPECTED["test_rows"], "val/test size mismatch"
tr_rows, va_rows, te_rows = rows_of(RAW["train"]), rows_of(RAW["validation"]), rows_of(RAW["test"])
tr_counts = Counter(int(r["label"]) for r in tr_rows)
assert all(tr_counts[c] == EXPECTED["train_per_class"] for c in (0, 1, 2)), f"train not balanced: {tr_counts}"
q_counts = Counter((int(r["question_id"]), int(r["label"])) for r in tr_rows)
assert all(v == 150 for v in q_counts.values()) and len(q_counts) == 27, "train is not 150 per class per question"
assert dict(Counter(int(r["label"]) for r in te_rows)) == EXPECTED["test_label_counts"], "test label counts differ"
fp_tr, fp_va, fp_te = fingerprint(tr_rows), fingerprint(va_rows), fingerprint(te_rows)
assert fp_tr == EXPECTED["train_fp"], "TRAIN fingerprint differs from the verified imbalance_free_dataset"
assert fp_te == EXPECTED["test_fp"], "TEST fingerprint differs: the test set is not the one used for VBRI"
print("OK: train 4050 (1350/class, 150/class/question), validation 333, test 333 (225/67/41). Fingerprints match.")

# ------------------------------ augmented dataset (ablation + data figures) --
AUG_OK, RAW_AUG = False, None
_a = find_dataset_dirs("augmented_dataset", AUG_DATASET_DIR_OVERRIDE)
if _a:
    print("USING AUGMENTED DATASET (ablation only):", _a[0])
    RAW_AUG = load_from_disk(str(_a[0]))
    aug_tr = rows_of(RAW_AUG["train"])
    same_eval = fingerprint(rows_of(RAW_AUG["validation"])) == fp_va and fingerprint(rows_of(RAW_AUG["test"])) == fp_te
    if not same_eval:
        print("!!! augmented_dataset has a different validation/test split -> not used")
    else:
        AUG_OK = True
        ac = dict(sorted(Counter(int(r["label"]) for r in aug_tr).items()))
        if len(aug_tr) != EXPECTED_AUG["train_rows"] or ac != EXPECTED_AUG["label_counts"]:
            print(f"!!! augmented_dataset train is {len(aug_tr)} rows {ac}; expected {EXPECTED_AUG}. Used, but check it.")
        else:
            print("OK: augmented_dataset train 4279 (1746/1612/921), same validation and test split.")
else:
    print("!!! augmented_dataset not found: the 'unbalanced' ablation rows and the original-data figures will be skipped.")

# ------------------------------ the 3,258 original answers -------------------
def _is_original(r):
    m = r.get("augmentation_method")
    return m is None or (isinstance(m, float) and math.isnan(m)) or str(m).strip().lower() in ("", "original", "none", "nan")

ORIG_OK, ORIG_ROWS = False, None
if AUG_OK:
    orig_train = [r for r in aug_tr if _is_original(r)]
    ORIG_ROWS = orig_train + va_rows + te_rows
    oc = dict(sorted(Counter(int(r["label"]) for r in ORIG_ROWS).items()))
    otc = dict(sorted(Counter(int(r["label"]) for r in orig_train).items()))
    oq = {q: tuple(sum(1 for r in ORIG_ROWS if int(r["question_id"]) == q and int(r["label"]) == c) for c in (0, 1, 2)) for q in range(1, 10)}
    n_part = len({int(r["participant_id"]) for r in ORIG_ROWS})
    checks = {"rows": len(ORIG_ROWS) == EXPECTED_ORIG["rows"], "classes": oc == EXPECTED_ORIG["class_counts"],
              "train": otc == EXPECTED_ORIG["orig_train_counts"], "per_question": oq == EXPECTED_ORIG["per_question"],
              "participants": n_part == EXPECTED_ORIG["participants"]}
    ORIG_OK = all(checks.values())
    print("Original 3,258 answers rebuilt:" if ORIG_OK else "!!! original data check FAILED:", checks, oc)

DATASET_REPORT = {"path": str(DATA_DIR), "aug_path": str(_a[0]) if _a else None, "aug_ok": AUG_OK, "orig_ok": ORIG_OK,
                  "train_fp": fp_tr, "val_fp": fp_va, "test_fp": fp_te,
                  "train": len(RAW["train"]), "validation": len(RAW["validation"]), "test": len(RAW["test"])}
atomic_write_json(WORK_DIR / "dataset_used.json", DATASET_REPORT)

# keep test info aligned with prediction order
TEST_IDS = [int(r["id"]) for r in te_rows]
TEST_QIDS = [int(r["question_id"]) for r in te_rows]
TEST_Y = [int(r["label"]) for r in te_rows]
TEST_PIDS = [int(r["participant_id"]) for r in te_rows]
TEST_ANS_RAW = [str(r["answer"]).strip() for r in te_rows]
