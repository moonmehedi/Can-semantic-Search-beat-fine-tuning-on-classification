def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def atomic_write_json(path, obj):
    """Write JSON so that a crash can never leave a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default

CLASSES = ["Normal", "BPD", "Inconclusive"]

def confusion(y, p, k=3):
    cm = [[0] * k for _ in range(k)]
    for a, b in zip(y, p):
        cm[int(a)][int(b)] += 1
    return cm

def summarize(y, p):
    """All metrics in %, computed from labels y and predictions p (0=Normal, 1=BPD, 2=Inconclusive)."""
    cm = confusion(y, p)
    n = len(y)
    per = []
    for c in range(3):
        tp = cm[c][c]
        fp = sum(cm[i][c] for i in range(3)) - tp
        fn = sum(cm[c]) - tp
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        per.append((pr, rc, f1, sum(cm[c])))
    tp = cm[1][1]; fn = sum(cm[1]) - tp
    fp = sum(cm[i][1] for i in (0, 2)); tn = n - tp - fn - fp
    safe = lambda a, b: 100.0 * a / b if b else 0.0
    return {
        "n": n, "confusion": cm,
        "accuracy": 100.0 * sum(cm[i][i] for i in range(3)) / n,
        "macro_f1": 100.0 * sum(x[2] for x in per) / 3,
        "weighted_f1": 100.0 * sum(x[2] * x[3] for x in per) / n,
        "bpd_f1": 100.0 * per[1][2], "bpd_precision": 100.0 * per[1][0],
        "bpd_sensitivity": 100.0 * per[1][1], "inconclusive_recall": 100.0 * per[2][1],
        "bin_sensitivity": safe(tp, tp + fn), "bin_specificity": safe(tn, tn + fp),
        "bin_ppv": safe(tp, tp + fp), "bin_npv": safe(tn, tn + fn),
        "bin_accuracy": safe(tp + tn, n),
        "bin_dor": (tp * tn) / (fp * fn) if fp * fn else float("nan"),
    }

def auroc(scores, ybin):
    """AUROC by the rank (Mann-Whitney) method, ties get average rank."""
    n = len(scores)
    n_pos = sum(1 for y in ybin if y); n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    r_pos = sum(ranks[i] for i in range(n) if ybin[i])
    return (r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

def spec_at_sens(scores, ybin, target=0.85):
    """Specificity at the highest threshold whose sensitivity reaches `target`."""
    P = sum(1 for y in ybin if y); N = len(ybin) - P
    if P == 0 or N == 0:
        return float("nan"), None
    for t in sorted(set(scores), reverse=True):
        tp = sum(1 for s, y in zip(scores, ybin) if s >= t and y)
        fp = sum(1 for s, y in zip(scores, ybin) if s >= t and not y)
        if tp / P >= target:
            return 100.0 * (1 - fp / N), t
    return 0.0, None

def fingerprint(rows, keys=("question_id", "label", "answer")):
    lines = sorted("|".join(str(r[k]).strip() for k in keys) for r in rows)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()

def pct_stats(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan"), float("nan")
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)