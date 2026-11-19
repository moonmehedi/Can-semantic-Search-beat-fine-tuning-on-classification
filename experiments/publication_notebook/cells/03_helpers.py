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

# ------------------------------ statistics ----------------------------------
def mcnemar(correct_a, correct_b):
    """McNemar test on paired correctness. b = A wrong & B right, c = A right & B wrong."""
    b = sum(1 for a, bb in zip(correct_a, correct_b) if not a and bb)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_asymptotic": 1.0, "p_exact": 1.0}
    chi2 = (abs(b - c) - 1.0) ** 2 / n                     # with continuity correction
    p_asym = math.erfc(math.sqrt(chi2 / 2.0))
    p_exact = min(1.0, 2.0 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
    return {"b": b, "c": c, "chi2": chi2, "p_asymptotic": p_asym, "p_exact": p_exact}

def bootstrap_ci(y, preds_by_name, metric_keys, n_boot=None, seed=0, diffs=()):
    """Paired percentile bootstrap (95%) over test answers.
    preds_by_name: {name: list of predictions}; diffs: pairs (a, b) for which a-b is also resampled."""
    import numpy as _np
    n_boot = n_boot or N_BOOT
    rng = _np.random.default_rng(seed)
    y = _np.asarray(y)
    P = {k: _np.asarray(v) for k, v in preds_by_name.items()}
    draws = {(name, m): [] for name in P for m in metric_keys}
    ddraws = {(a, b, m): [] for a, b in diffs for m in metric_keys}
    for _ in range(n_boot):
        ix = rng.integers(0, len(y), len(y))
        yy = y[ix].tolist()
        cur = {name: summarize(yy, p[ix].tolist()) for name, p in P.items()}
        for name in P:
            for m in metric_keys:
                draws[(name, m)].append(cur[name][m])
        for a, b in diffs:
            for m in metric_keys:
                ddraws[(a, b, m)].append(cur[a][m] - cur[b][m])
    q = lambda v: (float(_np.percentile(v, 2.5)), float(_np.percentile(v, 97.5)))
    out = {name: {m: {"point": summarize(y.tolist(), P[name].tolist())[m], "ci95": q(draws[(name, m)])} for m in metric_keys} for name in P}
    dout = {f"{a} - {b}": {m: {"point": out[a][m]["point"] - out[b][m]["point"], "ci95": q(ddraws[(a, b, m)]),
                               "p_two_sided": float(min(1.0, 2 * min(_np.mean(_np.asarray(ddraws[(a, b, m)]) <= 0),
                                                                    _np.mean(_np.asarray(ddraws[(a, b, m)]) >= 0))))}
                            for m in metric_keys} for a, b in diffs}
    return out, dout


# ------------------------------ figure style --------------------------------
import matplotlib
if not hasattr(sys, "ps1") and "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_FT, C_VB, C_BASE, C_LEX = "#2a78d6", "#eb6834", "#9a9893", "#c9c7c1"   # fine-tuned, VBRI, dense baseline, lexical
INK, INK2, GRID = "#1f1f1e", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})

def save_fig(fig, name):
    """Save a figure as PDF (for LaTeX) and PNG, show it inline, then close it."""
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png")
    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)
    print("  figure saved:", name)

def draw_cm(ax, cm, title, cmap="Blues"):
    cm = np.asarray(cm)
    rs = cm.sum(1, keepdims=True)
    norm = np.divide(cm, rs, out=np.zeros_like(cm, dtype=float), where=rs != 0)
    ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.55 else INK)
    acc = np.trace(cm) / max(cm.sum(), 1) * 100
    ax.set_title(f"{title}\nacc. {acc:.1f}%, BPD recall {norm[1, 1] * 100:.1f}%", fontsize=9)
    ax.set_xticks(range(3), CLASSES, rotation=20)
    ax.set_yticks(range(3), CLASSES)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
