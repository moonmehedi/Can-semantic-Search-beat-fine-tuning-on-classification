"""Regenerate the result figures for the VBRI paper from saved experiment outputs.

Every number plotted here is read from (or copied verbatim from) the experiment
outputs in ../experiments:
  * till_all_model.ipynb   - fine-tuning logs (Sept run, Tesla T4), confusion matrices
  * embedding.ipynb         - 9-VDB / global-store results, k-sweep, index build times
  * BPD_9VDB_Complete_Publication_Package/model_artifacts/*.csv

Run:  python make_figures.py      (needs matplotlib + numpy)
Output: figures/*.pdf
"""
import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
HERE = Path(__file__).resolve().parent
EXP = HERE
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# --- Style ------------------------------------------------------------------
FT = "#2a78d6"      # fine-tuned encoders
VB = "#eb6834"      # VBRI (proposed)
BASE = "#9a9893"    # retrieval / lexical baselines
INK, INK2, GRID = "#1f1f1e", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})
CLASSES = ["Normal", "BPD", "Inconclusive"]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


# --- 1. Training / validation loss curves (parsed from notebook logs) -------
TRAIN_CELLS = {49: "ELECTRA", 56: "BERT", 63: "DistilBERT", 70: "RoBERTa",
               85: "XLNet", 92: "DeBERTa-v3", 94: "ModernBERT", 104: "T5-small"}


def parse_log_table(text):
    """Parse a wrapped pandas DataFrame printout into {column: {row: value}}."""
    cols, data = [], {}
    for line in text.splitlines():
        s = line.strip().rstrip("\\").strip()
        if not s:
            continue
        toks = s.split()
        if not re.fullmatch(r"\d+", toks[0]):          # header line
            cols = toks
            continue
        row = int(toks[0])
        for c, v in zip(cols, toks[1:]):
            try:
                data.setdefault(c, {})[row] = float(v)
            except ValueError:
                pass
    return data


def loss_history(cell):
    for o in cell.get("outputs", []):
        t = "".join(o.get("text", [])) if "text" in o else "".join(o.get("data", {}).get("text/plain", []))
        if "eval_loss" in t and "epoch" in t:
            d = parse_log_table(t)
            ep, tr, va = {}, {}, {}
            for r, e in d["epoch"].items():
                if r in d.get("loss", {}) and not np.isnan(d["loss"][r]):
                    tr[e] = d["loss"][r]
                if r in d.get("eval_loss", {}) and not np.isnan(d["eval_loss"][r]):
                    va[e] = d["eval_loss"][r]
            return tr, va, d.get("train_runtime", {})
    return None


nb = json.load(open(EXP / "till_all_model.ipynb"))
fig, axes = plt.subplots(2, 4, figsize=(11, 4.6))
runtimes = {}
for ax, (ci, name) in zip(axes.flat, TRAIN_CELLS.items()):
    tr, va, rt = loss_history(nb["cells"][ci])
    rt = [v for v in rt.values() if not np.isnan(v)]
    runtimes[name] = rt[-1] if rt else None
    e1, e2 = sorted(tr), sorted(va)
    ax.plot(e1, [tr[e] for e in e1], color=FT, lw=2, marker="o", ms=4, label="Training loss")
    ax.plot(e2, [va[e] for e in e2], color=INK, lw=2, ls="--", marker="s", ms=4, label="Validation loss")
    best = min(e2, key=lambda e: va[e])
    ax.axvline(best, color=VB, lw=1, ls=":")
    ax.set_title(name, color=INK)
    ax.set_xlabel("Epoch")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
axes[0, 0].set_ylabel("Cross-entropy loss")
axes[1, 0].set_ylabel("Cross-entropy loss")
h, l = axes[0, 0].get_legend_handles_labels()
h.append(matplotlib.lines.Line2D([], [], color=VB, ls=":", lw=1))
l.append("Selected checkpoint (min. validation loss)")
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "ft_loss_curves")
print("train runtimes (s):", runtimes)

# --- 2. Confusion matrices of fine-tuned models (Sept run, test N=333) ------
CM_FT = {
    "ModernBERT": [[199, 17, 9], [7, 58, 2], [5, 7, 29]],
    "ELECTRA": [[192, 20, 13], [7, 58, 2], [6, 7, 28]],
    "RoBERTa": [[184, 22, 19], [6, 57, 4], [3, 9, 29]],
    "BERT": [[180, 31, 14], [10, 54, 3], [6, 4, 31]],
    "XLNet": [[178, 38, 9], [9, 57, 1], [7, 11, 23]],
    "DistilBERT": [[163, 41, 21], [5, 56, 6], [5, 6, 30]],
    "DeBERTa-v3": [[225, 0, 0], [67, 0, 0], [41, 0, 0]],
    "T5-small": [[108, 114, 3], [38, 29, 0], [22, 18, 1]],
}
CM_VBRI = {
    "Global pooled store": [[156, 48, 21], [7, 50, 10], [9, 12, 20]],
    "9-VDB (raw, k=5)": [[169, 41, 15], [7, 57, 3], [13, 12, 16]],
    "9-VDB Hybrid": [[166, 40, 19], [6, 58, 3], [10, 12, 19]],
}


def draw_cm(ax, cm, title, cmap="Blues"):
    cm = np.array(cm)
    norm = cm / cm.sum(1, keepdims=True)
    ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if norm[i, j] > 0.55 else INK)
    acc = np.trace(cm) / cm.sum() * 100
    ax.set_title(f"{title}\nacc. {acc:.1f}%, BPD recall {norm[1, 1]*100:.1f}%", fontsize=9)
    ax.set_xticks(range(3), CLASSES, rotation=20)
    ax.set_yticks(range(3), CLASSES)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


fig, axes = plt.subplots(2, 4, figsize=(11, 6))
for ax, (n, cm) in zip(axes.flat, CM_FT.items()):
    draw_cm(ax, cm, n)
for ax in axes[:, 0]:
    ax.set_ylabel("True label")
for ax in axes[1]:
    ax.set_xlabel("Predicted label")
fig.tight_layout()
save(fig, "ft_confusion_matrices")

fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
draw_cm(axes[0], CM_FT["ModernBERT"], "ModernBERT (fine-tuned)")
for ax, (n, cm) in zip(axes[1:], CM_VBRI.items()):
    draw_cm(ax, cm, n, cmap="Oranges")
axes[0].set_ylabel("True label")
for ax in axes:
    ax.set_xlabel("Predicted label")
fig.tight_layout()
save(fig, "head_to_head_confusion")

# --- 3. Effectiveness comparison --------------------------------------------
rows = [  # name, group, accuracy, macro-F1, BPD sensitivity
    ("ModernBERT", "ft", 85.89, 80.00, 86.57), ("ELECTRA", "ft", 83.48, 77.00, 86.57),
    ("RoBERTa", "ft", 81.08, 75.00, 85.07), ("BERT", "ft", 79.58, 75.00, 80.60),
    ("XLNet", "ft", 77.48, 71.00, 85.07), ("DistilBERT", "ft", 74.77, 70.00, 83.58),
    ("Global BGE", "b", 67.87, 59.52, 74.63), ("MPNet (Q+A)", "b", 67.57, 57.28, 68.66),
    ("BM25", "b", 63.66, 48.79, 58.21), ("TF-IDF", "b", 59.76, 48.89, 52.24),
    ("9-VDB raw", "v", 72.67, 62.91, 85.07), ("9-VDB Hybrid", "v", 72.97, 64.48, 86.57),
]
col = {"ft": FT, "b": BASE, "v": VB}
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
y = np.arange(len(rows))[::-1]
for ax, k, lab in zip(axes, [2, 3, 4], ["Accuracy (%)", "Macro F1 (%)", "BPD sensitivity (%)"]):
    vals = [r[k] for r in rows]
    ax.barh(y, vals, color=[col[r[1]] for r in rows], height=0.7, edgecolor="white", linewidth=1)
    for yi, v in zip(y, vals):
        ax.text(v + 1, yi, f"{v:.1f}", va="center", fontsize=7.5, color=INK2)
    ax.set_xlim(0, 105)
    ax.set_xlabel(lab)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(y, [r[0] for r in rows])
handles = [matplotlib.patches.Patch(color=c) for c in (FT, BASE, VB)]
fig.legend(handles, ["Fine-tuned encoder", "Retrieval baseline (global / lexical)", "VBRI (ours, training-free)"],
           loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.05, 1, 1))
save(fig, "effectiveness_comparison")

# --- 4. Cost comparison ------------------------------------------------------
# training time: Trainer train_runtime (s) on Tesla T4; VBRI: sum of 9 index encodings.
# inference: eval_runtime / 333 validation samples (steady-state epochs), batched, T4.
# VBRI inference: BGE query encoding (0.77 s / 333) + 9-VDB search (0.08 ms).
# storage: task-specific weights in fp32 (params x 4 B) vs. 4,050 x 768 x 4 B index.
cost = [  # name, group, train_s, infer_ms, storage_MB
    ("ModernBERT", "ft", 179.2, 4.10, 596), ("ELECTRA", "ft", 1061.7, 9.46, 440),
    ("RoBERTa", "ft", 667.8, 8.62, 500), ("BERT", "ft", 1066.0, 8.96, 440),
    ("XLNet", "ft", 665.8, 12.55, 468), ("DistilBERT", "ft", 474.9, 4.54, 264),
    ("VBRI (9-VDB)", "v", 12.25, 2.39, 12.4),
]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
y = np.arange(len(cost))[::-1]
specs = [(2, "Training / index-build time (s, log)", True, "{:.0f}"),
         (3, "Inference time per response (ms)", False, "{:.2f}"),
         (4, "Task-specific storage (MB, log)", True, "{:.1f}")]
for ax, (k, lab, lg, fmt) in zip(axes, specs):
    vals = [c[k] for c in cost]
    ax.barh(y, vals, color=[col[c[1]] for c in cost], height=0.7, edgecolor="white", linewidth=1)
    if lg:
        ax.set_xscale("log")
    for yi, v in zip(y, vals):
        ax.text(v * (1.12 if lg else 1) + (0 if lg else 0.25), yi, fmt.format(v), va="center", fontsize=7.5, color=INK2)
    ax.set_xlabel(lab)
    ax.grid(axis="y", visible=False)
axes[0].set_xlim(5, 5000)
axes[1].set_xlim(0, 15)
axes[2].set_xlim(5, 2000)
axes[0].set_yticks(y, [c[0] for c in cost])
fig.tight_layout()
save(fig, "cost_comparison")

# --- 5. Ablation: isolation x class balance, then calibration and guard -----
stages = ["Global store\n(unbalanced)", "9-VDB\n(unbalanced)", "Global store\n(balanced)",
          "9-VDB\n(balanced)", "+ margin\ncalibration", "+ C-SSRS\nguard"]
acc = [65.47, 65.77, 67.87, 72.67, 72.67, 72.97]
mf1 = [57.07, 56.25, 59.52, 62.91, 64.16, 64.48]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharex=True)
x = np.arange(len(stages))
cols = [BASE, BASE, BASE, VB, VB, VB]
for ax, vals, lab in zip(axes, [acc, mf1], ["Accuracy (%)", "Macro F1 (%)"]):
    ax.bar(x, vals, color=cols, width=0.65, edgecolor="white", linewidth=1)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.6, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_ylim(50, 77)
    ax.set_ylabel(lab)
    ax.set_xticks(x, stages, fontsize=7.5)
    ax.grid(axis="x", visible=False)
fig.tight_layout()
save(fig, "ablation_progression")

# --- 6. k-sweep (embedding.ipynb, BGE-base, balanced stores) ----------------
ks = [1, 3, 5, 7, 9, 11, 15]
g_f1 = [57.26, 59.08, 59.52, 57.43, 58.51, 59.56, 56.84]
v_f1 = [60.43, 63.16, 62.91, 61.31, 57.54, 57.91, 54.57]
g_ct = [33.9, 35.6, 35.7, 36.3, 37.1, 37.9, 39.0]
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
axes[0].plot(ks, g_f1, color=BASE, lw=2, marker="o", ms=5, label="Global pooled store")
axes[0].plot(ks, v_f1, color=VB, lw=2, marker="o", ms=5, label="9-VDB (isolated)")
axes[0].set_xlabel("Number of neighbours k")
axes[0].set_ylabel("Macro F1 (%)")
axes[0].legend(loc="lower left")
axes[1].plot(ks, g_ct, color=BASE, lw=2, marker="o", ms=5, label="Global pooled store")
axes[1].plot(ks, [0] * len(ks), color=VB, lw=2, marker="o", ms=5, label="9-VDB (0% by design)")
axes[1].set_xlabel("Number of neighbours k")
axes[1].set_ylabel("Neighbours from wrong criterion (%)")
axes[1].set_ylim(-2, 45)
axes[1].legend(loc="center right")
for ax in axes:
    ax.set_xticks(ks)
fig.tight_layout()
save(fig, "k_sweep")
