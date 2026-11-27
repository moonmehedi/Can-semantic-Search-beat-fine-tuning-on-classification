"""Encoder / input-text figure and the BPD-only paired test, from the final run outputs.

Usage:  python make_fig_encoder_text.py [RETRIEVAL_DIR] [OUTPUT_PDF]
RETRIEVAL_DIR is the notebook's "retrieval" output folder (default: ../experiments/final_run_outputs/retrieval).
Needs matplotlib and numpy.
"""
import csv
import json
import sys
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RETR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "experiments" / "final_run_outputs" / "retrieval"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "figures" / "generated" / "encoder_input_text.pdf"

ANS, QA = "#2a78d6", "#9a9893"
INK, INK2, GRID = "#1f1f1e", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

rows = json.loads((RETR / "retrieval_results.json").read_text(encoding="utf-8"))["rows"]
families = ["MiniLM", "MPNet", "PubMedBERT", "BGE"]
x = np.arange(len(families))
w = 0.38
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
for ax, key, title in zip(axes, ("accuracy", "macro_f1"), ("(a) Accuracy", "(b) Macro F1")):
    for offset, fmt, color, name in ((-w / 2, "Q+A", QA, "Question + answer"), (w / 2, "answer", ANS, "Answer only")):
        vals = [rows[f"Pooled {f} ({fmt})"]["test"][key] for f in families]
        ax.bar(x + offset, vals, w, color=color, label=name, edgecolor="white", linewidth=1)
        for xi, v in zip(x + offset, vals):
            ax.text(xi, v + 0.8, f"{v:.1f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_title(title)
    ax.set_xticks(x, families)
    ax.set_ylim(0, 78)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Score (%)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.07))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
print("wrote", OUT)


def preds(name):
    with open(RETR / "predictions" / f"{name}.csv", encoding="utf-8") as f:
        return [(r["label"], r["pred"]) for r in csv.DictReader(f)]


# exact McNemar test on the BPD answers only: pooled BGE index vs VBRI raw
pooled, vbri = preds("Pooled_BGE_answer"), preds("VBRI_raw")
only_pooled = sum(l == "1" and p == "1" and v != "1" for (l, p), (_, v) in zip(pooled, vbri))
only_vbri = sum(l == "1" and p != "1" and v == "1" for (l, p), (_, v) in zip(pooled, vbri))
n = only_pooled + only_vbri
p_exact = min(1.0, 2 * sum(comb(n, i) for i in range(min(only_pooled, only_vbri) + 1)) / 2 ** n)
print(f"BPD answers found only by VBRI: {only_vbri}, only by the pooled index: {only_pooled}, exact McNemar p = {p_exact:.3f}")
