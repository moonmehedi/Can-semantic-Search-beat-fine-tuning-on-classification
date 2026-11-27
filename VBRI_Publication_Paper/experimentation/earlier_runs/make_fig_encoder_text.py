"""Figure: encoder choice and input text (question+answer vs. answer only).

Values are parsed from the saved output of the encoder benchmark cell in
till_all_model.ipynb (pooled index, k=3, class-weighted vote).

Run:  python make_fig_encoder_text.py      (needs matplotlib + numpy)
Output: figures/encoder_input_text.pdf
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "till_all_model.ipynb"
OUT = HERE / "figures" / "encoder_input_text.pdf"
OUT.parent.mkdir(exist_ok=True)

# Same palette and style as make_figures.py
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

# Result lines look like: "2. MiniLM (Answer-Only)  MiniLM Answer-Only 60.66% 62.84% 42.39% 52.51% 5.39"
ROW = re.compile(
    r"\d\. .*?\)\s+(MiniLM|MPNet|BGE-base|PubMedBERT)\s+(Q\+A|Answer-Only)\s+"
    r"([\d.]+)%\s+[\d.]+%\s+[\d.]+%\s+([\d.]+)%"
)
results = {}
for cell in json.loads(NOTEBOOK.read_text())["cells"]:
    if "Enhanced Semantic Search Benchmark v3" not in "".join(cell["source"]):
        continue
    text = "".join("".join(o.get("text", "")) for o in cell.get("outputs", []))
    for family, fmt, acc, mf1 in ROW.findall(text):
        results[family, fmt] = (float(acc), float(mf1))
assert len(results) == 8, f"expected 8 benchmark rows, found {len(results)}"

families = ["MiniLM", "MPNet", "PubMedBERT", "BGE-base"]
x = np.arange(len(families))
w = 0.38
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
for ax, idx, label in zip(axes, (0, 1), ("Accuracy (%)", "Macro F1 (%)")):
    for offset, fmt, color, name in ((-w / 2, "Q+A", QA, "Question + answer"),
                                     (w / 2, "Answer-Only", ANS, "Answer only")):
        vals = [results[f, fmt][idx] for f in families]
        ax.bar(x + offset, vals, w, color=color, label=name, edgecolor="white", linewidth=1)
        for xi, v in zip(x + offset, vals):
            ax.text(xi, v + 0.8, f"{v:.1f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_title(f"({'ab'[idx]}) {label}")
    ax.set_xticks(x, families)
    ax.set_ylim(0, 78)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Score (%)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT)
print("wrote", OUT)
