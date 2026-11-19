from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# Keep Matplotlib fully local and deterministic for this one-off figure export.
matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 200,
        "savefig.dpi": 400,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


CLASS_NAMES = ["Normal", "BPD", "Inconclusive"]
MODEL_ORDER = [
    "ELECTRA",
    "BERT",
    "DistilBERT",
    "RoBERTa",
    "XLNet",
    "DeBERTa",
    "ModernBERT",
    "T5",
]

# Exact counts extracted from the existing vector PDF via pdftotext.
# This preserves the originally reported evaluation results without rerunning training.
CONFUSION_MATRICES = {
    "ELECTRA": np.array([[207, 12, 6], [11, 53, 3], [9, 7, 25]], dtype=int),
    "BERT": np.array([[186, 20, 19], [18, 47, 2], [5, 6, 30]], dtype=int),
    "DistilBERT": np.array([[186, 27, 12], [9, 54, 4], [5, 10, 26]], dtype=int),
    "RoBERTa": np.array([[196, 22, 7], [7, 55, 5], [9, 10, 22]], dtype=int),
    "XLNet": np.array([[204, 15, 6], [10, 56, 1], [6, 8, 27]], dtype=int),
    "DeBERTa": np.array([[225, 0, 0], [67, 0, 0], [41, 0, 0]], dtype=int),
    "ModernBERT": np.array([[212, 8, 5], [24, 41, 2], [13, 4, 24]], dtype=int),
    "T5": np.array([[138, 69, 18], [8, 53, 6], [4, 8, 29]], dtype=int),
}


def make_figure(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13.4, 18.4), facecolor="white")
    gs = fig.add_gridspec(
        nrows=5,
        ncols=2,
        height_ratios=[1, 1, 1, 1, 0.085],
        hspace=0.55,
        wspace=0.28,
    )

    cmap = plt.cm.Blues
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    image = None

    for index, model_name in enumerate(MODEL_ORDER):
        row = index // 2
        col = index % 2
        ax = fig.add_subplot(gs[row, col])

        cm = CONFUSION_MATRICES[model_name]
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

        image = ax.imshow(cm_norm, cmap=cmap, norm=norm)
        ax.set_title(f"{model_name} Confusion Matrix", pad=10, fontweight="bold")
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_box_aspect(1)

        # Draw clean white separators between cells instead of relying on clipped layout artifacts.
        ax.set_xticks(np.arange(-0.5, len(CLASS_NAMES), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(CLASS_NAMES), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.6)
        ax.tick_params(which="minor", bottom=False, left=False)

        for spine in ax.spines.values():
            spine.set_visible(False)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = cm_norm[i, j]
                text_color = "white" if value >= 0.55 else "#111827"
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}\n{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                    fontweight="bold",
                )

    cax = fig.add_subplot(gs[4, :])
    colorbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    colorbar.set_label("Row-normalized proportion", labelpad=8)
    colorbar.set_ticks(np.linspace(0, 1, 6))

    fig.suptitle("Confusion Matrices for All Models", fontsize=16, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.055)

    png_path = output_dir / "all_models_confusion_matrices.png"
    pdf_path = output_dir / "all_models_confusion_matrices.pdf"
    fig.savefig(png_path, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main() -> None:
    root = Path("Thesis_Paper_editing_version_literature_review") / "Figures"
    make_figure(root)


if __name__ == "__main__":
    main()
