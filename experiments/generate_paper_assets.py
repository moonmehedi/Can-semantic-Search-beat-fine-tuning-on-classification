from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


THESIS_DIR = Path("Thesis_Paper_editing_version_literature_review")
OUTPUT_DIR = THESIS_DIR / "generated"

DATASET_SUMMARY = [
    ("Participants", "362"),
    ("Open-ended questions", "9"),
    ("Original QA pairs", "3,258"),
    ("Training samples (augmented)", "5,238"),
    ("Validation samples", "333"),
    ("Test samples", "333"),
    ("Classes", "3 (Normal, BPD, Inconclusive)"),
]

MODEL_METRICS = [
    {"Rank": 1, "Model": "ELECTRA", "Accuracy": 0.849850, "F1": 0.851111, "Precision": 0.855018, "Recall": 0.849850},
    {"Rank": 2, "Model": "RoBERTa", "Accuracy": 0.834835, "F1": 0.837797, "Precision": 0.843701, "Recall": 0.834835},
    {"Rank": 3, "Model": "XLNet", "Accuracy": 0.819820, "F1": 0.826749, "Precision": 0.842643, "Recall": 0.819820},
    {"Rank": 4, "Model": "BERT", "Accuracy": 0.777778, "F1": 0.787308, "Precision": 0.813152, "Recall": 0.777778},
    {"Rank": 5, "Model": "DistilBERT", "Accuracy": 0.765766, "F1": 0.776396, "Precision": 0.800074, "Recall": 0.765766},
    {"Rank": 6, "Model": "T5", "Accuracy": 0.699700, "F1": 0.718933, "Precision": 0.787652, "Recall": 0.699700},
    {"Rank": 7, "Model": "DeBERTa", "Accuracy": 0.201201, "F1": 0.067402, "Precision": 0.040482, "Recall": 0.201201},
]

CLASSWISE_RECALL = [
    {"Model": "ELECTRA", "Normal": 0.90, "BPD": 0.78, "Inconclusive": 0.68},
    {"Model": "RoBERTa", "Normal": 0.88, "BPD": 0.81, "Inconclusive": 0.63},
    {"Model": "XLNet", "Normal": 0.84, "BPD": 0.79, "Inconclusive": 0.78},
]


def make_accuracy_figure(output_dir: Path) -> None:
    rows = sorted(MODEL_METRICS, key=lambda row: row["Accuracy"], reverse=True)
    width, height = 2200, 1200
    margin_left, margin_right = 350, 180
    margin_top, margin_bottom = 150, 120
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    row_gap = 24
    bar_height = (chart_height - row_gap * (len(rows) - 1)) / len(rows)
    max_score = 0.9

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    try:
        title_font = ImageFont.truetype("DejaVuSerif.ttf", 52)
        label_font = ImageFont.truetype("DejaVuSerif.ttf", 34)
        tick_font = ImageFont.truetype("DejaVuSerif.ttf", 28)
        value_font = ImageFont.truetype("DejaVuSerif.ttf", 30)
    except OSError:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        tick_font = ImageFont.load_default()
        value_font = ImageFont.load_default()

    draw.text((width / 2, 40), "Test-set Accuracy Comparison Across Models", fill="black", font=title_font, anchor="ma")

    chart_left = margin_left
    chart_right = margin_left + chart_width
    chart_top = margin_top
    chart_bottom = margin_top + chart_height

    axis_color = "#111827"
    grid_color = "#d1d5db"
    draw.line((chart_left, chart_top, chart_left, chart_bottom), fill=axis_color, width=3)
    draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill=axis_color, width=3)

    for tick in range(0, 10):
        score = tick / 10
        x = chart_left + (score / max_score) * chart_width
        draw.line((x, chart_top, x, chart_bottom), fill=grid_color, width=1)
        draw.text((x, chart_bottom + 20), f"{score:.1f}", fill="black", font=tick_font, anchor="ma")

    for idx, row in enumerate(rows):
        y = chart_top + idx * (bar_height + row_gap)
        bar_right = chart_left + (row["Accuracy"] / max_score) * chart_width

        if row["Rank"] == 1:
            color = "#0f766e"
        elif row["Model"] == "DeBERTa":
            color = "#b91c1c"
        else:
            color = "#2563eb"

        draw.rounded_rectangle((chart_left, y, bar_right, y + bar_height), radius=10, fill=color)
        draw.text((chart_left - 20, y + bar_height / 2), row["Model"], fill="black", font=label_font, anchor="rm")
        draw.text((bar_right + 20, y + bar_height / 2), f"{row['Accuracy']:.3f}", fill="black", font=value_font, anchor="lm")

    draw.text((width / 2, height - 35), "Accuracy", fill="black", font=label_font, anchor="ma")
    image.save(output_dir / "test_accuracy_comparison.png")


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def write_results_snippet(output_dir: Path) -> None:
    lines = [
        "% Generated from experiments/bpd-project (2).ipynb",
        "",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Summary of the augmented dataset used for transformer evaluation.}",
        r"\label{tab:dataset_summary}",
        r"\begin{tabular}{ll}",
        r"\hline",
        r"Item & Value \\",
        r"\hline",
    ]
    for item, value in DATASET_SUMMARY:
        lines.append(f"{latex_escape(item)} & {latex_escape(value)} \\\\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{Final augmented test-set performance ranked by weighted F1-score.}",
            r"\label{tab:model_test_metrics}",
            r"\begin{tabular}{lcccc}",
            r"\hline",
            r"Model & Accuracy & F1 & Precision & Recall \\",
            r"\hline",
        ]
    )
    for row in MODEL_METRICS:
        lines.append(
            f"{row['Model']} & {row['Accuracy']:.3f} & {row['F1']:.3f} & {row['Precision']:.3f} & {row['Recall']:.3f} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Class-wise recall for the three strongest models on the augmented test set.}",
            r"\label{tab:classwise_recall}",
            r"\begin{tabular}{lccc}",
            r"\hline",
            r"Model & Normal & BPD & Inconclusive \\",
            r"\hline",
        ]
    )
    for row in CLASSWISE_RECALL:
        lines.append(
            f"{row['Model']} & {row['Normal']:.2f} & {row['BPD']:.2f} & {row['Inconclusive']:.2f} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{figure}[t]",
            r"\centering",
            r"\includegraphics[width=0.9\linewidth]{generated/test_accuracy_comparison.png}",
            r"\caption{Accuracy comparison across all evaluated transformer models on the augmented test set. ELECTRA achieved the highest test accuracy.}",
            r"\label{fig:test_accuracy_comparison}",
            r"\end{figure}",
            "",
        ]
    )
    (output_dir / "paper_results_snippets.tex").write_text("\n".join(lines))


def write_csv(output_dir: Path) -> None:
    rows = ["Rank,Model,Accuracy,F1,Precision,Recall"]
    for row in MODEL_METRICS:
        rows.append(
            f"{row['Rank']},{row['Model']},{row['Accuracy']:.6f},{row['F1']:.6f},{row['Precision']:.6f},{row['Recall']:.6f}"
        )
    (output_dir / "model_test_metrics.csv").write_text("\n".join(rows) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    make_accuracy_figure(OUTPUT_DIR)
    write_results_snippet(OUTPUT_DIR)
    write_csv(OUTPUT_DIR)
    print(f"Generated assets in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
