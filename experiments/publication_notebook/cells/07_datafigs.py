def make_data_figures():
    stats = {}
    # ---- Figure: dataset overview (original 3,258 answers) ----
    if ORIG_OK:
        oc = Counter(int(r["label"]) for r in ORIG_ROWS)
        n = len(ORIG_ROWS)
        props = np.array([[sum(1 for r in ORIG_ROWS if int(r["question_id"]) == q and int(r["label"]) == c) for c in (0, 1, 2)]
                          for q in range(1, 10)], dtype=float)
        props = props / props.sum(1, keepdims=True)
        lens = {c: [len(str(r["answer"]).split()) for r in ORIG_ROWS if int(r["label"]) == c] for c in (0, 1, 2)}
        stats["original"] = {"rows": n, "class_counts": {CLASSES[c]: oc[c] for c in (0, 1, 2)},
                             "question_label_proportions": {f"Q{q}": dict(zip(CLASSES, props[q - 1].round(4).tolist())) for q in range(1, 10)},
                             "mean_words": {CLASSES[c]: float(np.mean(lens[c])) for c in (0, 1, 2)}}
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1, 1.5]})
        ax = axes[0]
        vals = [oc[c] for c in (0, 1, 2)]
        bars = ax.bar(CLASSES, vals, color=[C_BASE, C_VB, C_FT], width=0.62, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 25, f"{v:,}\n({v / n * 100:.1f}%)", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_ylabel("Question-answer pairs")
        ax.set_ylim(0, max(vals) * 1.25)
        ax.set_title(f"(a) Class distribution (N = {n:,})")
        ax.grid(axis="x", visible=False)
        ax = axes[1]
        im = ax.imshow(props, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(9):
            for j in range(3):
                ax.text(j, i, f"{props[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if props[i, j] > 0.55 else INK)
        ax.set_xticks(range(3), CLASSES)
        ax.set_yticks(range(9), [f"Q{q} {DSM5[q]}" for q in range(1, 10)], fontsize=8)
        ax.set_title("(b) Label proportion per question")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.tight_layout()
        save_fig(fig, "fig_data_overview")
    else:
        print("skipped fig_data_overview (original data not available)")

    # ---- Figure: the training views (what each model learns from / stores) ----
    views = {}
    if ORIG_OK:
        views["Original training split"] = Counter(int(r["label"]) for r in ORIG_ROWS[: sum(1 for r in aug_tr if _is_original(r))])
    if AUG_OK:
        views["Augmented, unbalanced\n(ablation only)"] = Counter(int(r["label"]) for r in aug_tr)
    views["Balanced, used by all models\n(150 per class per question)"] = Counter(int(r["label"]) for r in tr_rows)
    share = np.array([[100.0 * sum(1 for r in tr_rows if int(r["question_id"]) == q and int(r["label"]) == c and not _is_original(r)) / 150.0
                       for c in (0, 1, 2)] for q in range(1, 10)])
    stats["training_views"] = {k.replace("\n", " "): {CLASSES[c]: v[c] for c in (0, 1, 2)} for k, v in views.items()}
    stats["balanced_store_augmented_share_pct"] = {f"Q{q}": dict(zip(CLASSES, share[q - 1].round(1).tolist())) for q in range(1, 10)}
    stats["balanced_store_original_rows"] = sum(1 for r in tr_rows if _is_original(r))
    stats["balanced_store_aug_methods"] = dict(Counter(str(r.get("augmentation_method")) for r in tr_rows if not _is_original(r)))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    names = list(views)
    x = np.arange(3)
    w = 0.8 / len(names)
    cols = [C_LEX, C_BASE, C_VB][-len(names):]
    for i, nm in enumerate(names):
        v = [views[nm][c] for c in (0, 1, 2)]
        b = ax.bar(x + (i - (len(names) - 1) / 2) * w, v, width=w * 0.95, color=cols[i], label=nm, edgecolor="white")
        for bb, vv in zip(b, v):
            ax.text(bb.get_x() + bb.get_width() / 2, vv + 20, str(vv), ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("Training answers")
    ax.set_title("(a) Class counts of the training views")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    im = ax.imshow(share, cmap="Oranges", vmin=0, vmax=100, aspect="auto")
    for i in range(9):
        for j in range(3):
            ax.text(j, i, f"{share[i, j]:.0f}%", ha="center", va="center", fontsize=8, color="white" if share[i, j] > 55 else INK)
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(9), [f"Q{q}" for q in range(1, 10)])
    ax.set_title("(b) Augmented share of the balanced set")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, label="% augmented text")
    fig.tight_layout()
    save_fig(fig, "fig_training_views")
    atomic_write_json(PAPER_DIR / "data_stats.json", stats)
    return stats

if RUN_DATA_FIGURES:
    DATA_STATS = make_data_figures()
    print(json.dumps({k: v for k, v in DATA_STATS.items() if k != "question_label_proportions"}, indent=1)[:1500])
