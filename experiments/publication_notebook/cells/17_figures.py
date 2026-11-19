import matplotlib.ticker, matplotlib.lines, matplotlib.patches

def _ft_order(names):
    order = MODELS + ["DeBERTa-fp32"]
    return sorted(names, key=lambda n: order.index(n) if n in order else 99)

def make_result_figures():
    by = ft_by_model()
    R, A = RETR, ANALYSIS
    s0 = SEEDS[0]
    best = A["best_finetuned"]["model"] if A else None

    # ---- 1. loss curves (first seed) ----
    hist = {}
    for m in _ft_order(by):
        h = read_json(RUNS_DIR / f"seed{s0}" / m / "history.json", None)
        if h:
            hist[m] = h
    if hist:
        n = len(hist); cols = 4; rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(11, 2.6 * rows), squeeze=False)
        for ax, (m, h) in zip(axes.flat, hist.items()):
            tr = [(e["epoch"], e["loss"]) for e in h if "loss" in e and "eval_loss" not in e and "train_runtime" not in e]
            va = [(e["epoch"], e["eval_loss"]) for e in h if "eval_loss" in e]
            ax.plot(*zip(*tr), color=C_FT, lw=2, marker="o", ms=3.5, label="Training loss") if tr else None
            ax.plot(*zip(*va), color=INK, lw=2, ls="--", marker="s", ms=3.5, label="Validation loss") if va else None
            if va:
                ax.axvline(min(va, key=lambda t: t[1])[0], color=C_VB, lw=1, ls=":")
            ax.set_title(m)
            ax.set_xlabel("Epoch")
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        for ax in axes.flat[n:]:
            ax.axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel("Cross-entropy loss")
        h_, l_ = axes[0, 0].get_legend_handles_labels()
        h_.append(matplotlib.lines.Line2D([], [], color=C_VB, ls=":", lw=1)); l_.append("Kept checkpoint (lowest validation loss)")
        fig.legend(h_, l_, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        save_fig(fig, "ft_loss_curves")

    # ---- 2. fine-tuned confusion matrices (first seed) ----
    cms = {m: rs[0]["test"]["confusion"] for m, rs in ((m, [r for r in by[m] if r["seed"] == s0]) for m in _ft_order(by)) if rs}
    if cms:
        n = len(cms); cols = 4; rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(11, 3.0 * rows), squeeze=False)
        for ax, (m, cm) in zip(axes.flat, cms.items()):
            draw_cm(ax, cm, m)
        for ax in axes.flat[n:]:
            ax.axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel("True label")
        fig.tight_layout()
        save_fig(fig, "ft_confusion_matrices")

    if R:
        pc = R["per_criterion"]
        qs = [str(q) for q in range(1, 10)]
        # ---- 3. per-criterion macro-F1 and contamination ----
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
        x = np.arange(9)
        axes[0].bar(x - 0.2, [pc[q]["pooled"]["macro_f1"] for q in qs], 0.38, color=C_BASE, label="Pooled index", edgecolor="white")
        axes[0].bar(x + 0.2, [pc[q]["vbri"]["macro_f1"] for q in qs], 0.38, color=C_VB, label="VBRI (9 isolated databases)", edgecolor="white")
        axes[0].set_ylabel("Macro F1 (%)"); axes[0].set_ylim(0, 100); axes[0].legend(fontsize=8, loc="upper left")
        axes[0].set_title("(a) Macro F1 per criterion")
        cont = [pc[q]["pooled_contamination_pct"] for q in qs]
        axes[1].bar(x, cont, 0.6, color=C_BASE, edgecolor="white")
        mc = R["rows"]["VBRI raw"]["k"]
        mean_c = float(np.mean(cont))
        axes[1].axhline(mean_c, color=INK, ls="--", lw=1)
        axes[1].text(-0.45, mean_c + 1, f"mean {mean_c:.1f}%", ha="left", va="bottom", fontsize=8, color=INK2)
        axes[1].set_ylabel("Neighbours from another criterion (%)"); axes[1].set_ylim(0, max(60, max(cont) + 10))
        axes[1].set_title(f"(b) Contamination of the pooled index (k = {mc})")
        for ax in axes:
            ax.set_xticks(x, [f"Q{q}" for q in qs]); ax.grid(axis="x", visible=False)
        fig.tight_layout()
        save_fig(fig, "contamination_per_criterion")

        # ---- 4. ablation ----
        ab = R["ablation"]
        steps = [(k, v["test"]) for k, v in ab.items()]
        steps += [("+ uncertainty margin", R["rows"]["VBRI calibrated"]["test"]), ("+ C-SSRS rule (hybrid)", R["rows"]["VBRI hybrid"]["test"])]
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
        y = np.arange(len(steps))[::-1]
        colr = [C_VB if ("Isolated, balanced" in k or k.startswith("+")) else C_BASE for k, _ in steps]
        for ax, key, lab in zip(axes, ("accuracy", "macro_f1", "bpd_sensitivity"), ("Accuracy (%)", "Macro F1 (%)", "BPD sensitivity (%)")):
            vals = [m[key] for _, m in steps]
            ax.barh(y, vals, color=colr, height=0.68, edgecolor="white")
            for yy, v in zip(y, vals):
                ax.text(v + 0.8, yy, f"{v:.1f}", va="center", fontsize=7.5, color=INK2)
            ax.set_xlim(0, 105); ax.set_xlabel(lab); ax.grid(axis="y", visible=False)
        axes[0].set_yticks(y, [k for k, _ in steps], fontsize=8)
        fig.tight_layout()
        save_fig(fig, "ablation_progression")

        # ---- 5. k-sweep ----
        ks = R["k_sweep"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
        for nm, colr_ in (("Pooled, balanced", C_BASE), ("Isolated, balanced (VBRI)", C_VB)):
            kk = sorted(int(k) for k in ks[nm])
            axes[0].plot(kk, [ks[nm][str(k)]["val_macro_f1"] for k in kk], color=colr_, lw=2, ls="--", marker="o", ms=4, label=f"{nm}: validation")
            axes[0].plot(kk, [ks[nm][str(k)]["test_macro_f1"] for k in kk], color=colr_, lw=2, marker="s", ms=4, label=f"{nm}: test")
            axes[1].plot(kk, [ks[nm][str(k)]["test_contamination_pct"] for k in kk], color=colr_, lw=2, marker="o", ms=4, label=nm)
        axes[0].axvline(R["rows"]["VBRI raw"]["k"], color=INK, lw=1, ls=":")
        axes[0].set_ylabel("Macro F1 (%)"); axes[0].legend(fontsize=7)
        axes[0].set_title(f"(a) k chosen on validation: k = {R['rows']['VBRI raw']['k']}")
        axes[1].set_ylabel("Neighbours from another criterion (%)"); axes[1].set_ylim(-2, 60); axes[1].legend(fontsize=7)
        axes[1].set_title("(b) Contamination")
        for ax in axes:
            ax.set_xlabel("Number of neighbours k"); ax.set_xticks(K_GRID)
        fig.tight_layout()
        save_fig(fig, "k_sweep")

        # ---- 6. effectiveness comparison ----
        rows_ = []
        for m in _ft_order(by):
            rs = by[m]
            rows_.append((f"{m}", "ft", *[pct_stats([r["test"][k] for r in rs]) for k in ("accuracy", "macro_f1", "bpd_sensitivity")]))
        dense = [k for k, v in R["rows"].items() if v["family"] == "pooled_dense"]
        best_fmt = {}
        for k in dense:
            enc = R["rows"][k]["encoder"]
            if enc not in best_fmt or R["rows"][k]["val"]["macro_f1"] > R["rows"][best_fmt[enc]]["val"]["macro_f1"]:
                best_fmt[enc] = k
        for k in list(best_fmt.values()) + [k for k, v in R["rows"].items() if v["family"] == "lexical"]:
            rows_.append((k, "base", *[(R["rows"][k]["test"][x], 0.0) for x in ("accuracy", "macro_f1", "bpd_sensitivity")]))
        for k in ("VBRI raw", "VBRI calibrated", "VBRI hybrid"):
            rows_.append((k, "vb", *[(R["rows"][k]["test"][x], 0.0) for x in ("accuracy", "macro_f1", "bpd_sensitivity")]))
        colmap = {"ft": C_FT, "base": C_BASE, "vb": C_VB}
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 0.33 * len(rows_) + 1.3), sharey=True)
        y = np.arange(len(rows_))[::-1]
        for j, (ax, lab) in enumerate(zip(axes, ("Accuracy (%)", "Macro F1 (%)", "BPD sensitivity (%)"))):
            mu = [r[2 + j][0] for r in rows_]; sd = [r[2 + j][1] for r in rows_]
            ax.barh(y, mu, color=[colmap[r[1]] for r in rows_], height=0.7, edgecolor="white")
            eb = [i for i, v in enumerate(sd) if v > 0]
            if eb:
                ax.errorbar([mu[i] for i in eb], [y[i] for i in eb], xerr=[sd[i] for i in eb], fmt="none",
                            ecolor=INK2, elinewidth=0.8, capsize=2)
            for yy, v, s_ in zip(y, mu, sd):
                ax.text(v + s_ + 1, yy, f"{v:.1f}", va="center", fontsize=7, color=INK2)
            ax.set_xlim(0, 108); ax.set_xlabel(lab); ax.grid(axis="y", visible=False)
        axes[0].set_yticks(y, [r[0] for r in rows_], fontsize=8)
        handles = [matplotlib.patches.Patch(color=c) for c in (C_FT, C_BASE, C_VB)]
        fig.legend(handles, [f"Fine-tuned encoder (mean ± sd, {len(SEEDS)} seed{'s' if len(SEEDS) > 1 else ''})", "Pooled retrieval baseline", "VBRI (training-free)"],
                   loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        save_fig(fig, "effectiveness_comparison")

        # ---- 7. head-to-head confusion ----
        panels = []
        if best and any(r["seed"] == s0 for r in by.get(best, [])):
            panels.append((f"{best} (fine-tuned, seed {s0})", [r for r in by[best] if r["seed"] == s0][0]["test"]["confusion"], "Blues"))
        pooled_name = "Pooled BGE (answer)" if "Pooled BGE (answer)" in R["rows"] else f"Pooled {VBRI_ENCODER} (answer)"
        for nm in (pooled_name, "VBRI raw", "VBRI hybrid"):
            panels.append((nm, R["rows"][nm]["test"]["confusion"], "Oranges"))
        fig, axes = plt.subplots(1, len(panels), figsize=(2.85 * len(panels), 3.2), squeeze=False)
        for ax, (t, cm, cmap) in zip(axes[0], panels):
            draw_cm(ax, cm, t, cmap)
            ax.set_xlabel("Predicted label")
        axes[0, 0].set_ylabel("True label")
        fig.tight_layout()
        save_fig(fig, "head_to_head_confusion")

        # ---- 8. calibration before / after ----
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
        c = R["calibration"]["chosen"]
        draw_cm(axes[0], R["rows"]["VBRI raw"]["test"]["confusion"], "VBRI raw", "Oranges")
        draw_cm(axes[1], R["rows"]["VBRI calibrated"]["test"]["confusion"], rf"Calibrated ($\tau$={c['tau']}, $\delta$={c['delta']}, $\theta$={c['theta']})", "Oranges")
        axes[0].set_ylabel("True label")
        for ax in axes:
            ax.set_xlabel("Predicted label")
        fig.tight_layout()
        save_fig(fig, "calibration_before_after")

        # ---- 9. per-criterion confusion matrices of VBRI ----
        fig, axes = plt.subplots(3, 3, figsize=(9.5, 9))
        for ax, q in zip(axes.flat, qs):
            draw_cm(ax, pc[q]["vbri"]["confusion"], f"Q{q} {DSM5[int(q)]}", "Oranges")
        fig.tight_layout()
        save_fig(fig, "vdb_confusion_per_criterion")

    # ---- 10. cost comparison ----
    inf = {r["model"]: r for r in (read_json(WORK_DIR / "inference_benchmark.json", []) or [])}
    vt = VBRI_TIME
    if by and vt:
        names = [m for m in _ft_order(by) if not m.endswith("-fp32")]
        tr = [pct_stats([r["timing"]["effective_s"] for r in by[m]]) for m in names]
        stor = [next((r.get("weights_mb") for r in by[m] if r.get("weights_mb")), None) for m in names]
        g1 = [inf.get(m, {}).get("gpu_bs1_median_ms", float("nan")) for m in names]
        c1 = [inf.get(m, {}).get("cpu_bs1_median_ms", float("nan")) for m in names]
        names.append("VBRI"); tr.append((vt["index_build_s_median"], 0.0)); stor.append(vt["index_mb"])
        g1.append(vt.get("gpu_bs1_total_ms_median", float("nan"))); c1.append(vt.get("cpu_bs1_total_ms_median", float("nan")))
        cols = [C_FT] * (len(names) - 1) + [C_VB]
        fig, axes = plt.subplots(1, 4, figsize=(12.5, 0.36 * len(names) + 1.4), sharey=True)
        y = np.arange(len(names))[::-1]
        specs = [([t[0] for t in tr], [t[1] for t in tr], "Training / index build (s, log)", True, "{:.0f}"),
                 (g1, None, "GPU latency, batch 1 (ms)", False, "{:.1f}"),
                 (c1, None, "CPU latency, batch 1 (ms)", False, "{:.0f}"),
                 ([s if s else float("nan") for s in stor], None, "Task-specific storage (MB, log)", True, "{:.0f}")]
        for ax, (vals, err, lab, lg, fmt) in zip(axes, specs):
            ax.barh(y, vals, color=cols, height=0.7, edgecolor="white")
            if err:
                eb = [i for i, e in enumerate(err) if e and e > 0]
                if eb:
                    ax.errorbar([vals[i] for i in eb], [y[i] for i in eb], xerr=[err[i] for i in eb], fmt="none",
                                ecolor=INK2, elinewidth=0.8, capsize=2)
            if not [v for v in vals if v == v]:
                ax.text(0.5, 0.5, "not measured\n(no GPU in this run)", transform=ax.transAxes, ha="center", va="center", fontsize=8, color=INK2)
            if lg:
                ax.set_xscale("log")
            for yy, v in zip(y, vals):
                if v == v:
                    ax.text(v * (1.15 if lg else 1.0) + (0 if lg else max(x for x in vals if x == x) * 0.02), yy, fmt.format(v),
                            va="center", fontsize=7, color=INK2)
            ax.set_xlabel(lab); ax.grid(axis="y", visible=False)
            finite = [v for v in vals if v == v]
            if finite:
                ax.set_xlim((min(finite) / 2, max(finite) * 4) if lg else (0, max(finite) * 1.25))
        axes[0].set_yticks(y, names, fontsize=8)
        fig.tight_layout()
        save_fig(fig, "cost_comparison")

    if A:
        # ---- 11. participant level ----
        P = A["participant"]
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3))
        rng = np.random.default_rng(0)
        tc = np.array(P["true_counts"])
        for nm, col, off in (("VBRI raw", C_VB, -0.12), (best, C_FT, 0.12)):
            if nm in P["models"]:
                pcnt = np.array(P["models"][nm]["counts"])
                axes[0].scatter(tc + off + rng.uniform(-0.05, 0.05, len(tc)), pcnt + rng.uniform(-0.1, 0.1, len(tc)), s=10, alpha=0.5,
                                color=col, label=f"{nm} (r = {P['models'][nm]['pearson_r']:.2f})")
        mx = max(tc.max(), max(max(v["counts"]) for v in P["models"].values())) + 0.5
        axes[0].plot([0, mx], [0, mx], color=INK2, ls="--", lw=1)
        axes[0].set_xlabel("Expert BPD criteria per participant (test answers)"); axes[0].set_ylabel("Predicted BPD criteria")
        axes[0].legend(fontsize=7); axes[0].set_title("(a) Agreement per participant")
        nms = list(P["models"])
        maes = [P["models"][n]["mae"] for n in nms]
        axes[1].barh(np.arange(len(nms))[::-1], maes, color=[C_FT if n == best else (C_VB if n.startswith("VBRI") else C_BASE) for n in nms],
                     height=0.6, edgecolor="white")
        for yy, v in zip(np.arange(len(nms))[::-1], maes):
            axes[1].text(v + 0.005, yy, f"{v:.3f}", va="center", fontsize=8, color=INK2)
        axes[1].set_yticks(np.arange(len(nms))[::-1], nms, fontsize=8); axes[1].set_xlabel("Mean absolute error (criteria)")
        axes[1].set_title(f"(b) Error per participant (n = {P['n_participants']})"); axes[1].grid(axis="y", visible=False)
        fig.tight_layout()
        save_fig(fig, "patient_level_agreement")

        # ---- 12. ROC, BPD vs rest ----
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        colr = {best: C_FT, "VBRI raw": C_VB, "VBRI calibrated": "#b5451b", "Pooled BGE (answer)": C_BASE}
        for nm, d in A["roc"].items():
            ax.plot(d["fpr"], d["tpr"], color=colr.get(nm, INK2), lw=2, label=f"{nm} (AUROC {d['auroc']:.3f})")
            op = d["operating_point"]
            ax.plot(1 - op["spec"] / 100, op["sens"] / 100, marker="o", ms=6, color=colr.get(nm, INK2), mec="white")
        ax.plot([0, 1], [0, 1], color=GRID, lw=1)
        ax.axhline(0.85, color=INK2, lw=0.8, ls=":")
        ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity (BPD)")
        ax.legend(fontsize=7, loc="lower right"); ax.set_title("BPD vs rest (dots = operating points)")
        fig.tight_layout()
        save_fig(fig, "roc_bpd")

if RUN_FIGURES:
    make_result_figures()
    print("\nAll figures in", FIG_DIR)
