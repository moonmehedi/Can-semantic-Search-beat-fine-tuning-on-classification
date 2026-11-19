def tex_escape(s):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
           "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(ch, ch) for ch in str(s))

def _ms(vals, d=2):
    m, s = pct_stats(vals)
    if m != m:
        return "--"
    return f"{m:.{d}f}" if len(vals) < 2 else f"{m:.{d}f} $\\pm$ {s:.{d}f}"

def _w(name, lines):
    (TAB_DIR / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  table saved:", name)

def make_tables():
    by = ft_by_model()
    R, A, vt = RETR, ANALYSIS, VBRI_TIME
    inf = {r["model"]: r for r in (read_json(WORK_DIR / "inference_benchmark.json", []) or [])}
    NUM = {"code_version": CODE_VERSION, "seeds": SEEDS, "gpu": GPU_NAME, "env": ENV, "dataset": DATASET_REPORT}

    # ---- fine-tuned encoders (mean +- sd over seeds) ----
    L = [r"\begin{tabular}{@{}lccccccccc@{}}", r"\toprule",
         r"\textbf{Model} & \textbf{Params} & \textbf{Prec.} & \textbf{Train time (s)} & \textbf{Epochs} & \textbf{Accuracy} & \textbf{Weighted F1} & \textbf{Macro F1} & \textbf{BPD F1} & \textbf{BPD Sens.} \\",
         r"\midrule"]
    NUM["finetuned"] = {}
    for m in _ft_order(by):
        rs = by[m]
        g = lambda f: [f(r) for r in rs]
        L.append(f"{m} & {rs[0]['params_m']:.0f}M & {rs[0]['precision']} & {_ms(g(lambda r: r['timing']['effective_s']), 0)} & "
                 f"{_ms(g(lambda r: r['epochs_run']), 1)} & {_ms(g(lambda r: r['test']['accuracy']))} & {_ms(g(lambda r: r['test']['weighted_f1']))} & "
                 f"{_ms(g(lambda r: r['test']['macro_f1']))} & {_ms(g(lambda r: r['test']['bpd_f1']))} & {_ms(g(lambda r: r['test']['bpd_sensitivity']))} \\\\")
        NUM["finetuned"][m] = {k: pct_stats(g(f)) for k, f in {
            "train_effective_s": lambda r: r["timing"]["effective_s"], "train_gross_s": lambda r: r["timing"]["gross_s"],
            "train_wasted_s": lambda r: r["timing"]["wasted_s"], "epochs": lambda r: r["epochs_run"],
            "accuracy": lambda r: r["test"]["accuracy"], "macro_f1": lambda r: r["test"]["macro_f1"],
            "weighted_f1": lambda r: r["test"]["weighted_f1"], "bpd_f1": lambda r: r["test"]["bpd_f1"],
            "bpd_sensitivity": lambda r: r["test"]["bpd_sensitivity"], "specificity": lambda r: r["test"]["bin_specificity"],
            "auroc": lambda r: r.get("auroc_bpd", float("nan")), "spec_at_sens85": lambda r: r.get("spec_at_sens85", float("nan")),
            "val_macro_f1": lambda r: r["val"]["macro_f1"]}.items()}
        NUM["finetuned"][m].update({"n_seeds": len(rs), "params_m": rs[0]["params_m"], "precision": rs[0]["precision"],
                                    "collapsed_any": any(r["collapsed"] for r in rs), "n_sessions": [r["timing"]["n_sessions"] for r in rs]})
    L += [r"\bottomrule", r"\end{tabular}"]
    _w("tab_encoder_results", L)

    if R:
        # ---- master table ----
        L = [r"\begin{tabular}{@{}llccccccc@{}}", r"\toprule",
             r"\textbf{Paradigm} & \textbf{Model} & \textbf{Trainable} & \textbf{k} & \textbf{Contam.} & \textbf{Accuracy} & \textbf{Macro F1} & \textbf{BPD Sens.} & \textbf{BPD F1} \\",
             r"\midrule"]
        for m in _ft_order(by):
            rs = by[m]
            L.append(f"Fine-tuned & {m} & {rs[0]['params_m']:.0f}M & -- & N/A & {_ms([r['test']['accuracy'] for r in rs])} & "
                     f"{_ms([r['test']['macro_f1'] for r in rs])} & {_ms([r['test']['bpd_sensitivity'] for r in rs])} & {_ms([r['test']['bpd_f1'] for r in rs])} \\\\")
        L.append(r"\midrule")
        fam_lab = {"pooled_dense": "Pooled dense", "lexical": "Pooled lexical", "vbri": r"\textbf{VBRI (ours)}"}
        for fam in ("pooled_dense", "lexical", "vbri"):
            for nm, v in R["rows"].items():
                if v["family"] != fam:
                    continue
                t = v["test"]
                L.append(f"{fam_lab[fam]} & {tex_escape(nm)} & 0 & {v['k']} & {v.get('contamination_pct', 0.0) if fam != 'vbri' else 0.0:.1f} & "
                         f"{t['accuracy']:.2f} & {t['macro_f1']:.2f} & {t['bpd_sensitivity']:.2f} & {t['bpd_f1']:.2f} \\\\")
            L.append(r"\midrule" if fam != "vbri" else r"\bottomrule")
        L.append(r"\end{tabular}")
        _w("tab_master", L)
        NUM["retrieval_rows"] = {k: {"k": v["k"], "test": {x: v["test"][x] for x in ("accuracy", "macro_f1", "weighted_f1", "bpd_sensitivity", "bpd_f1", "bin_specificity", "bin_ppv", "bin_npv", "inconclusive_recall")},
                                     "contamination_pct": v.get("contamination_pct"), "test_k5_accuracy": v.get("test_k5", {}).get("accuracy")}
                                 for k, v in R["rows"].items()}
        NUM["calibration"] = R["calibration"]["chosen"]
        NUM["cssrs"] = R["cssrs"]
        NUM["reproduction_check"] = R.get("reproduction_check")
        NUM["per_criterion_test"] = R.get("per_criterion_test")
        NUM["ablation_mcnemar"] = R.get("ablation_mcnemar")

        # ---- ablation ----
        L = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r"\textbf{Configuration} & \textbf{Vectors} & \textbf{Isolation} & \textbf{Balance} & \textbf{Accuracy} & \textbf{Macro F1} & \textbf{BPD Sens.} \\", r"\midrule"]
        for nm, v in R["ablation"].items():
            bal = r"\checkmark" if v["train"] == "bal" else ("weights" if v["class_weights"] else "--")
            iso_mark = r"\checkmark" if v["isolated"] else "--"
            L.append(f"{tex_escape(nm)} & {v['n_vectors']:,} & {iso_mark} & {bal} & "
                     f"{v['test']['accuracy']:.2f} & {v['test']['macro_f1']:.2f} & {v['test']['bpd_sensitivity']:.2f} \\\\")
        c = R["calibration"]["chosen"]
        for nm, lab in (("VBRI calibrated", f"\\quad + uncertainty margin ($\\tau$={c['tau']}, $\\delta$={c['delta']}, $\\theta$={c['theta']})"),
                        ("VBRI hybrid", r"\quad + C-SSRS rule (hybrid)")):
            t = R["rows"][nm]["test"]
            L.append(f"{lab} & 4,050 & \\checkmark & \\checkmark & {t['accuracy']:.2f} & {t['macro_f1']:.2f} & {t['bpd_sensitivity']:.2f} \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        _w("tab_ablation", L)
        NUM["ablation"] = {k: {**{x: v["test"][x] for x in ("accuracy", "macro_f1", "bpd_sensitivity")}, "contamination_pct": v["contamination_pct"]}
                           for k, v in R["ablation"].items()}

        # ---- per criterion ----
        L = [r"\begin{tabular}{@{}clcccccccc@{}}", r"\toprule",
             r"\textbf{Q} & \textbf{Criterion} & \textbf{N} & \textbf{Contam.} & \textbf{Pooled acc.} & \textbf{VBRI acc.} & \textbf{$\Delta$} & \textbf{Pooled BPD F1} & \textbf{VBRI BPD F1} & \textbf{$\Delta$} \\", r"\midrule"]
        for q in range(1, 10):
            d = R["per_criterion"][str(q)]
            pa, va_, pf, vf = d["pooled"]["accuracy"], d["vbri"]["accuracy"], d["pooled"]["bpd_f1"], d["vbri"]["bpd_f1"]
            L.append(f"Q{q} & {DSM5[q]} & {d['n']} & {d['pooled_contamination_pct']:.1f} & {pa:.1f} & {va_:.1f} & {va_ - pa:+.1f} & {pf:.1f} & {vf:.1f} & {vf - pf:+.1f} \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        _w("tab_per_criterion", L)
        NUM["per_criterion"] = R["per_criterion"]

    # ---- cost ----
    if by and vt:
        L = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r"\textbf{Model} & \textbf{Gradient updates} & \textbf{Train / index (s)} & \textbf{GPU bs 1 (ms)} & \textbf{GPU bs 32 (ms/ans.)} & \textbf{CPU bs 1 (ms)} & \textbf{Storage (MB)} \\", r"\midrule"]
        NUM["cost"] = {}
        for m in _ft_order(by):
            if m.endswith("-fp32"):
                continue
            rs = by[m]; i = inf.get(m, {})
            wmb = next((r.get("weights_mb") for r in rs if r.get("weights_mb")), None)
            f = lambda k: f"{i[k]:.2f}" if k in i else "--"
            L.append(f"{m} & Yes & {_ms([r['timing']['effective_s'] for r in rs], 0)} & {f('gpu_bs1_median_ms')} & {f('gpu_bs32_ms_per_answer')} & {f('cpu_bs1_median_ms')} & {wmb if wmb else '--'} \\\\")
            NUM["cost"][m] = {"train_effective_s": pct_stats([r["timing"]["effective_s"] for r in rs]), "weights_mb": wmb, **{k: i.get(k) for k in ("gpu_bs1_median_ms", "gpu_bs32_ms_per_answer", "cpu_bs1_median_ms")}}
        g = lambda k: f"{vt[k]:.2f}" if k in vt else "--"
        L.append(r"\midrule")
        L.append(f"\\textbf{{VBRI}} & \\textbf{{No}} & {vt['index_build_s_median']:.1f} & {g('gpu_bs1_total_ms_median')} & {g('gpu_bs32_ms_per_answer')} & {g('cpu_bs1_total_ms_median')} & {vt['index_mb']:.1f} \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        _w("tab_cost", L)
        NUM["vbri_timing"] = vt

    # ---- screening view + statistics ----
    if A and R:
        best = A["best_finetuned"]["model"]
        L = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r"\textbf{Model} & \textbf{Accuracy} & \textbf{Sensitivity} & \textbf{Specificity} & \textbf{PPV} & \textbf{NPV} & \textbf{AUROC} \\", r"\midrule"]
        if best in by:
            rs = by[best]
            L.append(f"{best} (fine-tuned, {len(rs)} seeds) & {_ms([r['test']['bin_accuracy'] for r in rs])} & {_ms([r['test']['bin_sensitivity'] for r in rs])} & "
                     f"{_ms([r['test']['bin_specificity'] for r in rs])} & {_ms([r['test']['bin_ppv'] for r in rs])} & {_ms([r['test']['bin_npv'] for r in rs])} & "
                     f"{_ms([r.get('auroc_bpd', float('nan')) for r in rs], 3)} \\\\")
        for nm in ("VBRI raw", "VBRI calibrated", "VBRI hybrid", "Pooled BGE (answer)"):
            if nm in R["rows"]:
                t = R["rows"][nm]["test"]
                au = A["roc"].get(nm, {}).get("auroc")
                L.append(f"{nm} & {t['bin_accuracy']:.2f} & {t['bin_sensitivity']:.2f} & {t['bin_specificity']:.2f} & {t['bin_ppv']:.2f} & {t['bin_npv']:.2f} & {au:.3f} \\\\"
                         if au is not None else f"{nm} & {t['bin_accuracy']:.2f} & {t['bin_sensitivity']:.2f} & {t['bin_specificity']:.2f} & {t['bin_ppv']:.2f} & {t['bin_npv']:.2f} & -- \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        _w("tab_screening", L)

        L = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"\textbf{Model / difference} & \textbf{Accuracy} & \textbf{Macro F1} & \textbf{BPD Sens.} & \textbf{Specificity} \\", r"\midrule"]
        fmt = lambda v: f"{v['point']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]"
        for nm, d in A["bootstrap"]["ci"].items():
            L.append(f"{tex_escape(nm)} & " + " & ".join(fmt(d[m]) for m in ("accuracy", "macro_f1", "bpd_sensitivity", "bin_specificity")) + r" \\")
        L.append(r"\midrule")
        for nm, d in A["bootstrap"]["differences"].items():
            L.append(f"{tex_escape(nm)} & " + " & ".join(fmt(d[m]) for m in ("accuracy", "macro_f1", "bpd_sensitivity", "bin_specificity")) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}"]
        _w("tab_bootstrap", L)

        L = [r"\begin{tabular}{@{}p{2.2cm}p{5.2cm}p{1.6cm}p{2.2cm}p{4.4cm}@{}}", r"\toprule",
             r"\textbf{Case} & \textbf{Answer (excerpt)} & \textbf{True} & \textbf{Predictions} & \textbf{Retrieved neighbours (VBRI)} \\", r"\midrule"]
        for v in A["vignettes"]:
            nb = "; ".join(f"{n['label']} ({n['sim']:.2f})" for n in v["vbri_neighbours"])
            L.append(f"{tex_escape(v['case'])} (Q{v['question']}) & ``{tex_escape(v['answer'][:150])}\\ldots'' & {v['true']} & "
                     f"Pooled: {v['pooled_pred']} ({v['pooled_contamination_pct']:.0f}\\% contam.)\\newline VBRI: {v['vbri_raw_pred']}\\newline Calibrated: {v['vbri_calibrated_pred']} & {tex_escape(nb)} \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        _w("tab_vignettes", L)
        NUM["analysis"] = {k: v for k, v in A.items() if k not in ("roc", "participant")}
        NUM["roc"] = {k: {"auroc": v["auroc"], "spec_at_sens85": v["spec_at_sens85"], "operating_point": v["operating_point"]} for k, v in A["roc"].items()}
        NUM["participant"] = {k: v for k, v in A["participant"].items() if k not in ("true_counts",)}
        for nm in NUM["participant"]["models"]:
            NUM["participant"]["models"][nm] = {x: NUM["participant"]["models"][nm][x] for x in ("mae", "pearson_r")}
    NUM["data"] = read_json(PAPER_DIR / "data_stats.json", None)
    atomic_write_json(PAPER_DIR / "numbers.json", NUM)
    print("numbers.json written")

if RUN_TABLES:
    make_tables()
