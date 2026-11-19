def load_results():
    out = []
    for p in sorted(RUNS_DIR.glob("seed*/*/result.json")):
        r = read_json(p, None)
        if r and r.get("code_version") == CODE_VERSION and r.get("status") == "done":
            probs = list(csv.DictReader(open(p.parent / "test_probs.csv")))
            score = [float(x["p_bpd"]) for x in probs]
            yb = [int(x["label"]) == 1 for x in probs]
            r["auroc_bpd"] = auroc(score, yb)
            r["spec_at_sens85"], _ = spec_at_sens(score, yb, 0.85)
            out.append(r)
    return out

def write_tables():
    R = load_results()
    if not R:
        print("no finished runs yet"); return
    # per run
    with open(WORK_DIR / "results_all_runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "seed", "precision", "params_m", "epochs_run", "best_epoch", "collapsed",
                    "train_effective_s", "train_gross_s", "train_wasted_s", "eval_s", "save_s", "n_sessions", "resumed", "hf_last_session_runtime_s",
                    "test_acc", "test_macro_f1", "test_weighted_f1", "test_bpd_f1", "test_bpd_sens", "test_incon_recall",
                    "bin_spec", "bin_ppv", "bin_npv", "bin_acc", "auroc_bpd", "spec_at_sens85", "val_acc", "gpu", "weights_mb"])
        for r in R:
            t, m = r["timing"], r["test"]
            w.writerow([r["model"] + r["variant"], r["seed"], r["precision"], r["params_m"], r["epochs_run"], r["best_epoch"], r["collapsed"],
                        t["effective_s"], t["gross_s"], t["wasted_s"], t["eval_s"], t["save_s"], t["n_sessions"], t["resumed"], r["hf_train_runtime_last_session_s"],
                        round(m["accuracy"], 2), round(m["macro_f1"], 2), round(m["weighted_f1"], 2), round(m["bpd_f1"], 2), round(m["bpd_sensitivity"], 2),
                        round(m["inconclusive_recall"], 2), round(m["bin_specificity"], 2), round(m["bin_ppv"], 2), round(m["bin_npv"], 2),
                        round(m["bin_accuracy"], 2), round(r["auroc_bpd"], 4), round(r["spec_at_sens85"], 2), round(r["val"]["accuracy"], 2), r["gpu"], r.get("weights_mb")])
    # summary over seeds
    names = sorted({r["model"] + r["variant"] for r in R}, key=lambda n: (MODELS + ["DeBERTa-fp32"]).index(n) if n in MODELS + ["DeBERTa-fp32"] else 99)
    lines = []
    hdr = ["model", "seeds", "train time effective (s)", "gross (s)", "wasted (s)", "epochs", "accuracy", "macro F1", "BPD sens", "specificity", "AUROC(BPD)", "spec@sens85"]
    lines.append(hdr)
    for n in names:
        rs = [r for r in R if r["model"] + r["variant"] == n]
        f = lambda g, d=1: "%.*f ± %.*f" % ((d,) + pct_stats([g(r) for r in rs])[:1] + (d,) + pct_stats([g(r) for r in rs])[1:])
        lines.append([n, str(len(rs)), f(lambda r: r["timing"]["effective_s"], 0), f(lambda r: r["timing"]["gross_s"], 0), f(lambda r: r["timing"]["wasted_s"], 0),
                      f(lambda r: r["epochs_run"], 1), f(lambda r: r["test"]["accuracy"], 2), f(lambda r: r["test"]["macro_f1"], 2),
                      f(lambda r: r["test"]["bpd_sensitivity"], 2), f(lambda r: r["test"]["bin_specificity"], 2),
                      f(lambda r: r["auroc_bpd"] * 100, 1), f(lambda r: r["spec_at_sens85"], 1)])
    with open(WORK_DIR / "summary_mean_sd.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(lines)
    widths = [max(len(str(l[i])) for l in lines) for i in range(len(hdr))]
    print("\nSUMMARY (mean ± sd over seeds; time in seconds; GPU:", GPU_NAME, ")\n")
    for l in lines:
        print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(l)))
    return R

RESULTS = write_tables()