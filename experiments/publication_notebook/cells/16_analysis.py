NEG_RE = re.compile(r"\b(no|not|never|nothing|none|nobody|don't|dont|didn't|didnt|haven't|havent|hasn't|wasn't|isn't|can't|cannot|n't)\b", re.I)

def _read_probs(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    assert [int(r["id"]) for r in rows] == TEST_IDS, f"row order of {path} differs from the test set"
    P = np.array([[float(r["p_normal"]), float(r["p_bpd"]), float(r["p_inconclusive"])] for r in rows])
    pred = np.array([int(r["pred"]) for r in rows]) if "pred" in rows[0] else P.argmax(1)
    return P, pred

def roc_points(scores, ybin):
    s, y = np.asarray(scores, float), np.asarray(ybin, bool)
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    tps, fps = np.cumsum(y), np.cumsum(~y)
    keep = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tpr = np.r_[0.0, tps[keep] / y.sum()]
    fpr = np.r_[0.0, fps[keep] / (~y).sum()]
    return fpr.tolist(), tpr.tolist()

def ft_by_model():
    by = defaultdict(list)
    for r in load_results():
        by[r["model"] + r["variant"]].append(r)
    return by

def run_analysis():
    A = {}
    by = ft_by_model()
    yt = TEST_Y
    ybin = [y == 1 for y in yt]
    s0 = SEEDS[0]
    # best fine-tuned model = highest mean VALIDATION macro-F1 over seeds (no test-set selection)
    mean_val = {m: float(np.mean([r["val"]["macro_f1"] for r in rs])) for m, rs in by.items() if not any(r["collapsed"] for r in rs)}
    BEST = max(mean_val, key=mean_val.get) if mean_val else None
    A["best_finetuned"] = {"model": BEST, "selection": "highest mean validation macro-F1", "mean_val_macro_f1": mean_val}
    print("Best fine-tuned model (chosen on validation):", BEST)

    probs, preds = {}, {}
    for m in by:
        p = RUNS_DIR / f"seed{s0}" / m / "test_probs.csv"
        if p.exists():
            probs[m], preds[m] = _read_probs(p)
    for nm in ("Pooled BGE (answer)", "Pooled BM25", "VBRI raw", "VBRI calibrated", "VBRI hybrid"):
        p = PRED_DIR / f"{_slug(nm)}.csv"
        if p.exists():
            probs[nm], preds[nm] = _read_probs(p)
    if "Pooled BGE (answer)" not in preds and TEST_MODE:
        p = PRED_DIR / f"{_slug('Pooled ' + VBRI_ENCODER + ' (answer)')}.csv"
        probs["Pooled BGE (answer)"], preds["Pooled BGE (answer)"] = _read_probs(p)

    # ---- bootstrap confidence intervals (paired, over test answers) ----
    names = [n for n in (BEST, "VBRI raw", "VBRI hybrid", "Pooled BGE (answer)") if n in preds]
    diffs = [(BEST, v) for v in ("VBRI raw", "VBRI hybrid") if BEST in preds and v in preds]
    ci, dci = bootstrap_ci(yt, {n: preds[n] for n in names}, ["accuracy", "macro_f1", "bpd_sensitivity", "bin_specificity"], diffs=diffs)
    A["bootstrap"] = {"n_boot": N_BOOT, "ci": ci, "differences": dci, "seed_of_finetuned_run": s0}

    # ---- McNemar ----
    cor = {n: [int(p == y) for p, y in zip(preds[n].tolist(), yt)] for n in preds}
    pairs = [(BEST, "VBRI raw"), (BEST, "VBRI hybrid"), ("VBRI raw", "VBRI calibrated"), ("VBRI raw", "VBRI hybrid"), ("Pooled BGE (answer)", "VBRI raw")]
    A["mcnemar"] = {f"{a} vs {b}": mcnemar(cor[a], cor[b]) for a, b in pairs if a in cor and b in cor}
    bpd_ix = [i for i, y in enumerate(yt) if y == 1]
    if BEST in cor:
        A["mcnemar_bpd_only"] = {f"{BEST} vs VBRI raw": mcnemar([cor[BEST][i] for i in bpd_ix], [cor["VBRI raw"][i] for i in bpd_ix])}

    # ---- ROC (BPD vs rest), AUROC, specificity at 85% sensitivity ----
    A["roc"] = {}
    for n in [x for x in (BEST, "VBRI raw", "VBRI calibrated", "Pooled BGE (answer)") if x in probs]:
        sc = probs[n][:, 1].tolist()
        fpr, tpr = roc_points(sc, ybin)
        A["roc"][n] = {"auroc": auroc(sc, ybin), "spec_at_sens85": spec_at_sens(sc, ybin, 0.85)[0], "fpr": fpr, "tpr": tpr,
                       "operating_point": {"sens": summarize(yt, preds[n].tolist())["bin_sensitivity"],
                                           "spec": summarize(yt, preds[n].tolist())["bin_specificity"]}}

    # ---- participant level ----
    pids = sorted(set(TEST_PIDS))
    def counts(pred):
        return [sum(1 for i, p in enumerate(TEST_PIDS) if p == pid and int(pred[i]) == 1) for pid in pids]
    true_c = counts(yt)
    A["participant"] = {"n_participants": len(pids), "answers_per_participant_max": max(Counter(TEST_PIDS).values()),
                        "true_bpd_count_max": max(true_c), "models": {}}
    for n in [x for x in ("Pooled BGE (answer)", "VBRI raw", "VBRI hybrid", BEST) if x in preds]:
        pc = counts(preds[n])
        A["participant"]["models"][n] = {"mae": float(np.mean(np.abs(np.array(pc) - np.array(true_c)))),
                                         "pearson_r": float(np.corrcoef(true_c, pc)[0, 1]), "counts": pc}
    A["participant"]["true_counts"] = true_c

    # ---- error analysis of VBRI raw ----
    pr = preds["VBRI raw"]
    err = [i for i in range(len(yt)) if pr[i] != yt[i]]
    n2b = [i for i in err if yt[i] == 0 and pr[i] == 1]
    A["errors_vbri_raw"] = {"n_errors": len(err), "normal_to_bpd": len(n2b),
                            "normal_to_bpd_with_negation": sum(1 for i in n2b if NEG_RE.search(TEST_ANS_RAW[i])),
                            "by_true_class": {CLASSES[c]: sum(1 for i in err if yt[i] == c) for c in (0, 1, 2)}}

    # ---- vignettes with the retrieved neighbours ----
    VK = RETR["rows"]["VBRI raw"]["k"]
    iso = dense_run(VBRI_ENCODER, "ans", "bal", "test", VK, isolated=True)
    poo = dense_run(VBRI_ENCODER, "ans", "bal", "test", VK, isolated=False)
    cal = preds.get("VBRI calibrated", pr)
    def neigh(res, i):
        return [{"question": int(META["bal"]["q"][j]), "label": CLASSES[int(META["bal"]["y"][j])], "sim": round(float(s), 3),
                 "answer": META["bal"]["ans"][j][:160]} for j, s in zip(res["idx"][i], res["sims"][i])]
    picks = []
    c1 = [i for i in range(len(yt)) if poo["pred"][i] != yt[i] and iso["pred"][i] == yt[i]]
    if c1:
        picks.append(("Pooled index retrieves the wrong criterion", max(c1, key=lambda i: poo["contam"][i])))
    c2 = [i for i in range(len(yt)) if yt[i] == 2 and pr[i] != 2 and cal[i] == 2]
    if c2:
        picks.append(("Uncertainty margin flags an ambiguous answer", c2[0]))
    c3 = [i for i in range(len(yt)) if yt[i] == 1 and pr[i] == 1]
    if c3:
        picks.append(("Clear BPD-indicative answer", max(c3, key=lambda i: (iso["scores"][i][1], TEST_QIDS[i] == 1))))
    c4 = [i for i in n2b if NEG_RE.search(TEST_ANS_RAW[i])]
    if c4:
        picks.append(("Error: negated statement read as BPD", c4[0]))
    A["vignettes"] = [{"case": case, "id": TEST_IDS[i], "question": TEST_QIDS[i], "criterion": DSM5[TEST_QIDS[i]],
                       "answer": TEST_ANS_RAW[i][:240], "true": CLASSES[yt[i]], "pooled_pred": CLASSES[int(poo["pred"][i])],
                       "pooled_contamination_pct": float(poo["contam"][i]), "vbri_raw_pred": CLASSES[int(pr[i])],
                       "vbri_calibrated_pred": CLASSES[int(cal[i])], "vbri_neighbours": neigh(iso, i),
                       "pooled_neighbours": neigh(poo, i)} for case, i in picks]
    atomic_write_json(PAPER_DIR / "analysis.json", A)

    # ---- printout ----
    print("\nBootstrap 95% CI (test set):")
    for n, d in ci.items():
        print(f"  {n:24s} " + "  ".join(f"{m} {v['point']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]" for m, v in d.items()))
    for n, d in dci.items():
        print(f"  DIFF {n:30s} " + "  ".join(f"{m} {v['point']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]" for m, v in d.items()))
    print("\nMcNemar:", {k: (v["b"], v["c"], round(v["p_exact"], 4)) for k, v in A["mcnemar"].items()})
    print("ROC:", {k: (round(v["auroc"], 3), round(v["spec_at_sens85"], 1)) for k, v in A["roc"].items()})
    print("Participants:", {k: (round(v["mae"], 3), round(v["pearson_r"], 3)) for k, v in A["participant"]["models"].items()})
    print("VBRI raw errors:", A["errors_vbri_raw"])
    return A

if RUN_ANALYSIS:
    ANALYSIS = run_analysis()
else:
    ANALYSIS = read_json(PAPER_DIR / "analysis.json", None)
