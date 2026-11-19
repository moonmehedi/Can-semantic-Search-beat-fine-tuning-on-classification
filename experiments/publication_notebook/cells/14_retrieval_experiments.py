PRED_DIR = RETR_DIR / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

def _slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")

def save_preds(name, res, extra_pred=None):
    """Per-answer test predictions + normalised class scores (used later for ROC, statistics and vignettes)."""
    sc = np.asarray(res["scores"], dtype=float)
    tot = sc.sum(1, keepdims=True)
    tot[tot == 0] = 1.0
    P = sc / tot
    pred = extra_pred if extra_pred is not None else res["pred"]
    with open(PRED_DIR / f"{_slug(name)}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "question_id", "participant_id", "label", "pred", "p_normal", "p_bpd", "p_inconclusive", "contamination_pct"])
        for i in range(len(TEST_Y)):
            w.writerow([TEST_IDS[i], TEST_QIDS[i], TEST_PIDS[i], TEST_Y[i], int(pred[i]), *[f"{v:.6f}" for v in P[i]],
                        f"{float(res['contam'][i]):.1f}" if "contam" in res else ""])

def run_retrieval_experiments():
    R = {"settings": {"k_grid": K_GRID, "vbri_encoder": ENCODERS[VBRI_ENCODER], "selection": "validation macro-F1"},
         "rows": {}, "ablation": {}, "k_sweep": {}, "per_criterion": {}, "calibration": {}, "cssrs": {}}
    yv, yt = META["val"]["y"].tolist(), META["test"]["y"].tolist()

    def add_row(name, family, enc, run_fn, params_m):
        k, f1 = choose_k(run_fn)
        te, va = run_fn("test", k), run_fn("val", k)
        te5 = run_fn("test", 5)
        R["rows"][name] = {"family": family, "encoder": enc, "params_m": params_m, "k": k, "val_macro_f1_by_k": {str(kk): vv for kk, vv in f1.items()},
                           "test": summarize(yt, te["pred"].tolist()), "val": summarize(yv, va["pred"].tolist()),
                           "test_k5": summarize(yt, te5["pred"].tolist()),
                           "contamination_pct": float(np.mean(te["contam"])), "contamination_pct_k5": float(np.mean(te5["contam"]))}
        save_preds(name, te)
        m = R["rows"][name]["test"]
        print(f"  {name:28s} k={k:<2d} acc {m['accuracy']:6.2f}  macroF1 {m['macro_f1']:6.2f}  BPD sens {m['bpd_sensitivity']:6.2f}  contam {R['rows'][name]['contamination_pct']:5.1f}%")
        return te, k

    # ---- 1. pooled dense baselines (one index for all questions, balanced training set) ----
    print("Pooled dense retrieval baselines (balanced training set, k chosen on validation):")
    for enc in ENCODERS:
        pm = encoder_params_m(enc)
        for fmt, lab in (("ans", "answer"), ("qa", "Q+A")):
            add_row(f"Pooled {enc} ({lab})", "pooled_dense", enc,
                    lambda split, k, e=enc, f=fmt: dense_run(e, f, "bal", split, k, isolated=False), pm)
    # ---- 2. pooled lexical baselines ----
    print("Pooled lexical baselines:")
    for meth in ("BM25", "TF-IDF"):
        add_row(f"Pooled {meth}", "lexical", None, lambda split, k, m=meth: lexical_run(m, "bal", split, k), 0.0)
    # ---- 3. VBRI raw: nine isolated, balanced databases ----
    print("VBRI (nine isolated databases):")
    vb_fn = lambda split, k: dense_run(VBRI_ENCODER, "ans", "bal", split, k, isolated=True)
    vb_te, VK = add_row("VBRI raw", "vbri", VBRI_ENCODER, vb_fn, encoder_params_m(VBRI_ENCODER))
    vb_va = vb_fn("val", VK)

    # ---- 4. calibration: temperature, margin and BPD threshold chosen on VALIDATION ----
    grid = []
    for tau in (0.05, 0.10, 0.15, 0.25, 0.50):
        Pv = temp_probs(vb_va["idx"], vb_va["sims"], META["bal"]["y"], tau)
        for delta in (0.00, 0.05, 0.08, 0.12, 0.15):
            for theta in (0.35, 0.40, 0.45, 0.50):
                mv = summarize(yv, calibrated_pred(Pv, delta, theta).tolist())
                grid.append({"tau": tau, "delta": delta, "theta": theta, "val_macro_f1": mv["macro_f1"], "val_bpd_sens": mv["bpd_sensitivity"]})
    best = max(grid, key=lambda g: (round(g["val_macro_f1"], 6), round(g["val_bpd_sens"], 6), -g["delta"], -g["tau"]))
    Pt = temp_probs(vb_te["idx"], vb_te["sims"], META["bal"]["y"], best["tau"])
    cal_pred = calibrated_pred(Pt, best["delta"], best["theta"])
    R["calibration"] = {"chosen": best, "grid": grid}
    R["rows"]["VBRI calibrated"] = {**{k: v for k, v in R["rows"]["VBRI raw"].items() if k not in ("test", "val", "test_k5")},
                                    "family": "vbri", "test": summarize(yt, cal_pred.tolist()),
                                    "calibration": {k: best[k] for k in ("tau", "delta", "theta")}}
    save_preds("VBRI calibrated", {"scores": Pt, "pred": cal_pred, "contam": vb_te["contam"]})
    m = R["rows"]["VBRI calibrated"]["test"]
    print(f"  {'VBRI calibrated':28s} tau={best['tau']} delta={best['delta']} theta={best['theta']}  acc {m['accuracy']:6.2f}  macroF1 {m['macro_f1']:6.2f}  BPD sens {m['bpd_sensitivity']:6.2f}")

    # ---- 5. C-SSRS safety rule (Q5 only) ----
    for base_name, base_pred, base_scores in (("VBRI raw", vb_te["pred"], vb_te["scores"]), ("VBRI calibrated", cal_pred, Pt)):
        hp, fired = apply_cssrs(base_pred, "test")
        _, fired_val = apply_cssrs(vb_va["pred"] if base_name == "VBRI raw" else
                                   calibrated_pred(temp_probs(vb_va["idx"], vb_va["sims"], META["bal"]["y"], best["tau"]), best["delta"], best["theta"]), "val")
        nm = "VBRI raw + C-SSRS" if base_name == "VBRI raw" else "VBRI hybrid"
        R["rows"][nm] = {**{k: v for k, v in R["rows"][base_name].items() if k not in ("test", "val", "test_k5")},
                         "family": "vbri", "test": summarize(yt, hp.tolist())}
        R["cssrs"][nm] = {"fired_test": [{"id": TEST_IDS[i], "true": TEST_Y[i], "answer": TEST_ANS_RAW[i][:200]} for i in fired],
                          "fired_val": len(fired_val)}
        save_preds(nm, {"scores": base_scores, "pred": hp, "contam": vb_te["contam"]})
        m = R["rows"][nm]["test"]
        print(f"  {nm:28s} rule fired on {len(fired)} test answer(s) (true labels {[TEST_Y[i] for i in fired]})  acc {m['accuracy']:6.2f}  BPD sens {m['bpd_sensitivity']:6.2f}")

    # ---- 6. ablation: isolation x class balance (same encoder and k as VBRI) ----
    print(f"Ablation (encoder {VBRI_ENCODER}, answer text, k={VK}):")
    abl = [("Pooled, unbalanced", "unb", False, False), ("Isolated, unbalanced", "unb", True, False),
           ("Pooled, unbalanced + class weights", "unb", False, True), ("Isolated, unbalanced + class weights", "unb", True, True),
           ("Pooled, balanced", "bal", False, False), ("Isolated, balanced (VBRI raw)", "bal", True, False)]
    abl_preds = {}
    for nm, tr, iso, cw in abl:
        if tr == "unb" and not AUG_OK:
            continue
        r = dense_run(VBRI_ENCODER, "ans", tr, "test", VK, isolated=iso, class_weighted=cw)
        abl_preds[nm] = r["pred"]
        R["ablation"][nm] = {"train": tr, "isolated": iso, "class_weights": cw, "n_vectors": int(len(META[tr]["y"])),
                             "test": summarize(yt, r["pred"].tolist()), "contamination_pct": float(np.mean(r["contam"]))}
        m = R["ablation"][nm]["test"]
        print(f"  {nm:40s} acc {m['accuracy']:6.2f}  macroF1 {m['macro_f1']:6.2f}  BPD sens {m['bpd_sensitivity']:6.2f}")
    cor = {k: [int(p == y) for p, y in zip(v.tolist(), yt)] for k, v in abl_preds.items()}
    R["ablation_mcnemar"] = {}
    for a, b in (("Pooled, balanced", "Isolated, balanced (VBRI raw)"), ("Isolated, unbalanced", "Isolated, balanced (VBRI raw)"),
                 ("Pooled, unbalanced", "Pooled, balanced")):
        if a in cor and b in cor:
            R["ablation_mcnemar"][f"{a} vs {b}"] = mcnemar(cor[a], cor[b])

    # ---- 7. k-sweep (validation and test) ----
    for nm, iso in (("Pooled, balanced", False), ("Isolated, balanced (VBRI)", True)):
        R["k_sweep"][nm] = {}
        for k in K_GRID:
            rv = dense_run(VBRI_ENCODER, "ans", "bal", "val", k, isolated=iso)
            rt = dense_run(VBRI_ENCODER, "ans", "bal", "test", k, isolated=iso)
            R["k_sweep"][nm][str(k)] = {"val_macro_f1": summarize(yv, rv["pred"].tolist())["macro_f1"],
                                   "test_macro_f1": summarize(yt, rt["pred"].tolist())["macro_f1"],
                                   "test_accuracy": summarize(yt, rt["pred"].tolist())["accuracy"],
                                   "test_contamination_pct": float(np.mean(rt["contam"]))}

    # ---- 8. per-criterion breakdown (pooled balanced vs VBRI raw) ----
    pb = dense_run(VBRI_ENCODER, "ans", "bal", "test", VK, isolated=False)
    q = META["test"]["q"]
    for qq in range(1, 10):
        msk = q == qq
        yy = [y for y, mm in zip(yt, msk) if mm]
        R["per_criterion"][str(qq)] = {"n": int(msk.sum()), "criterion": DSM5[qq],
                                  "pooled": summarize(yy, pb["pred"][msk].tolist()), "vbri": summarize(yy, vb_te["pred"][msk].tolist()),
                                  "pooled_contamination_pct": float(np.mean(pb["contam"][msk]))}
    try:
        from scipy import stats as _st
        da = [R["per_criterion"][str(qq)]["vbri"]["accuracy"] - R["per_criterion"][str(qq)]["pooled"]["accuracy"] for qq in range(1, 10)]
        R["per_criterion_test"] = {"mean_acc_gain": float(np.mean(da)),
                                   "paired_t_p": float(_st.ttest_1samp(da, 0.0).pvalue),
                                   "wilcoxon_p": float(_st.wilcoxon([d for d in da if d != 0]).pvalue) if any(d != 0 for d in da) else 1.0}
    except Exception as e:
        R["per_criterion_test"] = {"error": str(e)}

    # ---- 9. reproduction check against the earlier VBRI notebook (k = 5, same data) ----
    if not TEST_MODE:
        r5 = R["rows"]["VBRI raw"]["test_k5"]
        p5 = R["rows"]["Pooled BGE (answer)"]["test_k5"]
        R["reproduction_check"] = {"vbri_raw_k5_accuracy": r5["accuracy"], "expected": 72.67,
                                   "pooled_bge_k5_accuracy": p5["accuracy"], "expected_pooled": 67.87}
        ok = abs(r5["accuracy"] - 72.67) < 1.0 and abs(p5["accuracy"] - 67.87) < 1.0
        print(("REPRODUCTION OK" if ok else "!!! REPRODUCTION DIFFERS") +
              f": VBRI raw k=5 acc {r5['accuracy']:.2f} (earlier 72.67), pooled BGE k=5 acc {p5['accuracy']:.2f} (earlier 67.87)")
    atomic_write_json(RETR_DIR / "retrieval_results.json", R)
    return R

if RUN_RETRIEVAL:
    RETR = run_retrieval_experiments()
else:
    RETR = read_json(RETR_DIR / "retrieval_results.json", None)
