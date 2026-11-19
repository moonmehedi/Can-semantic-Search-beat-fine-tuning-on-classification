def vbri_benchmark():
    from sentence_transformers import SentenceTransformer
    ans = lambda ds: [str(a).strip() for a in ds["answer"]]
    tr_a, tr_q, tr_y = ans(RAW["train"]), [int(v) for v in RAW["train"]["question_id"]], [int(v) for v in RAW["train"]["label"]]
    va_a, va_q, va_y = ans(RAW["validation"]), [int(v) for v in RAW["validation"]["question_id"]], [int(v) for v in RAW["validation"]["label"]]
    te_a = ans(RAW["test"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = time.perf_counter(); bge = SentenceTransformer("BAAI/bge-base-en-v1.5", device=dev); load_s = time.perf_counter() - t

    def sync():
        if dev == "cuda": torch.cuda.synchronize()
    def encode(texts, bs=64):
        return bge.encode(texts, batch_size=bs, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    encode(tr_a[:64]); sync()                                  # warm-up
    builds = []
    for _ in range(3):
        sync(); t = time.perf_counter(); E = encode(tr_a); sync(); builds.append(time.perf_counter() - t)
    stores = {}
    for q in sorted(set(tr_q)):
        idx = [i for i, qq in enumerate(tr_q) if qq == q]
        stores[q] = (E[idx], np.array([tr_y[i] for i in idx]))
    index_mb = sum(s[0].nbytes + s[1].nbytes for s in stores.values()) / 1e6

    def vote(emb, q, k):
        M, L = stores[q]
        sims = M @ emb
        top = np.argpartition(-sims, k)[:k]
        sc = np.zeros(3)
        for j in top:
            sc[L[j]] += max(float(sims[j]), 0.0)
        return sc

    # --- k chosen on the VALIDATION set ---
    Ev = encode(va_a)
    val_f1 = {}
    for k in (1, 3, 5, 7, 9):
        pred = [int(np.argmax(vote(Ev[i], va_q[i], k))) for i in range(len(va_a))]
        val_f1[k] = summarize(va_y, pred)["macro_f1"]
    best_k = max(val_f1, key=lambda k: (val_f1[k], -k))
    print("validation macro-F1 by k:", {k: round(v, 2) for k, v in val_f1.items()}, "-> chosen k =", best_k)

    Et = encode(te_a)
    res = {"dataset": DATASET_REPORT["path"], "model_load_s": load_s, "index_build_s_runs": builds,
           "index_build_s_median": statistics.median(builds), "index_mb": index_mb, "n_vectors": int(E.shape[0]),
           "val_macro_f1_by_k": val_f1, "chosen_k": best_k}
    for k in sorted({best_k, 5}):
        sc = [vote(Et[i], TEST_QIDS[i], k) for i in range(len(te_a))]
        pred = [int(np.argmax(s)) for s in sc]
        res[f"test_k{k}"] = summarize(TEST_Y, pred)
        tot = [s.sum() or 1.0 for s in sc]
        pb = [float(s[1] / z) for s, z in zip(sc, tot)]
        yb = [y == 1 for y in TEST_Y]
        res[f"test_k{k}"]["auroc_bpd"] = auroc(pb, yb)
        res[f"test_k{k}"]["spec_at_sens85"] = spec_at_sens(pb, yb, 0.85)[0]
        with open(WORK_DIR / f"vbri_test_scores_k{k}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["id", "question_id", "label", "p_normal", "p_bpd", "p_inconclusive"])
            for pid, q, y, s, z in zip(TEST_IDS, TEST_QIDS, TEST_Y, sc, tot):
                w.writerow([pid, q, y, *[f"{v / z:.6f}" for v in s]])

    # --- per-query latency: text -> prediction (encode + search), one answer at a time ---
    def per_query(n, device_label):
        enc_ms, srch_ms = [], []
        for i in range(n):
            sync(); t0 = time.perf_counter()
            e = bge.encode([te_a[i]], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
            sync(); t1 = time.perf_counter()
            vote(e, TEST_QIDS[i], best_k)
            t2 = time.perf_counter()
            enc_ms.append((t1 - t0) * 1000); srch_ms.append((t2 - t1) * 1000)
        tot = sorted(a + b for a, b in zip(enc_ms, srch_ms))
        return {f"{device_label}_encode_ms_median": statistics.median(enc_ms), f"{device_label}_search_ms_median": statistics.median(srch_ms),
                f"{device_label}_total_ms_median": statistics.median(tot), f"{device_label}_total_ms_mean": statistics.mean(tot),
                f"{device_label}_total_ms_p95": tot[int(0.95 * (len(tot) - 1))], f"{device_label}_n": n}
    for i in range(10):
        bge.encode([te_a[i]], show_progress_bar=False)
    if dev == "cuda":
        res.update(per_query(len(te_a), "gpu_bs1"))
    bge.to("cpu"); dev = "cpu"
    res.update(per_query(100, "cpu_bs1"))
    res["cpu_threads"] = torch.get_num_threads()
    atomic_write_json(WORK_DIR / "vbri_benchmark.json", res)

    k = best_k; m = res[f"test_k{k}"]
    print(f"\nVBRI (k={k}, chosen on validation)  accuracy {m['accuracy']:.2f}  macro-F1 {m['macro_f1']:.2f}  BPD sens {m['bpd_sensitivity']:.2f}  spec {m['bin_specificity']:.2f}  AUROC {m['auroc_bpd']:.3f}")
    m5 = res["test_k5"]; print(f"VBRI (k=5)                           accuracy {m5['accuracy']:.2f}  macro-F1 {m5['macro_f1']:.2f}  BPD sens {m5['bpd_sensitivity']:.2f}")
    print(f"index build {res['index_build_s_median']:.2f}s for {res['n_vectors']} answers (model load {load_s:.1f}s, not counted) | index {index_mb:.1f} MB")
    for lab in ("gpu_bs1", "cpu_bs1"):
        if f"{lab}_total_ms_median" in res:
            print(f"{lab}: encode {res[lab + '_encode_ms_median']:.2f} ms + search {res[lab + '_search_ms_median']:.3f} ms = {res[lab + '_total_ms_median']:.2f} ms per answer (median)")
    return res

if RUN_VBRI_BENCH:
    VBRI = vbri_benchmark()