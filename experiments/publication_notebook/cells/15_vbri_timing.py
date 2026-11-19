def vbri_timing(n_cpu=100):
    """Index build time and per-answer latency (text -> prediction) of VBRI, measured fresh (no cache)."""
    k = RETR["rows"]["VBRI raw"]["k"] if RETR else 5
    tr_a, tr_q, tr_y = META["bal"]["ans"], META["bal"]["q"], META["bal"]["y"]
    te_a, te_q = META["test"]["ans"], META["test"]["q"]
    dev = DEVICE
    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()
    t = time.perf_counter()
    bge = SentenceTransformer(ENCODERS[VBRI_ENCODER], device=dev)
    load_s = time.perf_counter() - t
    enc = lambda texts, bs=ENCODE_BS: bge.encode(texts, batch_size=bs, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    enc(tr_a[:ENCODE_BS]); sync()                             # warm-up
    builds = []
    for _ in range(3):
        sync(); t = time.perf_counter()
        E = enc(tr_a)
        stores = {q: (E[tr_q == q], tr_y[tr_q == q]) for q in range(1, 10)}
        sync(); builds.append(time.perf_counter() - t)
    index_mb = sum(m.nbytes + l.nbytes for m, l in stores.values()) / 1e6

    def classify(e, q):
        M, L = stores[q]
        s = M @ e
        top = np.argpartition(-s, k - 1)[:k]
        sc = np.zeros(3)
        for j in top:
            sc[L[j]] += max(float(s[j]), 0.0)
        return int(sc.argmax())

    def per_query(n, label):
        enc_ms, srch_ms = [], []
        for i in range(n):
            sync(); t0 = time.perf_counter()
            e = bge.encode([te_a[i]], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
            sync(); t1 = time.perf_counter()
            classify(e, te_q[i])
            t2 = time.perf_counter()
            enc_ms.append((t1 - t0) * 1000); srch_ms.append((t2 - t1) * 1000)
        tot = sorted(a + b for a, b in zip(enc_ms, srch_ms))
        return {f"{label}_encode_ms_median": statistics.median(enc_ms), f"{label}_search_ms_median": statistics.median(srch_ms),
                f"{label}_total_ms_median": statistics.median(tot), f"{label}_total_ms_mean": statistics.mean(tot),
                f"{label}_total_ms_p95": tot[int(0.95 * (len(tot) - 1))], f"{label}_n": n}

    res = {"encoder": ENCODERS[VBRI_ENCODER], "k": k, "model_load_s": load_s, "index_build_s_runs": builds,
           "index_build_s_median": statistics.median(builds), "index_mb": index_mb, "n_vectors": int(E.shape[0]),
           "encoder_params_m": round(sum(p.numel() for p in bge.parameters()) / 1e6, 1), "gpu": GPU_NAME}
    for i in range(10):
        bge.encode([te_a[i]], show_progress_bar=False)
    if dev == "cuda":
        res.update(per_query(len(te_a), "gpu_bs1"))
        sync(); t = time.perf_counter()                       # batched: whole test set at once
        Eb = enc(te_a, bs=32)
        for i in range(len(te_a)):
            classify(Eb[i], te_q[i])
        sync(); res["gpu_bs32_ms_per_answer"] = (time.perf_counter() - t) * 1000 / len(te_a)
    bge.to("cpu"); dev = "cpu"
    for i in range(5):
        bge.encode([te_a[i]], show_progress_bar=False)
    res.update(per_query(min(n_cpu, len(te_a)), "cpu_bs1"))
    res["cpu_threads"] = torch.get_num_threads()
    atomic_write_json(WORK_DIR / "vbri_timing.json", res)
    print(f"index build {res['index_build_s_median']:.2f}s for {res['n_vectors']} answers (model load {load_s:.1f}s not counted), index {index_mb:.1f} MB")
    for lab in ("gpu_bs1", "cpu_bs1"):
        if f"{lab}_total_ms_median" in res:
            print(f"{lab}: encode {res[lab + '_encode_ms_median']:.2f} ms + search {res[lab + '_search_ms_median']:.3f} ms "
                  f"= {res[lab + '_total_ms_median']:.2f} ms per answer (median)")
    del bge; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res

if RUN_VBRI_TIMING:
    VBRI_TIME = vbri_timing(n_cpu=20 if TEST_MODE else 100)
else:
    VBRI_TIME = read_json(WORK_DIR / "vbri_timing.json", None)
