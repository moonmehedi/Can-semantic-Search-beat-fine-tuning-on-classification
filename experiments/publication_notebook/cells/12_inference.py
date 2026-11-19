def bench_model(model_name, run_dir, n_cpu=100):
    mdir = run_dir / "final_model"
    tok = AutoTokenizer.from_pretrained(str(mdir))
    model = AutoModelForSequenceClassification.from_pretrained(str(mdir)).eval()
    Q, A = TEST_Q_NORM, TEST_A_NORM
    def fwd(q, a, dev):
        enc = tok(q, a, truncation=True, padding=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            return model(**enc).logits
    def stats(ms):
        ms = sorted(ms); return {"mean_ms": statistics.mean(ms), "median_ms": statistics.median(ms), "p95_ms": ms[int(0.95 * (len(ms) - 1))]}
    res = {"model": model_name, "weights_mb": sum(f.stat().st_size for f in mdir.glob("*") if f.suffix in (".safetensors", ".bin")) / 1e6}
    if torch.cuda.is_available():
        model.to("cuda")
        for i in range(20):
            fwd([Q[i]], [A[i]], "cuda")
        ms = []
        for q, a in zip(Q, A):
            torch.cuda.synchronize(); t = time.perf_counter()
            fwd([q], [a], "cuda")
            torch.cuda.synchronize(); ms.append((time.perf_counter() - t) * 1000)
        res.update({f"gpu_bs1_{k}": v for k, v in stats(ms).items()})
        fwd(Q[:32], A[:32], "cuda"); torch.cuda.synchronize()
        t = time.perf_counter()
        for i in range(0, len(Q), 32):
            fwd(Q[i:i + 32], A[i:i + 32], "cuda")
        torch.cuda.synchronize()
        res["gpu_bs32_ms_per_answer"] = (time.perf_counter() - t) * 1000 / len(Q)
    model.to("cpu")
    for i in range(5):
        fwd([Q[i]], [A[i]], "cpu")
    ms = []
    for q, a in zip(Q[:n_cpu], A[:n_cpu]):
        t = time.perf_counter(); fwd([q], [a], "cpu"); ms.append((time.perf_counter() - t) * 1000)
    res.update({f"cpu_bs1_{k}": v for k, v in stats(ms).items()})
    res["cpu_threads"] = torch.get_num_threads()
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res

if RUN_INFERENCE_BENCH:
    bench_seed = KEEP_WEIGHTS_SEEDS[0]
    out_path = WORK_DIR / "inference_benchmark.csv"
    rows = []
    for m in MODELS:
        d = RUNS_DIR / f"seed{bench_seed}" / m
        if (d / "final_model").exists():
            print("benchmarking", m, "...")
            rows.append(bench_model(m, d))
    if rows:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "model", k))
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows:
                w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
        print("\n%-12s %10s %14s %16s %14s" % ("model", "weights MB", "GPU bs1 (ms)", "GPU bs32 (ms/ans)", "CPU bs1 (ms)"))
        for r in rows:
            print("%-12s %10.1f %14.2f %16.2f %14.2f" % (r["model"], r["weights_mb"], r.get("gpu_bs1_median_ms", float("nan")),
                                                       r.get("gpu_bs32_ms_per_answer", float("nan")), r["cpu_bs1_median_ms"]))
    else:
        print("No saved weights found for seed", bench_seed)