def status_table():
    rows = []
    for seed in SEEDS:
        for m in MODELS + ["DeBERTa-fp32"]:
            d = RUNS_DIR / f"seed{seed}" / m
            if not d.exists():
                continue
            r = read_json(d / "result.json", None)
            if r and r.get("code_version") == CODE_VERSION:
                rows.append((seed, m, "done", f"{r['timing']['effective_s']:.0f}s", f"{r['test']['accuracy']:.2f}"))
            else:
                led = read_json(d / "ledger.json", {})
                rows.append((seed, m, "partial (will resume)", f"{sum(s.get('active_s',0) for s in led.get('sessions',[])):.0f}s so far", "-"))
    return rows

def run_all():
    failed = []
    for seed in SEEDS:
        for m in MODELS:
            try:
                # DeBERTa-v3 cannot use the Trainer's fp16 scaler (FP16 gradient unscale error), so it trains in fp32
                res = run_training(m, seed, precision="fp32" if (m == "DeBERTa" and PRECISION == "fp16") else None)
                if m == "DeBERTa" and res.get("collapsed") and (PRECISION == "fp16" or (PROFILE == "windows" and PRECISION == "bf16")):
                    print(f"[seed{seed}/DeBERTa] collapsed under {PRECISION} -> one documented retry in fp32")
                    run_training(m, seed, variant="-fp32", precision="fp32")
            except Exception as e:
                print(f"\n!!! seed{seed}/{m} failed: {type(e).__name__}: {e}")
                traceback.print_exc()
                failed.append((seed, m, str(e)))
    print("\nRun status:")
    for r in status_table():
        print("  ", *r)
    if failed:
        print("\nFAILED runs (run this cell again: they resume from their last checkpoint):", failed)
    return failed

if RUN_MAIN_TRAINING:
    _failed = run_all()