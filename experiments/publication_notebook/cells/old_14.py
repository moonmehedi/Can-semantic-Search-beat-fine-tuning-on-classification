SMOKE_FLAG = WORK_DIR / "smoke_passed.json"

def run_smoke_test():
    root = WORK_DIR / "smoke"
    shutil.rmtree(root, ignore_errors=True)
    sub = {"train": 800, "val": 120}
    steps_per_epoch = sub["train"] // (MODEL_SPECS["DistilBERT"]["bs"] * MODEL_SPECS["DistilBERT"]["accum"])
    crash_at = int(steps_per_epoch * 1.5)
    kw = dict(model_name="DistilBERT", seed=1, run_root=root, epochs=4, patience=99, subset=sub, heartbeat_s=1.0, keep_weights=False)

    print(">>> Smoke test A: first session, will be killed at step", crash_at)
    try:
        run_training(crash_at_step=crash_at, simulate_kill=True, **kw)
        raise AssertionError("the simulated crash did not happen")
    except RuntimeError as e:
        print("    crashed as planned:", e)
    print(">>> Smoke test B: resume")
    B = run_training(crash_at_step=crash_at, simulate_kill=True, **kw)
    print(">>> Smoke test C: uninterrupted reference")
    C = run_training(**{**kw, "seed": 2})

    tb, tc = B["timing"], C["timing"]
    checks = {
        "B was resumed in a 2nd session": tb["resumed"] and tb["n_sessions"] == 2,
        "B has wasted time (lost partial epoch)": tb["wasted_s"] > 0,
        "B effective < B gross": tb["effective_s"] < tb["gross_s"],
        "B effective close to uninterrupted C (+-40% or 8 s)": abs(tb["effective_s"] - tc["effective_s"]) <= max(0.4 * tc["effective_s"], 8.0),
        "B first session was marked interrupted": B["timing"]["session_status"][0] == "interrupted",
    }
    print("\n   effective  gross  wasted | B(resumed) %.1f %.1f %.1f | C(clean) %.1f %.1f %.1f" % (
        tb["effective_s"], tb["gross_s"], tb["wasted_s"], tc["effective_s"], tc["gross_s"], tc["wasted_s"]))
    for k, v in checks.items():
        print(("   PASS  " if v else "   FAIL  ") + k)
    if not all(checks.values()):
        raise AssertionError("Smoke test failed - do not start the long run. Check the ledger in " + str(root))
    atomic_write_json(SMOKE_FLAG, {"code_version": CODE_VERSION, "passed_utc": utc_now(), "checks": checks,
                                   "B": tb, "C": tc})
    shutil.rmtree(root, ignore_errors=True)
    print("\nSMOKE TEST PASSED")

_flag = read_json(SMOKE_FLAG, None)
if RUN_SMOKE_TEST and not (_flag and _flag.get("code_version") == CODE_VERSION):
    run_smoke_test()
elif RUN_SMOKE_TEST:
    print("Smoke test already passed earlier (", _flag["passed_utc"], ") - skipped.")