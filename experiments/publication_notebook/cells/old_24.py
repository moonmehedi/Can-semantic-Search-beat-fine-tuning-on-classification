def package():
    zpath = WORK_DIR.parent / "final_baseline_run_results.zip"
    n = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in WORK_DIR.rglob("*"):
            if p.is_dir() or "final_model" in p.parts or any(part.startswith("checkpoint-") for part in p.parts) or "smoke" in p.parts:
                continue
            z.write(p, p.relative_to(WORK_DIR.parent)); n += 1
    print(f"Wrote {zpath} ({n} files, {zpath.stat().st_size/1e6:.1f} MB). Download it from the Kaggle output panel.")

package()