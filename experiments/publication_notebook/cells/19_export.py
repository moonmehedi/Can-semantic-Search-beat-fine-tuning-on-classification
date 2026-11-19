def _skip(p):
    parts = p.parts
    return (p.is_dir() or "final_model" in parts or "smoke" in parts or "emb" in parts
            or any(x.startswith("checkpoint-") for x in parts) or p.name.endswith(".tmp"))

def export_results():
    # 1) full zip (Kaggle: download it from the output panel; Windows: it sits next to the work folder)
    zpath = WORK_DIR.parent / "final_baseline_run_results.zip"
    n = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(WORK_DIR.rglob("*")):
            if not _skip(p):
                z.write(p, p.relative_to(WORK_DIR)); n += 1
    print(f"Wrote {zpath} ({n} files, {zpath.stat().st_size / 1e6:.1f} MB).")
    # 2) the same content (PDF figures, no PNG) printed into this notebook, for a VS Code remote kernel.
    #    Save the notebook locally, then run:  python experiments/publication_notebook/extract_bundle.py
    if EMBED_BUNDLE_IN_NOTEBOOK:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(WORK_DIR.rglob("*")):
                if not _skip(p) and p.suffix != ".png":
                    z.write(p, p.relative_to(WORK_DIR))
        raw = buf.getvalue()
        data = base64.b64encode(raw).decode("ascii")
        print(f"BPD_BUNDLE_BEGIN sha256={hashlib.sha256(raw).hexdigest()} bytes={len(raw)}")
        for i in range(0, len(data), 2000):
            print(data[i:i + 2000])
        print("BPD_BUNDLE_END")
        print(f"\nBundle printed ({len(raw) / 1e6:.2f} MB). Now SAVE this notebook in VS Code and run extract_bundle.py locally.")

def copy_to_paper():
    """Windows profile: copy final figures (PDF) and tables straight into the paper folder."""
    if PAPER_REPO_DIR is None or not Path(PAPER_REPO_DIR).exists():
        print("Paper folder not found (set BPD_PAPER_DIR); figures and tables stay in", PAPER_DIR)
        return
    dst_f, dst_t = Path(PAPER_REPO_DIR) / "figures" / "generated", Path(PAPER_REPO_DIR) / "tables"
    dst_f.mkdir(parents=True, exist_ok=True); dst_t.mkdir(parents=True, exist_ok=True)
    nf = nt = 0
    for p in sorted(FIG_DIR.glob("*.pdf")):
        shutil.copy2(p, dst_f / p.name); nf += 1
    for p in sorted(TAB_DIR.glob("*.tex")):
        shutil.copy2(p, dst_t / p.name); nt += 1
    for extra in ("numbers.json", "analysis.json", "data_stats.json"):
        if (PAPER_DIR / extra).exists():
            shutil.copy2(PAPER_DIR / extra, dst_t / extra)
    print(f"Copied {nf} figures to {dst_f} and {nt} tables (+ numbers.json) to {dst_t}")

if RUN_EXPORT:
    if TEST_MODE and COPY_TO_PAPER:
        print("TEST MODE: not copying test figures into the paper folder.")
    elif COPY_TO_PAPER:
        copy_to_paper()
    export_results()
