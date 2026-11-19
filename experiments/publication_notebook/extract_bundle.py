#!/usr/bin/env python3
"""Unpack the results of final_baseline_run.ipynb and copy figures and tables into the paper folder.

Usage (from the BPD folder):
    python experiments/publication_notebook/extract_bundle.py                       # reads the bundle printed in the saved notebook
    python experiments/publication_notebook/extract_bundle.py --zip results.zip     # or a zip downloaded from Kaggle

Results go to experiments/final_run_outputs/; figures (PDF) to VBRI_Publication_Paper/figures/generated/;
LaTeX tables and numbers.json to VBRI_Publication_Paper/tables/. Uses only the Python standard library.
"""
import argparse, base64, hashlib, io, json, shutil, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # .../BPD
NB = ROOT / "experiments" / "final_baseline_run.ipynb"
OUT = ROOT / "experiments" / "final_run_outputs"
PAPER = ROOT / "VBRI_Publication_Paper"


def bundle_from_notebook(nb_path):
    nb = json.load(open(nb_path, encoding="utf-8"))
    text = ""
    for cell in nb["cells"]:
        for o in cell.get("outputs", []):
            if o.get("output_type") == "stream":
                t = o.get("text", "")
                text += "".join(t) if isinstance(t, list) else t
    a = text.rfind("BPD_BUNDLE_BEGIN")
    if a < 0:
        sys.exit("No bundle found. Run the export cell on Kaggle and SAVE the notebook in VS Code first.")
    b = text.find("BPD_BUNDLE_END", a)
    if b < 0:
        sys.exit("The bundle is incomplete (no end marker). Re-run the export cell and save again.")
    header, _, body = text[a:b].partition("\n")
    meta = dict(kv.split("=", 1) for kv in header.split()[1:])
    raw = base64.b64decode("".join(body.split()))
    if hashlib.sha256(raw).hexdigest() != meta["sha256"]:
        sys.exit("Checksum mismatch: the bundle in the notebook is damaged. Re-run the export cell.")
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", default=str(NB))
    ap.add_argument("--zip", default=None, help="zip downloaded from Kaggle instead of the printed bundle")
    ap.add_argument("--out", default=str(OUT), help="where to unpack the results")
    ap.add_argument("--paper", default=str(PAPER), help="paper folder that receives figures/generated and tables")
    args = ap.parse_args()
    out, paper = Path(args.out), Path(args.paper)
    raw = Path(args.zip).read_bytes() if args.zip else bundle_from_notebook(args.notebook)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for info in z.infolist():
            name = info.filename
            if name.startswith("final_baseline_run/"):
                name = name[len("final_baseline_run/"):]
            target = (out / name).resolve()
            if out.resolve() not in target.parents:              # never write outside the output folder
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(info))
    figs = out / "paper_assets" / "figures"
    tabs = out / "paper_assets" / "tables"
    dst_f, dst_t = paper / "figures" / "generated", paper / "tables"
    dst_f.mkdir(parents=True, exist_ok=True); dst_t.mkdir(parents=True, exist_ok=True)
    nf = nt = 0
    for p in sorted(figs.glob("*.pdf")) if figs.exists() else []:
        shutil.copy2(p, dst_f / p.name); nf += 1
    for p in sorted(tabs.glob("*.tex")) if tabs.exists() else []:
        shutil.copy2(p, dst_t / p.name); nt += 1
    for extra in ("numbers.json", "analysis.json", "data_stats.json"):
        if (out / "paper_assets" / extra).exists():
            shutil.copy2(out / "paper_assets" / extra, dst_t / extra)
    print(f"Unpacked to {out}")
    print(f"Copied {nf} figures to {dst_f} and {nt} tables (+ numbers.json) to {dst_t}")


if __name__ == "__main__":
    main()
