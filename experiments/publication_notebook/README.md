# Publication notebook

* `../final_baseline_run.ipynb` — Kaggle version (VS Code + Kaggle remote kernel). Generated, do not edit by hand.
* `../final_baseline_run_windows.ipynb` — Windows workstation version (one local NVIDIA GPU). Generated, do not edit by hand.
* `requirements_windows.txt` — packages for the Windows PC (install PyTorch cu128 first, see the notebook's first cell).
* `cells/` — the source of every notebook cell. Edit these, then rebuild with `python3 build_notebook.py`.
* `build_notebook.py` — assembles both notebooks from the same cells (profile `kaggle` / `windows`) plus `*_script.py` copies for local testing.
* `extract_bundle.py` — after the Kaggle run, save the notebook and run `python experiments/publication_notebook/extract_bundle.py`
  (or `--zip <downloaded zip>`). Copies figures to `VBRI_Publication_Paper/figures/generated/` and tables to `VBRI_Publication_Paper/tables/`.

Local test (CPU, tiny subset, results not valid): `BPD_TEST_MODE=1 BPD_WORK_DIR=/some/dir python final_baseline_run_script.py` from `experiments/`.
