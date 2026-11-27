# Experiment code: fine-tuned encoders vs. training-free semantic search (VBRI)

## Main experiment
`final_experiment.ipynb` runs data checks, fine-tuning (7 encoders x 3 seeds), retrieval baselines,
VBRI, calibration, ablation, statistics, figures and LaTeX tables.

* Hardware used: one NVIDIA Tesla T4 (Kaggle). Packages: `requirements.txt`.
* Data: the folders `imbalance_free_dataset` and `augmented_dataset` (not included). Place them next to
  the notebook, or set `DATASET_DIR_OVERRIDE` and `AUG_DATASET_DIR_OVERRIDE` in the first code cell.
  The notebook checks row counts and fingerprints.
* Run all cells. Finished training runs are skipped and interrupted runs resume.
* Outputs are written to the work folder set in the first code cell. Cell outputs are not stored here.

Notes on the reported run
* DeBERTa-v3 was trained in fp32. The other encoders used fp16 mixed precision. The fp32 cast and the
  DeBERTa precision setting were added after the other 18 runs; they do not change those runs.
* transformers 5.0.0 reported mismatched LayerNorm parameter names (beta/gamma vs. weight/bias) when
  some models reloaded their best checkpoint, so their LayerNorm values may not come from the best epoch.

## Extra script
`make_fig_encoder_text.py RETRIEVAL_DIR OUTPUT_PDF` draws the encoder / input-text figure and prints the
exact McNemar test on BPD answers (pooled index vs. VBRI). `RETRIEVAL_DIR` is the `retrieval` folder in
the notebook's work folder.

## earlier_runs/
Notebooks and figure scripts from an earlier version of the study. The current paper does not use them.
