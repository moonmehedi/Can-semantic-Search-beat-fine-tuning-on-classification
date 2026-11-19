# Final baseline run: fine-tuned encoders with an honest, resume-safe training time

**Goal.** Re-train the fine-tuned baselines from scratch and record a *correct* training time, even if Kaggle stops the session and we resume.

**Dataset.** `imbalance_free_dataset` only (train 4,050 = 1,350 per class, validation 333, test 333). This is the same training data that VBRI stores, so the comparison is controlled. The notebook refuses to run on any other dataset.

**How the training time is kept correct**
* A timer callback measures active training time (`perf_counter`) and writes a heartbeat to disk every 20 s.
* Every checkpoint gets a `timing_at_checkpoint.json` with the total active time up to that point.
* After a crash the run resumes from the last *valid* checkpoint and continues from the time stored in that checkpoint, not from zero and not from Hugging Face's `train_runtime` (which only counts the resumed part).
* Per run we report: **effective time** (time of an uninterrupted run, use this in the paper), **gross time** (all compute spent, including work lost in a crash) and **wasted time** (gross minus effective).
* A run is skipped only if it was completed by *this* code (`result.json` with the same `CODE_VERSION`). Old saved models are never reused.

**Kaggle settings:** Accelerator = GPU T4 (one GPU is used), Persistence = *Files only* (or more), Internet = On.
**If the session dies:** open the notebook again and press *Run All*. Finished runs are skipped, the interrupted run resumes from its last checkpoint.

**Stages:** 1 setup -> 2 smoke test (crash and resume check) -> 3 training (seeds x models) -> 4 summary tables -> 5 inference benchmark -> 6 VBRI benchmark (same session, same hardware) -> 7 zip results.