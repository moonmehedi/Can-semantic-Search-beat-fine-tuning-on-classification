MODEL_SPECS = {
    "ELECTRA":    dict(checkpoint="google/electra-base-discriminator", bs=16, eval_bs=16, accum=1, lr=5e-6, wd=0.1,  warmup=0.2),
    "BERT":       dict(checkpoint="bert-base-uncased",                 bs=16, eval_bs=16, accum=1, lr=5e-6, wd=0.1,  warmup=0.2),
    "DistilBERT": dict(checkpoint="distilbert/distilbert-base-uncased", bs=16, eval_bs=16, accum=1, lr=5e-6, wd=0.1,  warmup=0.2),
    "RoBERTa":    dict(checkpoint="FacebookAI/roberta-base",           bs=16, eval_bs=16, accum=1, lr=5e-6, wd=0.1,  warmup=0.2),
    "XLNet":      dict(checkpoint="xlnet/xlnet-base-cased",            bs=4,  eval_bs=2,  accum=4, lr=5e-6, wd=0.1,  warmup=0.2, eval_accum=8, pad_left=True),
    "DeBERTa":    dict(checkpoint="microsoft/deberta-v3-base",         bs=8,  eval_bs=8,  accum=2, lr=2e-5, wd=0.01, warmup=0.1),
    "ModernBERT": dict(checkpoint="answerdotai/ModernBERT-base",       bs=8,  eval_bs=8,  accum=2, lr=2e-5, wd=0.01, warmup=0.1),
}

from transformers import (AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,
                          Trainer, TrainingArguments, EarlyStoppingCallback, set_seed)

def _logits(pred):
    p = pred.predictions if hasattr(pred, "predictions") else pred
    return p[0] if isinstance(p, tuple) else p

def compute_metrics_hf(eval_pred):
    logits = _logits(eval_pred)
    y = eval_pred.label_ids
    m = summarize([int(v) for v in y], [int(v) for v in np.argmax(logits, axis=-1)])
    return {"accuracy": m["accuracy"] / 100.0, "f1": m["weighted_f1"] / 100.0}


class TimedTrainer(Trainer):
    """Trainer that also reports how much time went into evaluation and checkpoint saving."""
    timer = None

    def evaluate(self, *a, **k):
        t = time.perf_counter()
        out = super().evaluate(*a, **k)
        if self.timer is not None and self.timer._open:
            self.timer.eval_s += time.perf_counter() - t
        return out

    def _save_checkpoint(self, *a, **k):
        t = time.perf_counter()
        out = super()._save_checkpoint(*a, **k)
        if self.timer is not None and self.timer._open:
            self.timer.save_s += time.perf_counter() - t
        return out


def build_training_args(spec, seed, run_dir, epochs, precision):
    params = inspect.signature(TrainingArguments.__init__).parameters
    kw = dict(
        output_dir=str(run_dir),
        per_device_train_batch_size=spec["bs"], per_device_eval_batch_size=spec["eval_bs"],
        gradient_accumulation_steps=spec["accum"], num_train_epochs=epochs,
        learning_rate=spec["lr"], weight_decay=spec["wd"], lr_scheduler_type="linear",
        save_strategy="epoch", logging_strategy="epoch",
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        seed=seed, data_seed=seed, report_to="none",
        fp16=(precision == "fp16"), bf16=(precision == "bf16"),
    )
    kw["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "epoch"
    if "warmup_ratio" in params:
        kw["warmup_ratio"] = spec["warmup"]
    else:
        kw["warmup_steps"] = spec["warmup"]          # newer transformers accept a float ratio here
    if spec.get("eval_accum"):
        kw["eval_accumulation_steps"] = spec["eval_accum"]
    return TrainingArguments(**kw)


_TOKEN_CACHE = {}
def get_tokenised(model_name, subset=None):
    spec = MODEL_SPECS[model_name]
    if model_name not in _TOKEN_CACHE:
        tok = AutoTokenizer.from_pretrained(spec["checkpoint"])
        if spec.get("pad_left"):
            tok.padding_side = "left"                 # XLNet needs left padding
        def f(ex):
            return tok(ex["question"], ex["answer"], truncation=True)   # question + answer as a pair, dynamic padding later
        out = DatasetDict()
        for split in ("train", "validation", "test"):
            ds = CLEAN[split]
            out[split] = ds.map(f, batched=True, remove_columns=[c for c in ds.column_names if c != "label"])
        _TOKEN_CACHE[model_name] = (tok, out)
    tok, out = _TOKEN_CACHE[model_name]
    if subset:                                         # smoke test only
        tr = out["train"].shuffle(seed=0).select(range(subset["train"]))
        va = out["validation"].select(range(subset["val"]))
        out = DatasetDict({"train": tr, "validation": va, "test": out["test"]})
    return tok, out


def softmax_rows(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True)).tolist()


def write_test_probs(path, probs):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "question_id", "label", "p_normal", "p_bpd", "p_inconclusive"])
        for i, (pid, q, y, p) in enumerate(zip(TEST_IDS, TEST_QIDS, TEST_Y, probs)):
            w.writerow([pid, q, y, *[f"{v:.6f}" for v in p]])


def _last_state(run_dir):
    for ck in list_checkpoints(run_dir):
        st = read_json(ck / "trainer_state.json", None)
        if st:
            return st
    return None


def run_training(model_name, seed, variant="", precision=None, run_root=None, epochs=None, patience=None,
                 subset=None, crash_at_step=None, simulate_kill=False, heartbeat_s=20.0, keep_weights=None):
    """Train (or resume, or load a finished run of this code version). Returns the run's result dict."""
    spec = MODEL_SPECS[model_name]
    precision = precision or PRECISION
    epochs = epochs or MAX_EPOCHS
    patience = PATIENCE if patience is None else patience
    run_root = Path(run_root) if run_root else RUNS_DIR
    run_dir = run_root / f"seed{seed}" / f"{model_name}{variant}"
    run_dir.mkdir(parents=True, exist_ok=True)
    keep_weights = (seed in KEEP_WEIGHTS_SEEDS) if keep_weights is None else keep_weights
    tag = f"seed{seed}/{model_name}{variant}"

    done = read_json(run_dir / "result.json", None)
    if done and done.get("code_version") == CODE_VERSION and done.get("status") == "done":
        print(f"[{tag}] already finished by this notebook -> skipped (effective time {done['timing']['effective_s']:.0f}s)")
        return done

    print(f"\n{'='*78}\n[{tag}] precision={precision}  gpu={GPU_NAME}\n{'='*78}")
    set_seed(seed)
    tok, ds = get_tokenised(model_name, subset)
    model = AutoModelForSequenceClassification.from_pretrained(spec["checkpoint"], num_labels=3)
    if spec.get("pad_left"):
        model.config.pad_token_id = tok.pad_token_id
    n_params = sum(p.numel() for p in model.parameters())

    ck, tm = pick_resume_checkpoint(run_dir)
    train_done = (run_dir / "train_complete.json").exists() and ck is not None
    base = {"effective_s": tm["effective_s"], "eval_s": tm["eval_s"], "save_s": tm["save_s"]} if tm else {}
    resumed_from = int(tm["global_step"]) if tm else 0
    if ck is None and (run_dir / "ledger.json").exists() and not train_done:
        print(f"[{tag}] earlier session left no usable checkpoint -> restarting this run from zero (the lost time is kept as 'wasted')")
    if ck is not None and not train_done:
        es = (read_json(ck / "trainer_state.json", {}) or {}).get("stateful_callbacks", {})
        print(f"[{tag}] RESUMING from {ck.name} (time stored in checkpoint: {tm['effective_s']:.0f}s); "
              f"early-stopping state in checkpoint: {es.get('EarlyStoppingCallback', 'not stored by this transformers version')}")

    args = build_training_args(spec, seed, run_dir, epochs, precision)
    meta = {"model": model_name, "variant": variant, "seed": seed, "gpu": GPU_NAME, "precision": precision}
    timer = RunTimer(run_dir, base=base, resumed_from_step=resumed_from, meta=meta, heartbeat_s=heartbeat_s)
    cbs = [EarlyStoppingCallback(early_stopping_patience=patience), timer]
    if crash_at_step:
        marker = run_dir / "crash_marker"
        class _CrashOnce(TrainerCallback):
            def on_step_end(self, args, state, control, **kw):
                if state.global_step >= crash_at_step and not marker.exists():
                    marker.write_text("crashed")
                    raise RuntimeError("SIMULATED CRASH (smoke test)")
                return control
        cbs.append(_CrashOnce())
    tparams = inspect.signature(Trainer.__init__).parameters
    tk = dict(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"],
              data_collator=DataCollatorWithPadding(tokenizer=tok), compute_metrics=compute_metrics_hf, callbacks=cbs)
    tk["processing_class" if "processing_class" in tparams else "tokenizer"] = tok
    trainer = TimedTrainer(**tk)
    trainer.timer = timer

    if train_done:
        st = _last_state(run_dir)
        best = st.get("best_model_checkpoint") if st else None
        print(f"[{tag}] training had already finished -> loading best checkpoint {best} for evaluation (no extra training)")
        trainer.model = AutoModelForSequenceClassification.from_pretrained(best).to(args.device)
        history = st.get("log_history", [])
        summary = read_json(run_dir / "ledger.json", {}).get("summary") or read_json(run_dir / "train_complete.json", {})
    else:
        try:
            trainer.train(resume_from_checkpoint=str(ck) if ck else None)
        except Exception:
            if not simulate_kill:
                timer.close("crashed")            # a real kill cannot do this; the heartbeat then holds the last known time
            raise
        history = trainer.state.log_history
        summary = read_json(run_dir / "ledger.json")["summary"]

    # ---- predictions (not part of the training time) ---------------------------------
    val_logits = _logits(trainer.predict(ds["validation"]))
    test_logits = _logits(trainer.predict(ds["test"]))
    val_y = [int(v) for v in ds["validation"]["label"]]
    val_pred = [int(v) for v in np.argmax(val_logits, axis=-1)]
    test_probs = softmax_rows(test_logits)
    test_pred = [int(np.argmax(p)) for p in test_probs]
    val_m, test_m = summarize(val_y, val_pred), summarize(TEST_Y, test_pred)
    collapsed = len(set(val_pred)) == 1                # model predicts one class for every validation answer
    write_test_probs(run_dir / "test_probs.csv", test_probs)
    atomic_write_json(run_dir / "history.json", history)

    evals = [h for h in history if "eval_loss" in h]
    best_ev = min(evals, key=lambda h: h["eval_loss"]) if evals else {}
    result = {
        "code_version": CODE_VERSION, "status": "done", "model": model_name, "variant": variant, "seed": seed,
        "checkpoint": spec["checkpoint"], "params_m": round(n_params / 1e6, 1), "precision": precision, "gpu": GPU_NAME,
        "dataset": DATASET_REPORT["path"], "epochs_run": len(evals), "best_epoch": best_ev.get("epoch"),
        "best_val_loss": best_ev.get("eval_loss"), "collapsed": collapsed,
        "timing": summary, "val": val_m, "test": test_m,
        "hf_train_runtime_last_session_s": next((h.get("train_runtime") for h in reversed(history) if "train_runtime" in h), None),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "finished_utc": utc_now(),
    }

    # ---- weights: keep best model for the benchmark seed, drop everything else -----------
    if keep_weights:
        trainer.save_model(str(run_dir / "final_model"))
        tok.save_pretrained(str(run_dir / "final_model"))
        result["weights_mb"] = round(sum(f.stat().st_size for f in (run_dir / "final_model").glob("*")
                                         if f.suffix in (".safetensors", ".bin")) / 1e6, 1)
    atomic_write_json(run_dir / "result.json", result)         # completion marker, written BEFORE the cleanup below
    for c in list_checkpoints(run_dir):
        shutil.rmtree(c, ignore_errors=True)

    t = summary
    print(f"[{tag}] DONE  test acc {test_m['accuracy']:.2f}  BPD sens {test_m['bpd_sensitivity']:.2f}  | "
          f"train time effective {t['effective_s']:.0f}s, gross {t['gross_s']:.0f}s, wasted {t['wasted_s']:.0f}s, "
          f"sessions {t['n_sessions']}{'  (resumed)' if t['resumed'] else ''}  | epochs {len(evals)}"
          f"{'  | !!! COLLAPSED (single class)' if collapsed else ''}")
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result