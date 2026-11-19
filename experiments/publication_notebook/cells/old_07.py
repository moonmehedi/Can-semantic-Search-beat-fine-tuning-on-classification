try:
    from transformers import TrainerCallback
except Exception:                      # lets the timing logic be unit-tested without transformers
    class TrainerCallback:
        pass


class RunTimer(TrainerCallback):
    """Active-training-time bookkeeping that survives crashes and resumes."""

    def __init__(self, run_dir, base=None, resumed_from_step=0, meta=None, heartbeat_s=20.0):
        self.run_dir = Path(run_dir)
        self.ledger_path = self.run_dir / "ledger.json"
        base = base or {}
        self.base_effective = float(base.get("effective_s", 0.0))
        self.base_eval = float(base.get("eval_s", 0.0))
        self.base_save = float(base.get("save_s", 0.0))
        self.resumed_from_step = int(resumed_from_step)
        self.meta = dict(meta or {})
        self.heartbeat_s = float(heartbeat_s)
        self.eval_s = 0.0            # evaluation time in this session
        self.save_s = 0.0            # checkpoint-saving time in this session
        self._t0 = None
        self._frozen = None
        self._last_hb = 0.0
        self._open = False
        self._ledger = None

    # ---- accessors
    def active_s(self):
        if self._t0 is None:
            return 0.0
        return self._frozen if self._frozen is not None else time.perf_counter() - self._t0

    def effective_s(self):
        return self.base_effective + self.active_s()

    def eval_total_s(self):
        return self.base_eval + self.eval_s

    def save_total_s(self):
        return self.base_save + self.save_s

    # ---- ledger
    def _load_ledger(self):
        led = read_json(self.ledger_path, None)
        if not isinstance(led, dict) or "sessions" not in led:
            led = {"code_version": CODE_VERSION, "created_utc": utc_now(), "sessions": []}
        return led

    def _heartbeat(self, status=None, step=None, epoch=None):
        s = self._ledger["sessions"][-1]
        s["active_s"] = round(self.active_s(), 3)
        s["effective_s_so_far"] = round(self.effective_s(), 3)
        s["last_heartbeat_utc"] = utc_now()
        if step is not None:
            s["last_step"] = int(step)
        if epoch is not None:
            s["last_epoch"] = float(epoch)
        if status:
            s["status"] = status
        atomic_write_json(self.ledger_path, self._ledger)
        self._last_hb = time.perf_counter()

    @staticmethod
    def summarize_ledger(led, effective_s, eval_s, save_s):
        gross = sum(float(s.get("active_s", 0.0)) for s in led["sessions"])
        return {
            "effective_s": round(effective_s, 3), "eval_s": round(eval_s, 3), "save_s": round(save_s, 3),
            "gross_s": round(gross, 3), "wasted_s": round(max(0.0, gross - effective_s), 3),
            "n_sessions": len(led["sessions"]),
            "resumed": any(s.get("resumed_from_step", 0) > 0 for s in led["sessions"]),
            "session_status": [s.get("status") for s in led["sessions"]],
        }

    def close(self, status, step=None, epoch=None):
        """Finish this session (status: completed / crashed). Safe to call twice."""
        if not self._open:
            return
        self._frozen = time.perf_counter() - self._t0
        self._heartbeat(status=status, step=step, epoch=epoch)
        self._ledger["summary"] = self.summarize_ledger(
            self._ledger, self.effective_s(), self.eval_total_s(), self.save_total_s())
        atomic_write_json(self.ledger_path, self._ledger)
        self._open = False
        if status == "completed":
            atomic_write_json(self.run_dir / "train_complete.json",
                              {"finished_utc": utc_now(), **self._ledger["summary"]})

    # ---- Trainer callback hooks
    def on_train_begin(self, args, state, control, **kwargs):
        led = self._load_ledger()
        for s in led["sessions"]:                      # a session still marked running was killed
            if s.get("status") == "running":
                s["status"] = "interrupted"
        led.pop("summary", None)
        self._t0 = time.perf_counter()
        self._frozen = None
        led["sessions"].append({
            "session": len(led["sessions"]) + 1, "started_utc": utc_now(),
            "resumed_from_step": self.resumed_from_step,
            "base_effective_s": round(self.base_effective, 3),
            "active_s": 0.0, "status": "running", **self.meta,
        })
        self._ledger = led
        self._open = True
        self._heartbeat()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self._open and time.perf_counter() - self._last_hb >= self.heartbeat_s:
            self._heartbeat(step=state.global_step, epoch=state.epoch)
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._open:
            self._heartbeat(step=state.global_step, epoch=state.epoch)
        return control

    def on_save(self, args, state, control, **kwargs):
        if not self._open:
            return control
        ck = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if ck.exists():
            atomic_write_json(ck / "timing_at_checkpoint.json", {
                "global_step": int(state.global_step), "epoch": float(state.epoch or 0.0),
                "effective_s": round(self.effective_s(), 3),
                "eval_s": round(self.eval_total_s(), 3), "save_s": round(self.save_total_s(), 3),
                "written_utc": utc_now(), "session": len(self._ledger["sessions"]),
            })
        self._heartbeat(step=state.global_step, epoch=state.epoch)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.close("completed", step=state.global_step, epoch=state.epoch)
        return control


# ---------------------------- checkpoint helpers ------------------------------
def list_checkpoints(run_dir):
    cks = [p for p in Path(run_dir).glob("checkpoint-*") if p.is_dir() and p.name.split("-")[-1].isdigit()]
    return sorted(cks, key=lambda p: int(p.name.split("-")[-1]), reverse=True)

def checkpoint_ok(ck):
    """A checkpoint is usable only if weights, trainer state and OUR timing file are all present and readable."""
    ck = Path(ck)
    has_weights = any((ck / n).exists() for n in
                      ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"))
    st = read_json(ck / "trainer_state.json", None)
    tm = read_json(ck / "timing_at_checkpoint.json", None)
    return bool(has_weights and isinstance(st, dict) and isinstance(tm, dict) and "effective_s" in tm)

def pick_resume_checkpoint(run_dir, log=print):
    """Latest valid checkpoint (and its timing). Invalid newer ones (e.g. half-written when Kaggle died) are deleted."""
    for ck in list_checkpoints(run_dir):
        if checkpoint_ok(ck):
            return ck, read_json(ck / "timing_at_checkpoint.json")
        log(f"  [resume] checkpoint {ck.name} is incomplete or has no timing file -> deleting it")
        shutil.rmtree(ck, ignore_errors=True)
    return None, None