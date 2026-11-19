import torch, transformers, datasets
import numpy as np

# ModernBERT needs transformers >= 4.48. Upgrade if needed, then the kernel must be restarted once.
def _ver(v):
    return tuple(int(x) for x in re.findall(r"\d+", v)[:2])
if _ver(transformers.__version__) < (4, 48):
    _pip("transformers>=4.48", "accelerate")
    raise SystemExit("transformers was upgraded for ModernBERT. RESTART THE KERNEL, then run all cells again.")

cuda_ok = torch.cuda.is_available()
GPU_NAME = torch.cuda.get_device_name(0) if cuda_ok else "cpu"
CAP = torch.cuda.get_device_capability(0) if cuda_ok else (0, 0)
# T4 (compute capability 7.5) has no real bf16 -> fp16 for every model. bf16 only on Ampere+ (cc >= 8).
PRECISION = "bf16" if CAP[0] >= 8 else ("fp16" if cuda_ok else "fp32")

ENV = {
    "code_version": CODE_VERSION, "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "gpu": GPU_NAME, "compute_capability": list(CAP), "precision": PRECISION,
    "visible_gpus": torch.cuda.device_count() if cuda_ok else 0,
    "torch": torch.__version__, "transformers": transformers.__version__, "datasets": datasets.__version__,
    "python": platform.python_version(), "cpu_count": os.cpu_count(), "platform": platform.platform(),
}
print(json.dumps(ENV, indent=2))
if not cuda_ok:
    print("\n!!! No GPU found. Training times will not be meaningful. Enable a GPU accelerator in Kaggle.")
elif "T4" not in GPU_NAME:
    print(f"\n!!! GPU is '{GPU_NAME}', not a Tesla T4. Times are only comparable with other runs on the same GPU type.")
(WORK_DIR / "env.json").write_text(json.dumps(ENV, indent=2))