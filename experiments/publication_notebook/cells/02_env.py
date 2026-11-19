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

VRAM_TOTAL_GB = VRAM_FREE_GB = None
if PROFILE == "windows" and cuda_ok:
    # A PyTorch build without kernels for this GPU fails later with "no kernel image is available". Check it now.
    _arch = f"sm_{CAP[0]}{CAP[1]}"
    _archs = torch.cuda.get_arch_list()
    if _archs and _arch not in _archs and not any(a.startswith(f"sm_{CAP[0]}") for a in _archs):
        raise SystemExit(f"This PyTorch build ({torch.__version__}, archs {_archs}) has no kernels for {GPU_NAME} ({_arch}).\n"
                         "Install a CUDA 12.8+ build, e.g.:  pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128")
    _x = torch.randn(256, 256, device="cuda")
    (_x @ _x).sum().item()                                # fails here if the GPU cannot run kernels
    torch.backends.cuda.matmul.allow_tf32 = True           # faster fp32 matmuls on Ampere and newer (training itself uses bf16)
    torch.backends.cudnn.allow_tf32 = True
    _free, _total = torch.cuda.mem_get_info()
    VRAM_FREE_GB, VRAM_TOTAL_GB = round(_free / 1e9, 1), round(_total / 1e9, 1)
    print(f"GPU memory: {VRAM_FREE_GB} GB free of {VRAM_TOTAL_GB} GB")
    if VRAM_FREE_GB < 12:
        print("!!! Less than 12 GB of GPU memory is free. Close other programs that use the GPU (games, other notebooks, "
              "other training jobs) before training, or runs may crash with 'CUDA out of memory' and the times are not clean.")

ENV = {
    "code_version": CODE_VERSION, "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "gpu": GPU_NAME, "compute_capability": list(CAP), "precision": PRECISION,
    "visible_gpus": torch.cuda.device_count() if cuda_ok else 0,
    "torch": torch.__version__, "transformers": transformers.__version__, "datasets": datasets.__version__,
    "python": platform.python_version(), "cpu_count": os.cpu_count(), "platform": platform.platform(),
    "profile": PROFILE, "vram_total_gb": VRAM_TOTAL_GB, "vram_free_gb_at_start": VRAM_FREE_GB,
    "full_batch": FULL_BATCH, "eval_bs": EVAL_BS, "encode_bs": ENCODE_BS,
}
print(json.dumps(ENV, indent=2))
if not cuda_ok:
    print("\n!!! No GPU found. Training times will not be meaningful. "
          + ("Enable a GPU accelerator in Kaggle." if PROFILE == "kaggle" else "Install a CUDA build of PyTorch (see the first markdown cell)."))
elif PROFILE == "kaggle" and "T4" not in GPU_NAME:
    print(f"\n!!! GPU is '{GPU_NAME}', not a Tesla T4. Times are only comparable with other runs on the same GPU type.")
elif PROFILE == "windows":
    print(f"\nAll times in this run are measured on: {GPU_NAME} (report this GPU in the paper).")
(WORK_DIR / "env.json").write_text(json.dumps(ENV, indent=2), encoding="utf-8")