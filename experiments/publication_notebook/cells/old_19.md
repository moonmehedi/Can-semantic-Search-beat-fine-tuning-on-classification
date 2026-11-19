## 9. Inference benchmark (same session, same GPU as training)

Per-response latency **including tokenisation**, measured with warm-up and `torch.cuda.synchronize()`. Models are loaded in fp32 (so weights size = parameters x 4 bytes).
* GPU, batch size 1 (one answer at a time, like a clinic tool)
* GPU, batch size 32 (time per answer)
* CPU, batch size 1 (the resource-constrained case), first 100 test answers