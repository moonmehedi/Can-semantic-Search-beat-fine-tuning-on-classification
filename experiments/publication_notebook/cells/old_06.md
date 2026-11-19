## 3. Resume-safe timer

`RunTimer` is a Trainer callback. It keeps the numbers that matter in files on disk, so they survive a Kaggle restart:

| file | content |
|---|---|
| `ledger.json` | one entry per session: start time, GPU, active seconds (heartbeat every 20 s), status (`running`, `interrupted`, `crashed`, `completed`) and a summary |
| `checkpoint-N/timing_at_checkpoint.json` | cumulative effective training time at the moment checkpoint N was saved |
| `train_complete.json` | written when training finished normally |

Definitions: **effective** = time stored in the checkpoint we resumed from + active time after the resume. **gross** = sum of the active time of all sessions. **wasted** = gross - effective (the partial epoch lost in a crash). Evaluation and checkpoint-saving time are part of training time (as in Hugging Face's own `train_runtime`) and are also reported separately.