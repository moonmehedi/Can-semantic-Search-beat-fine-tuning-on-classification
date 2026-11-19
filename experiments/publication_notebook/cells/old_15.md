## 7. Main training: seeds x models

Order: seed 42 for all models first (so a complete result exists early), then seeds 43 and 44.
If DeBERTa-v3 collapses to a single class under fp16 (it did in earlier runs), it is retried **once in fp32**; both rows are kept and the fp32 run is clearly labelled `DeBERTa-fp32`.