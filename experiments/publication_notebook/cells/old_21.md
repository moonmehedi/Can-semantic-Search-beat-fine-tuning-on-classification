## 10. VBRI benchmark (same session, same hardware)

Frozen `BAAI/bge-base-en-v1.5`, nine per-criterion databases built from the **same 4,050 training answers** the baselines were trained on, similarity-weighted k-NN vote (answers are used as raw text, as in the original VBRI experiments).
This part measures: index-build time, per-query latency (encode + search) on GPU and CPU, index size, and accuracy. **k is chosen on the validation set** (not on the test set); k = 5 is also reported for comparison.