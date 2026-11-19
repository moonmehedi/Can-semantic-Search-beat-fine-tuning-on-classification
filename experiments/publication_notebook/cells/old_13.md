## 6. Smoke test: does the timer survive a crash?

Trains DistilBERT on a small subset three times: (A) is killed in the middle of epoch 2 *without any clean-up* (like Kaggle dying), (B) resumes and finishes, (C) is an uninterrupted reference run.
Expected: B resumes (2 sessions), `wasted > 0`, and B's **effective** time is close to C's. If this fails the notebook stops, so no hours are spent on wrong timings. Takes a few minutes; it is skipped once it has passed.