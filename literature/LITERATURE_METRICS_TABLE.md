# Literature Metrics Table

This file focuses on the main evaluation number reported by each PDF in `literature/`, especially `F1`, `accuracy`, or the paper's primary metric when `F1` is not the one used.

## Per-paper metric table

| Source file | Task / paper type | Relevant metric from the paper | Best reported result to keep in mind | Note for your thesis |
| --- | --- | --- | --- | --- |
| `04_Jiang2020_Reddit_Mental_Health_BERT.pdf` | Reddit mental health detection | F1 | Best user-level binary F1 = `0.931` for bipolar under UNCL; multi-label ensemble only `F1-micro = 0.2175`, `F1-macro = 0.195` | Strong binary results, but much weaker when the task becomes true multi-label classification |
| `08_Yang2024_MentaLLaMA.pdf` | Multi-task mental-health benchmark with LLMs | Weighted F1 | Best MentaLLaMA result in Table 2 = `85.68` weighted F1 on DR with `MentaLLaMA-chat-13B`; some discriminative baselines go up to `95.11` on DR | Useful for modern LLM comparison, but not a personality-disorder benchmark |
| `10_3674971.pdf` | Survey | No single benchmark metric | `N/A` | Review paper, so do not use it as an expected F1 target |
| `11_3643540.pdf` | Mental-health prediction with LLMs | Balanced accuracy | Best instruction-tuned result in Table 4 = `0.868` balanced accuracy with `Mental-FLAN-T5` on CSSRS Task #5 | Uses balanced accuracy instead of F1, so compare carefully |
| `12_s13278-022-00884-x.pdf` | Personality-disorder detection and monitoring on social networks | F-measure / accuracy | `0.84` F-measure for PD detection, `0.64` for disease-type detection, `0.70` for inappropriate-content filtering | Very relevant because it is directly about personality disorders |
| `16_Harrigian2020_Social_Media_Mental_Health_Data.pdf` | Dataset/resource analysis | No classifier metric | `N/A` | Resource paper, not a model benchmark |
| `17_A Transformer Based Approach To Detect Suicidal Ideation Using Pre-Trained Language Models.pdf` | Suicidal ideation detection | Accuracy and F1 | `RoBERTa`: `95.21%` accuracy, `95.47%` F1 | Strong result, but binary suicide detection is easier and not directly the same as personality-disorder classification |
| `18_3500931.3501011.pdf` | Mental-health QA system | Accuracy | Manual test shows `14/20` correct answers, about `70%` accuracy | QA system paper, not a diagnosis/classification benchmark |
| `19_Deciphering Personality Disorders in the Digital Realm_ A Deep Learning Approach Based on LSTM to Social Media Analysis.pdf` | Personality-disorder detection from social media | Accuracy / recall | LSTM deep learning reports about `0.89` accuracy and `0.90` recall; ML baselines are around `0.888-0.889` accuracy | Relevant topic, but reporting is weaker than the better papers and no clear F1 table was given |
| `1_Text classification models for personality disorders identification.pdf` | Multi-class personality-disorder identification | Accuracy and weighted F1 | `BERT-base-uncased`: `0.750 +/- 0.008` accuracy and `0.747 +/- 0.010` weighted F1 | One of the most relevant papers for your expected result |
| `1_s13278-024-01219-8 (1).pdf` | Same study as above | Accuracy and weighted F1 | `BERT-base-uncased`: `0.750 +/- 0.008` accuracy and `0.747 +/- 0.010` weighted F1 | Duplicate copy of the same paper above |
| `20_MHQA_Mental_Health_QA_Challenge.pdf` | Mental-health QA benchmark | F1 and accuracy | Best overall discriminative result = `77.8%` accuracy / `77.8%` F1 with `BioBERT SFT`; best LLM overall = `79.9%` accuracy / `79.8%` F1 with `GPT-4o` zero-shot | QA benchmark, not personality-disorder classification |
| `21_Learning to Detect Bipolar Disorder and Borderline Personality Disorder with Language and Speech in Non-Clinical Interviews.pdf` | Bipolar/BPD detection from interviews | AUROC | `0.810` for healthy vs BD, `0.733` for healthy vs BPD, `0.817` for BD vs BPD | Relevant disorders, but different modality: interview speech/language rather than social media text only |
| `2_Automatic Detection and Classification of Mental Illnesses from General Social Media Texts.pdf` | General mental-illness detection from social media | F1 | Best illness-specific F1 is `0.81` for eating disorder; most results are roughly `0.70-0.76`; depression is among the lowest around `0.70` | Good reference for general social-media mental-health text classification |
| `6_A survey on detecting mental disorders with natural language processing_ Literature review, trends and challenges.pdf` | Survey | No single benchmark metric | `N/A` | Review paper, not a direct benchmark |
| `6_Automatic Detection and Classification of Mental Illnesses from General Social Media Texts.pdf` | Same study as above | F1 | Best illness-specific F1 is `0.81` for eating disorder; most results are roughly `0.70-0.76` | Duplicate copy of the same paper above |
| `7_s41746-022-00589-7.pdf` | Narrative review | No single benchmark metric | `N/A` | Review paper, not a direct benchmark |
| `9_PsychBERT_ A Mental Health Language Model for Social Media Mental Health Behavioral Analysis.pdf` | Two-stage mental-health behavior analysis | F1 | Stage I binary F1 = `0.98` with `PsychBERT`; Stage II multi-class F1 = `0.63` with `PsychBERT` | Important reminder that binary detection can look very high, while multi-class is much harder |

## What looks like a realistic expected score for you

If your task is close to **multi-class personality-disorder identification from social media text**, the most relevant papers suggest:

- A realistic target is around `0.70-0.75` weighted F1 and around `0.73-0.75` accuracy
- A stronger result would be anything above about `0.75` weighted F1 on a fair split

If your task is closer to **binary PD detection** instead of multi-class subtype prediction, papers report much higher numbers:

- Around `0.84` F1 in the 2022 PD detection paper
- Around `0.89` accuracy in the LSTM paper

If your task is **fine-grained subtype detection**, expect it to be lower than coarse binary detection:

- One directly relevant paper reports only `0.64` F-measure for disease-type detection
- This means subtype prediction is likely the harder target, especially with class imbalance and overlap between disorders

## Bottom line

For a thesis like yours, the safest expectation is:

- `~0.70-0.75` weighted F1 for a solid multi-class PD classifier
- `~0.80+` only if the task is simpler, more binary, or the dataset is easier
- Be careful comparing against papers that use QA, AUROC, balanced accuracy, or easier binary settings
