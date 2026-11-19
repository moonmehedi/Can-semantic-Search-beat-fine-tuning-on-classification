# BACKGROUND

Borderline personality disorder (BPD) is typically evaluated through a structured clinical process rather than isolated statements. Assessment focuses on recurring patterns across domains such as fear of abandonment, unstable relationships, identity disturbance, impulsivity, self-harm, affective instability, chronic emptiness, anger dysregulation, and stress-related suspiciousness or dissociation [1]. Therefore, a screening-oriented text dataset for BPD requires responses linked to specific diagnostic themes rather than general expressions of distress. Although recent mental-health NLP research has produced many useful resources, several limitations remain. Many datasets are derived from social-media platforms such as Reddit and Twitter/X, leading to platform bias and limited demographic diversity [2,3]. Labels are often based on self-disclosure or self-reported diagnoses instead of expert assessment, raising concerns regarding reliability [3]. Privacy and consent issues are also significant because many datasets reuse publicly available content without direct participant involvement, and only a small proportion are openly accessible [2,4]. These concerns are particularly important for BPD, where screening depends on symptom persistence and contextual interpretation.

Within personality-disorder NLP, most studies focus on generalized prediction from online text rather than developing criterion-oriented resources for a single disorder. PD-TXT supports multi-class personality-disorder classification using data from Quora, Reddit, and YouTube, but it is not designed around BPD-specific screening questions [5]. Similarly, MindTime analyzes diary entries for BPD symptom detection, while Wang et al. distinguished bipolar disorder, BPD, and healthy controls using 50 non-clinical interviews, none of which employ BPD-specific diagnostic prompts [6,7]. MHQA, meanwhile, evaluates language-model knowledge rather than screening individual responses [8].

Against this background, BPDQAC was developed as a dataset based on a 14-question instrument aligned with DSM-5 criteria. It includes nine open-ended questions targeting Criterion A symptom domains and five yes/no questions covering Criteria B-F. The dataset documents the collection process, annotation framework, and screening-oriented design, providing a reusable resource for BPD-focused natural language processing and clinically motivated question-answer classification.

**Table 1. Relevant datasets and their limitations compared with BPDQAC.**

| Reference | Dataset Name | No. of Records | Dataset Style | Limitations |
|---|---|---:|---|---|
| [9] | Borderline personality disorder and depression in Bangladeshi university students | 535 survey records | Cross-sectional online survey using MSI-BPD, PHQ-9, demographic items, and COVID-19-related mental health questions | Relies on self-administered scale responses and self-reported symptom scores. It does not contain clinically annotated free-text answers or answer-level labels for supervised BPD screening. |
| [5] | PD-TXT | 9,129 sentences | Social media text from Quora, Reddit, and YouTube, labeled for multiple personality-disorder-related classes | Focuses on broad personality disorder classification rather than BPD-specific screening. It is social-media based, cue-word filtered, class-imbalanced, and not aligned with structured diagnostic questioning. |
| [6] | MindTime | Not reported in available source | Diary-style text collected through a mobile application for BPD symptom detection | Uses personal diary entries rather than responses to structured clinical questions. The data are not presented as DSM-aligned question-answer pairs or designed for answer-level screening classification. |
| [7] | AMoSS-I | 50 interviews | Multimodal non-clinical interviews using speech and transcripts from BD, BPD, and healthy-control participants | Uses a small sample and focuses on distinguishing BD, BPD, and healthy controls. The interviews concern non-clinical topics and do not contain expert-annotated answers to BPD-specific diagnostic questions. |
| [8] | MHQA | Approximately 58,600 QA pairs, including 2,475 expert-verified items | Multiple-choice mental health question-answering benchmark generated from PubMed abstracts | Assesses mental-health knowledge in language models rather than screening participants. It does not focus on BPD or classify individuals' responses to diagnostic questions. |

# REFERENCES

[1] American Psychiatric Association, Diagnostic and Statistical Manual of Mental Disorders: DSM-5-TR, American Psychiatric Association Publishing, Washington, DC, 2022.

[2] A. Montejo-Raez, M.D. Molina-Gonzalez, S.M. Jimenez-Zafra, M.A. Garcia-Cumbreras, L.J. Garcia-Lopez, A survey on detecting mental disorders with natural language processing: Literature review, trends and challenges, Computer Science Review 53 (2024) 100654.

[3] T. Zhang, A.M. Schoene, S. Ji, S. Ananiadou, Natural language processing applied to mental illness detection: a narrative review, NPJ Digital Medicine 5 (2022) 1-13.

[4] K. Harrigian, C. Aguirre, M. Dredze, On the state of social media data for mental health research, arXiv preprint arXiv:2011.05233, 2021.

[5] D. Jain, S. Arora, C.K. Jha, G. Malik, Text classification models for personality disorders identification, Social Network Analysis and Mining 14 (2024) 64. https://doi.org/10.1007/s13278-024-01219-8.

[6] M. Khazbak, Z. Wael, Z. Ehab, M. Gerorge, E. Eliwa, MindTime: deep learning approach for borderline personality disorder detection, in: 2021 International Mobile, Intelligent, and Ubiquitous Computing Conference (MIUCC), IEEE, 2021, pp. 337-344.

[7] B. Wang, Y. Wu, N. Taylor, T. Lyons, M. Liakata, A.J. Nevado-Holgado, K.E.A. Saunders, Learning to detect bipolar disorder and borderline personality disorder with language and speech in non-clinical interviews, in: Proceedings of Interspeech 2020, 2020, pp. 4481-4485. https://doi.org/10.21437/Interspeech.2020-3040.

[8] S. Racha, P. Joshi, A. Raman, N. Jangid, MHQA: A diverse, knowledge intensive mental health question answering challenge for language models, arXiv preprint arXiv:2502.15418, 2025.

[9] M.N.A. Khan, A. Islam, Borderline personality disorder and depression in Bangladeshi university students [Dataset], Mendeley Data, Version 1, 2023. https://doi.org/10.17632/prv6tgkfgr.1.
