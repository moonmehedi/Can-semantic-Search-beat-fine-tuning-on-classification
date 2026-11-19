#!/usr/bin/env python3
"""
Download all papers from the Thesis DataMining - Personality CSV.
Uses Semantic Scholar API as the main fallback for open-access PDFs.
"""

import os
import time
import re
import requests
from pathlib import Path

LIT_DIR = Path("/media/thela-gari/NewVolume/Experience/Being an RA AT MIST/literature")
LIT_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# All 20 papers: (number, safe_filename_base, title, direct_link_or_None)
PAPERS = [
    (1,  "01_Jain2024_PD_Text_Classification",
     "Text classification models for personality disorders identification",
     None),
    (2,  "02_Dinu2021_Mental_Illness_Detection",
     "Automatic Detection and Classification of Mental Illnesses from General Social Media Texts",
     None),
    (3,  "03_Ellouze2022_Deep_Learning_Disease_Symptoms",
     "A Deep Learning Architecture Based on Advanced Textual Language Models for Detecting Disease Through Its Symptoms Associated with a Reinforcement Learning Algorithm",
     None),
    (4,  "04_Jiang2020_Reddit_Mental_Health_BERT",
     "Detection of Mental Health from Reddit via Deep Contextualized Representations",
     "https://aclanthology.org/2020.louhi-1.16.pdf"),
    (5,  "05_Singh2020_Antisocial_Twitter_NLP",
     "A Framework for Early Detection of Antisocial Behavior on Twitter Using Natural Language Processing",
     None),
    (6,  "06_Montejo2024_Survey_Mental_Disorders_NLP",
     "A survey on detecting mental disorders with natural language processing Literature review trends and challenges",
     None),
    (7,  "07_Zhang2022_NLP_Mental_Illness_Narrative_Review",
     "Natural language processing applied to mental illness detection a narrative review",
     "https://www.researchgate.net/profile/Tianlin-Zhang/publication/359816180_Natural_language_processing_applied_to_mental_illness_detection_a_narrative_review/links/62500435d726197cfd45b5e2/Natural-language-processing-applied-to-mental-illness-detection-a-narrative-review.pdf?_sg%5B0%5D=started_experiment_milestone&_sg%5B1%5D=started_experiment_milestone&origin=journalDetail"),
    (8,  "08_Yang2024_MentaLLaMA",
     "MentaLLaMA Interpretable Mental Health Analysis on Social Media with Large Language Models",
     "https://dl.acm.org/doi/pdf/10.1145/3589334.3648137"),
    (9,  "09_Vajre2021_PsychBERT",
     "PsychBERT A Mental Health Language Model for Social Media Mental Health Behavioral Analysis",
     None),   # IEEE requires auth; will use S2
    (10, "10_Ellouze2024_AI_Personality_Survey",
     "Artificial Intelligence application for the analysis of personality traits and disorders in social media A Survey",
     None),
    (11, "11_Xu2024_MentalLLM",
     "Mental-LLM Leveraging Large Language Models for Mental Health Prediction via Online Text Data",
     None),
    (12, "12_Ellouze2022_Hybrid_PD_Detection",
     "A hybrid approach for the detection and monitoring of people having personality disorders on social networks",
     None),   # Springer requires auth; will use S2
    (13, "13_Khazbak2021_MindTime_BPD",
     "MindTime Deep Learning Approach for Borderline Personality Disorder Detection",
     None),
    (14, "14_Martinez2022_BERT_Self_Harm",
     "Early Risk Detection of Self-Harm Using BERT-Based Transformers",
     None),
    (15, "15_Greco2023_Transformer_Mental_Health_Survey",
     "Transformer-based language models for mental health issues a survey",
     None),
    (16, "16_Harrigian2020_Social_Media_Mental_Health_Data",
     "On the State of Social Media Data for Mental Health Research",
     "https://arxiv.org/pdf/2011.05233"),
    (17, "17_Transformer_Suicidal_Ideation_IEEE",
     "A Transformer Based Approach To Detect Suicidal Ideation Using Pre-Trained Language Models",
     None),
    (18, "18_Mental_Health_QA_BERT_KnowledgeGraph",
     "Mental Health Question and Answering System Based on Bert Model and Knowledge Graph Technology",
     None),
    (19, "19_Deciphering_PD_LSTM_Social_Media",
     "Deciphering Personality Disorders in the Digital Realm A Deep Learning Approach Based on LSTM to Social Media Analysis",
     None),
    (20, "20_MHQA_Mental_Health_QA_Challenge",
     "MHQA A Diverse Knowledge Intensive Mental Health Question Answering Challenge for Language Models",
     "https://arxiv.org/pdf/2502.15418"),
]


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def download_pdf(url: str, dest: Path) -> bool:
    """Try to download a PDF from url to dest. Returns True on success."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
        if resp.status_code == 200 and len(resp.content) > 5000:
            content_type = resp.headers.get("Content-Type", "")
            if "pdf" in content_type or resp.content[:4] == b"%PDF":
                dest.write_bytes(resp.content)
                return True
            else:
                print(f"    [warn] Not a PDF (content-type: {content_type})")
        else:
            print(f"    [warn] HTTP {resp.status_code}, size={len(resp.content)}")
    except Exception as e:
        print(f"    [error] {e}")
    return False


def semantic_scholar_pdf(title: str) -> str | None:
    """Search Semantic Scholar for a paper and return its open-access PDF URL."""
    try:
        params = {
            "query": title,
            "fields": "title,year,openAccessPdf",
            "limit": 5,
        }
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params,
            headers=HEADERS,
            timeout=20,
        )
        if resp.status_code == 200:
            data = resp.json()
            for paper in data.get("data", []):
                oa = paper.get("openAccessPdf")
                if oa and oa.get("url"):
                    return oa["url"]
    except Exception as e:
        print(f"    [S2 error] {e}")
    return None


def arxiv_pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}"


failed = []
success = []

for no, fname_base, title, direct_link in PAPERS:
    pdf_path = LIT_DIR / f"{fname_base}.pdf"
    print(f"\n[{no:02d}] {title[:70]}...")

    if pdf_path.exists() and pdf_path.stat().st_size > 5000:
        print(f"  → Already exists, skipping.")
        success.append((no, fname_base))
        continue

    downloaded = False

    # 1. Try direct link from CSV
    if direct_link:
        # Convert arXiv abstract to PDF
        link = direct_link
        link = re.sub(r"arxiv\.org/abs/", "arxiv.org/pdf/", link)
        print(f"  → Trying direct link: {link[:80]}")
        downloaded = download_pdf(link, pdf_path)
        if downloaded:
            print(f"  ✓ Downloaded via direct link")

    # 2. Try Semantic Scholar open-access PDF
    if not downloaded:
        print(f"  → Trying Semantic Scholar...")
        s2_url = semantic_scholar_pdf(title)
        if s2_url:
            print(f"  → S2 URL: {s2_url[:80]}")
            downloaded = download_pdf(s2_url, pdf_path)
            if downloaded:
                print(f"  ✓ Downloaded via Semantic Scholar")
        else:
            print(f"  [S2] No open-access PDF found")

    if downloaded:
        success.append((no, fname_base))
    else:
        print(f"  ✗ FAILED — could not download paper {no}")
        failed.append((no, title))

    time.sleep(1.5)  # polite delay

print("\n" + "=" * 60)
print(f"RESULTS: {len(success)}/{len(PAPERS)} papers downloaded")
print(f"Saved to: {LIT_DIR}")

if failed:
    print(f"\nFailed ({len(failed)}):")
    for no, title in failed:
        print(f"  [{no:02d}] {title[:80]}")
else:
    print("\nAll papers downloaded successfully!")
