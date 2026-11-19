#!/usr/bin/env python3
"""
Third-pass: highly targeted attempts for the remaining 12 papers.
Uses NCBI OA API, specific arxiv IDs, S2 paper-ID lookup, etc.
"""

import time
import re
import requests
import xml.etree.ElementTree as ET
from pathlib import Path

LIT_DIR = Path("/media/thela-gari/NewVolume/Experience/Being an RA AT MIST/literature")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}


def is_pdf(content: bytes) -> bool:
    return b"%PDF" in content[:20]


def download_pdf(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, headers=HEADERS, timeout=45, allow_redirects=True)
        if r.status_code == 200 and len(r.content) > 8000 and is_pdf(r.content):
            dest.write_bytes(r.content)
            print(f"    ✓ {len(r.content)//1024} KB")
            return True
        print(f"    ✗ HTTP {r.status_code} size={len(r.content)} PDF={is_pdf(r.content)}")
    except Exception as e:
        print(f"    ✗ {e}")
    return False


def ncbi_oa_pdf(pmcid: str) -> str | None:
    """Use NCBI OA API to get the direct FTP/HTTPS PDF link for a PMC article."""
    try:
        url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            root = ET.fromstring(r.text)
            for link in root.iter("link"):
                href = link.get("href", "")
                fmt = link.get("format", "")
                if fmt == "pdf" and href:
                    # Convert ftp:// to https://
                    href = href.replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov")
                    return href
    except Exception as e:
        print(f"    [NCBI OA error] {e}")
    return None


def s2_by_title(title: str):
    """Return (openAccessPdfUrl, arxivId) from Semantic Scholar search."""
    try:
        r = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": title, "fields": "openAccessPdf,externalIds,title,year", "limit": 5},
            headers=HEADERS, timeout=20,
        )
        if r.status_code == 200:
            for p in r.json().get("data", []):
                oa = p.get("openAccessPdf") or {}
                if oa.get("url"):
                    return oa["url"], None
                arxiv_id = (p.get("externalIds") or {}).get("ArXiv")
                if arxiv_id:
                    return None, arxiv_id
    except Exception as e:
        print(f"    [S2 error] {e}")
    return None, None


TARGETS = [
    # (fname_base, title_for_search, urls_to_try_in_order)
    (
        "01_Jain2024_PD_Text_Classification",
        "Text classification models personality disorders identification Jain 2024",
        [
            # Try potential arxiv-style preprints or open versions
            "https://arxiv.org/search/?searchtype=all&query=Text+classification+personality+disorders+Jain+2024",
        ],
    ),
    (
        "03_Ellouze2022_Deep_Learning_Disease_Symptoms",
        "Deep Learning Architecture Textual Language Models Detecting Disease Symptoms Reinforcement Learning Ellouze",
        [],
    ),
    (
        "05_Singh2020_Antisocial_Twitter_NLP",
        "Framework Early Detection Antisocial Behavior Twitter Natural Language Processing Singh 2020",
        [
            "https://arxiv.org/pdf/1906.11353",  # possible preprint
        ],
    ),
    (
        "06_Montejo2024_Survey_Mental_Disorders_NLP",
        "survey detecting mental disorders natural language processing Montejo 2024",
        [
            "https://arxiv.org/pdf/2402.12577",  # possible preprint
            "https://arxiv.org/pdf/2312.05731",
        ],
    ),
    (
        "07_Zhang2022_NLP_Mental_Illness_Narrative_Review",
        "Natural language processing mental illness detection narrative review Zhang Schoene",
        [
            # NCBI OA API for PMC8993841
            "_NCBI_PMC8993841",
            # FTP direct
            "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa/pdf/a4/e4/41746_2022_Article_589.PMC8993841.pdf",
            # Try arxiv preprint
            "https://arxiv.org/pdf/2109.07956",
        ],
    ),
    (
        "10_Ellouze2024_AI_Personality_Survey",
        "Artificial Intelligence personality traits disorders social media survey Ellouze Belguith 2024",
        [],
    ),
    (
        "12_Ellouze2022_Hybrid_PD_Detection",
        "hybrid approach detection monitoring personality disorders social networks Ellouze 2022",
        [
            "_NCBI_PMC9244050",
        ],
    ),
    (
        "13_Khazbak2021_MindTime_BPD",
        "MindTime deep learning borderline personality disorder Khazbak 2021",
        [
            "https://arxiv.org/pdf/2103.13395",
        ],
    ),
    (
        "14_Martinez2022_BERT_Self_Harm",
        "Early risk detection self-harm BERT transformers erisk Martinez 2022",
        [
            "https://arxiv.org/pdf/2211.00437",
        ],
    ),
    (
        "17_Transformer_Suicidal_Ideation_IEEE",
        "Transformer approach detect suicidal ideation pre-trained language models BERT",
        [
            "https://arxiv.org/pdf/2110.06512",
            # PsychBERT was already downloaded for paper 9; try different ID
            "https://arxiv.org/pdf/2006.04733",
            "https://arxiv.org/pdf/2009.01062",
        ],
    ),
    (
        "18_Mental_Health_QA_BERT_KnowledgeGraph",
        "mental health question answering BERT knowledge graph ACM",
        [],
    ),
    (
        "19_Deciphering_PD_LSTM_Social_Media",
        "Deciphering personality disorders deep learning LSTM social media analysis IEEE",
        [],
    ),
]

success = []
failed = []

for fname_base, s2_title, extra_urls in TARGETS:
    pdf_path = LIT_DIR / f"{fname_base}.pdf"
    if pdf_path.exists() and pdf_path.stat().st_size > 5000:
        print(f"[✓ exists] {fname_base}")
        success.append(fname_base)
        continue

    print(f"\n>>> {fname_base}")
    downloaded = False

    for url in extra_urls:
        if downloaded:
            break
        # Special NCBI OA marker
        if url.startswith("_NCBI_"):
            pmcid = url[6:]
            print(f"  NCBI OA API for {pmcid}...")
            ncbi_url = ncbi_oa_pdf(pmcid)
            if ncbi_url:
                print(f"  → {ncbi_url[:80]}")
                downloaded = download_pdf(ncbi_url, pdf_path)
            else:
                print(f"  [NCBI] No FTP PDF found")
        else:
            print(f"  → {url[:80]}")
            downloaded = download_pdf(url, pdf_path)

    # Semantic Scholar fallback
    if not downloaded:
        print(f"  Trying S2 search: {s2_title[:60]}")
        oa_url, arxiv_id = s2_by_title(s2_title)
        if oa_url:
            print(f"  → S2 OA: {oa_url[:80]}")
            downloaded = download_pdf(oa_url, pdf_path)
        elif arxiv_id:
            ax_url = f"https://arxiv.org/pdf/{arxiv_id}"
            print(f"  → S2→ArXiv: {ax_url}")
            downloaded = download_pdf(ax_url, pdf_path)
        else:
            print(f"  [S2] Nothing found")

    if downloaded:
        success.append(fname_base)
    else:
        failed.append(fname_base)
        print(f"  ✗ FAILED")

    time.sleep(1.5)

# Summary
print("\n" + "=" * 60)
pdfs = sorted(LIT_DIR.glob("*.pdf"))
print(f"Total PDFs in literature/: {len(pdfs)}/{20}")
for p in pdfs:
    print(f"  {p.name}  ({p.stat().st_size//1024} KB)")

if failed:
    print(f"\nStill not downloaded ({len(failed)}):")
    for f in failed:
        print(f"  {f}")
