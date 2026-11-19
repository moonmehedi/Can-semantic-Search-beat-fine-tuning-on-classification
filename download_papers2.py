#!/usr/bin/env python3
"""
Second-pass downloader: tries more targeted sources for papers
that failed in the first pass.
"""

import time
import re
import requests
from pathlib import Path

LIT_DIR = Path("/media/thela-gari/NewVolume/Experience/Being an RA AT MIST/literature")
LIT_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/pdf,*/*",
}

EMAIL = "researcher@example.com"  # for Unpaywall


def is_pdf(content: bytes) -> bool:
    return content[:4] == b"%PDF" or b"%PDF" in content[:20]


def download_pdf(url: str, dest: Path) -> bool:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=40, allow_redirects=True)
        if resp.status_code == 200 and len(resp.content) > 8000 and is_pdf(resp.content):
            dest.write_bytes(resp.content)
            print(f"    ✓ Saved {len(resp.content)//1024} KB")
            return True
        else:
            print(f"    ✗ HTTP {resp.status_code}, size={len(resp.content)}, PDF={is_pdf(resp.content)}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
    return False


def unpaywall_pdf(doi: str) -> str | None:
    try:
        url = f"https://api.unpaywall.org/v2/{doi}?email={EMAIL}"
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            best = data.get("best_oa_location") or {}
            pdf_url = best.get("url_for_pdf") or best.get("url")
            if pdf_url:
                return pdf_url
            for loc in data.get("oa_locations", []):
                if loc.get("url_for_pdf"):
                    return loc["url_for_pdf"]
    except Exception as e:
        print(f"    [Unpaywall error] {e}")
    return None


def s2_search_pdf(title: str) -> str | None:
    try:
        r = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": title, "fields": "openAccessPdf,externalIds", "limit": 5},
            headers=HEADERS, timeout=20,
        )
        if r.status_code == 200:
            for p in r.json().get("data", []):
                oa = p.get("openAccessPdf")
                if oa and oa.get("url"):
                    return oa["url"]
                eids = p.get("externalIds", {})
                arxiv_id = eids.get("ArXiv")
                if arxiv_id:
                    return f"https://arxiv.org/pdf/{arxiv_id}"
    except Exception as e:
        print(f"    [S2 error] {e}")
    return None


def europepmc_pdf(doi: str) -> str | None:
    try:
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=doi:{doi}&resultType=core&format=json"
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            results = r.json().get("resultList", {}).get("result", [])
            for result in results:
                pmcid = result.get("pmcid")
                if pmcid:
                    # Try EuropePMC PDF
                    pdf_url = f"https://europepmc.org/articles/{pmcid}/pdf"
                    print(f"    [EuropePMC] Found PMC: {pmcid}")
                    return pdf_url
    except Exception as e:
        print(f"    [EuropePMC error] {e}")
    return None


# Papers still to download: (filename_base, title, known_doi, extra_urls_to_try)
REMAINING = [
    (
        "01_Jain2024_PD_Text_Classification",
        "Text classification models for personality disorders identification 2024",
        None,
        [],
    ),
    (
        "02_Dinu2021_Mental_Illness_Detection",
        "Automatic Detection and Classification of Mental Illnesses from General Social Media Texts",
        None,
        [
            # ACL Anthology RANLP 2021 (trying likely IDs)
            "https://aclanthology.org/2021.ranlp-1.40.pdf",
            "https://aclanthology.org/2021.ranlp-1.41.pdf",
            "https://aclanthology.org/2021.ranlp-1.42.pdf",
            "https://aclanthology.org/2021.ranlp-1.43.pdf",
            "https://aclanthology.org/2021.ranlp-1.44.pdf",
            "https://aclanthology.org/2021.ranlp-1.39.pdf",
        ],
    ),
    (
        "03_Ellouze2022_Deep_Learning_Disease_Symptoms",
        "Deep Learning Architecture Advanced Textual Language Models Detecting Disease Symptoms Reinforcement Learning",
        None,
        [],
    ),
    (
        "05_Singh2020_Antisocial_Twitter_NLP",
        "Framework Early Detection Antisocial Behavior Twitter Natural Language Processing 2020",
        None,
        [],
    ),
    (
        "06_Montejo2024_Survey_Mental_Disorders_NLP",
        "survey detecting mental disorders natural language processing literature review trends challenges 2024",
        "10.1016/j.cosrev.2024.100654",
        [],
    ),
    (
        "07_Zhang2022_NLP_Mental_Illness_Narrative_Review",
        "Natural language processing applied to mental illness detection narrative review",
        "10.1038/s41746-022-00589-7",
        [
            # NPJ Digital Medicine is fully open access
            "https://www.nature.com/articles/s41746-022-00589-7.pdf",
            "https://europepmc.org/articles/PMC9360980/pdf",
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9360980/pdf",
        ],
    ),
    (
        "08_Yang2024_MentaLLaMA",
        "MentaLLaMA interpretable mental health analysis social media large language models",
        "10.1145/3589334.3648137",
        [
            "https://arxiv.org/pdf/2309.13567",
            "https://arxiv.org/pdf/2309.13567v2",
        ],
    ),
    (
        "09_Vajre2021_PsychBERT",
        "PsychBERT mental health language model social media behavioral analysis",
        None,
        [
            "https://arxiv.org/pdf/2110.06512",
        ],
    ),
    (
        "10_Ellouze2024_AI_Personality_Survey",
        "Artificial Intelligence application analysis personality traits disorders social media survey 2024",
        None,
        [],
    ),
    (
        "11_Xu2024_MentalLLM",
        "Mental-LLM leveraging large language models mental health prediction online text data",
        "10.1145/3659604",
        [
            "https://arxiv.org/pdf/2307.04385",
            "https://arxiv.org/pdf/2307.04385v2",
        ],
    ),
    (
        "12_Ellouze2022_Hybrid_PD_Detection",
        "hybrid approach detection monitoring personality disorders social networks 2022",
        "10.1007/s13278-022-00884-x",
        [],
    ),
    (
        "13_Khazbak2021_MindTime_BPD",
        "MindTime deep learning borderline personality disorder detection 2021",
        None,
        [],
    ),
    (
        "14_Martinez2022_BERT_Self_Harm",
        "Early risk detection self-harm BERT transformers",
        None,
        [],
    ),
    (
        "15_Greco2023_Transformer_Mental_Health_Survey",
        "Transformer-based language models mental health issues survey 2023",
        "10.1016/j.patrec.2023.01.027",
        [
            "https://arxiv.org/pdf/2306.09824",
        ],
    ),
    (
        "17_Transformer_Suicidal_Ideation_IEEE",
        "Transformer approach detect suicidal ideation pre-trained language models",
        None,
        [],
    ),
    (
        "18_Mental_Health_QA_BERT_KnowledgeGraph",
        "Mental health question answering system BERT model knowledge graph",
        None,
        [],
    ),
    (
        "19_Deciphering_PD_LSTM_Social_Media",
        "Deciphering personality disorders digital realm deep learning LSTM social media",
        None,
        [],
    ),
]

success = []
failed = []

for fname_base, title, doi, extra_urls in REMAINING:
    pdf_path = LIT_DIR / f"{fname_base}.pdf"
    if pdf_path.exists() and pdf_path.stat().st_size > 5000:
        print(f"  [✓ exists] {fname_base}")
        success.append(fname_base)
        continue

    print(f"\n>>> {fname_base}")
    downloaded = False

    # 1. Try extra/known URLs first
    for url in extra_urls:
        if downloaded:
            break
        print(f"  Trying: {url[:80]}")
        downloaded = download_pdf(url, pdf_path)

    # 2. Try Unpaywall with DOI
    if not downloaded and doi:
        print(f"  Trying Unpaywall (doi={doi})...")
        uw_url = unpaywall_pdf(doi)
        if uw_url:
            print(f"  → {uw_url[:80]}")
            downloaded = download_pdf(uw_url, pdf_path)

    # 3. Try EuropePMC for biology/medicine papers with DOI
    if not downloaded and doi:
        print(f"  Trying EuropePMC...")
        epmc_url = europepmc_pdf(doi)
        if epmc_url:
            print(f"  → {epmc_url[:80]}")
            downloaded = download_pdf(epmc_url, pdf_path)

    # 4. Semantic Scholar search with arxiv ID extraction
    if not downloaded:
        print(f"  Trying Semantic Scholar + ArXiv extraction...")
        s2_url = s2_search_pdf(title)
        if s2_url:
            print(f"  → {s2_url[:80]}")
            downloaded = download_pdf(s2_url, pdf_path)

    if downloaded:
        success.append(fname_base)
    else:
        print(f"  ✗ FAILED")
        failed.append(fname_base)

    time.sleep(1.5)

print("\n" + "=" * 60)
print(f"This pass: {len(success)} success, {len(failed)} failed")
if failed:
    print("Still failed:")
    for f in failed:
        print(f"  {f}")

# Final count in folder
pdfs = list(LIT_DIR.glob("*.pdf"))
print(f"\nTotal PDFs in literature/: {len(pdfs)}")
for p in sorted(pdfs):
    print(f"  {p.name}  ({p.stat().st_size//1024} KB)")
