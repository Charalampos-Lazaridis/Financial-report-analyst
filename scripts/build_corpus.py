import json
import re
from collections import Counter
from pathlib import Path

import requests
from pypdf import PdfReader

DATA = Path("data")
FILINGS = DATA / "filings"
PDF_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/{}.pdf"

CORPUS = [
    "AMD_2022_10K", "AMERICANEXPRESS_2022_10K", "BOEING_2022_10K", "PEPSICO_2022_10K",
    "3M_2022_10K", "AES_2022_10K", "CVSHEALTH_2022_10K", "JOHNSON_JOHNSON_2022_10K",
    "VERIZON_2022_10K", "AMCOR_2023_10K", "BESTBUY_2023_10K",
]

def download_pdfs():
    FILINGS.mkdir(parents=True, exist_ok=True)
    for name in CORPUS:
        path = FILINGS / f"{name}.pdf"
        if path.exists():
            print(f"skip {name} (already downloaded)")
            continue
        response = requests.get(PDF_URL.format(name), timeout=60)
        response.raise_for_status()  # stop with an error on 404 etc.
        path.write_bytes(response.content)
        print(f"downloaded {name}: {len(response.content) / 1e6:.1f} MB")


def build_golden_set():
    golden = []
    with open(DATA / "financebench_open_source.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["doc_name"] not in CORPUS:
                continue
            golden.append({
                "id": row["financebench_id"],
                "doc_name": row["doc_name"],
                "company": row["company"],
                "question_type": row["question_type"],
                "question": row["question"],
                "answer": row["answer"],
                "justification": row["justification"],
                "evidence": [
                    {"page": e["evidence_page_num"], "text": e["evidence_text"]}
                    for e in row["evidence"]
                ],
            })
    with open(DATA / "golden_set.jsonl", "w", encoding="utf-8") as f:
        for item in golden:
            f.write(json.dumps(item) + "\n")
    print(f"golden set: {len(golden)} questions -> data/golden_set.jsonl")
    return golden


def words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def overlap(evidence, page_text):
    """Share of the evidence's words that also appear on the page (0.0 to 1.0)."""
    ev = words(evidence)
    return len(ev & words(page_text)) / len(ev) if ev else 0.0


def check_pages(golden):
    readers = {}
    shifts = Counter()
    for item in golden:
        name = item["doc_name"]
        if name not in readers:
            readers[name] = PdfReader(FILINGS / f"{name}.pdf")
        pages = readers[name].pages
        for ev in item["evidence"]:
            p = ev["page"]
            scores = {
                shift: overlap(ev["text"], pages[p + shift].extract_text())
                for shift in (-1, 0, 1)
                if 0 <= p + shift < len(pages)
            }
            if not scores:
                print(f"  {item['id']} {name}: page {p} is outside the PDF")
                continue
            best = max(scores, key=scores.get)
            shifts[best] += 1
            if best != 0 or scores[best] < 0.8:
                print(f"  {item['id']} {name} page {p}: best shift {best:+d}, overlap {scores[best]:.0%}")
    print(f"page shifts: {dict(shifts)}  (0 = FinanceBench page == pypdf page index)")


if __name__ == "__main__":
    download_pdfs()
    golden = build_golden_set()
    check_pages(golden)