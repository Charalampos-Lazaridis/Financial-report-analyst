"""Retrieval: find the excerpts for a question (similar chunks, the company's statements, SEC key figures)."""

import json
from collections import Counter
from pathlib import Path

from qdrant_client import models

from app.ingest import COLLECTION, DOCUMENTS, embed, get_qdrant

STATEMENT_K = 3  # chunks from each financial statement
DOC_NAMES = {company: doc_name for doc_name, company, _ in DOCUMENTS}

TOP_K = 11  # how many chunks the model gets per question
STATEMENT_QUERIES = [
    "consolidated balance sheets total current assets inventories accounts receivable total current liabilities total assets",
    "consolidated statements of operations income revenue cost of sales operating income net income",
    "consolidated statements of cash flows depreciation and amortization capital expenditures net cash provided by operating activities",
]

KEY_FIGURES = json.loads(Path("data/key_figures.json").read_text(encoding="utf-8"))

# Done once, when the server starts, not on every question
qdrant = get_qdrant()

def retrieve(question: str, limit: int = TOP_K, company: str | None = None) -> list[dict]:
    """Return the chunks closest in meaning to the question, from one company or from all reports."""
    vector = embed([question])[0]
    query_filter = None
    if company:
        query_filter = models.Filter(must=[
            models.FieldCondition(key="doc_name", match=models.MatchValue(value=DOC_NAMES[company])),
        ])
    hits = qdrant.query_points(COLLECTION, query=vector, limit=limit, query_filter=query_filter).points
    return [{**hit.payload, "score": hit.score} for hit in hits]


def format_excerpts(chunks: list[dict], start: int = 1) -> str:
    """Number the chunks like footnotes, each with its company, year and page (or SEC key figures)."""
    parts = []
    for i, chunk in enumerate(chunks, start=start):
        where = "SEC XBRL key figures" if chunk["page"] is None else f"page {chunk['page']}"
        parts.append(f"[{i}] {chunk['company']} {chunk['year']}, {where}\n{chunk['text']}")
    return "\n\n".join(parts)


def format_musd(value: float) -> str:
    """Write a number the way a 10-K does: 86392.0 -> "86,392", -3547.0 -> "(3,547)"."""
    text = f"{abs(value):,.1f}".removesuffix(".0")
    return f"({text})" if value < 0 else text


def key_figures_chunk(company: str) -> dict | None:
    """The company's SEC XBRL key figures (this year and the year before) as one extra excerpt."""
    doc_name = DOC_NAMES[company]
    report = KEY_FIGURES.get(doc_name)
    if not report:
        return None
    year = report["fiscal_year"]
    lines = [f"Key figures from SEC XBRL data for the {company} 10-K, in millions of USD, for fiscal year {year} "
             f"(period ending {report['period_end']}) and the year before. Negative values in parentheses."]
    for name, figure in report["figures"].items():
        line = f"{name}: FY{year} {format_musd(figure['value_musd'])}"
        if figure.get("prior_musd") is not None:
            line += f", FY{year - 1} {format_musd(figure['prior_musd'])}"
        lines.append(f"{line}  [XBRL tag {figure['tag']}]")
    return {"doc_name": doc_name, "company": company, "year": year,
            "page": None, "chunk": "sec", "text": "\n".join(lines), "score": 0.0}


def retrieve_context(question: str) -> list[dict]:
    """Return the chunks closest to the question, plus the main financial statements of its company."""
    top = retrieve(question, limit=20)
    if not top:
        return []
    company = Counter(chunk["company"] for chunk in top).most_common(1)[0][0]  # the most common company

    chunks = top[:TOP_K]
    seen = {(chunk["doc_name"], chunk["page"], chunk["chunk"]) for chunk in chunks}
    for query in STATEMENT_QUERIES:
        for chunk in retrieve(query, limit=STATEMENT_K, company=company):
            key = (chunk["doc_name"], chunk["page"], chunk["chunk"])
            if key not in seen:
                seen.add(key)
                chunks.append(chunk)
    sec = key_figures_chunk(company)
    if sec:
        chunks.append(sec)  # last, so the page excerpts keep their numbers
    return chunks
