"""MCP server with one tool: look up a figure in a company's SEC XBRL data, 
  for any fiscal year.
"""
 
import re
from functools import lru_cache

import requests
from fastmcp import FastMCP

from app import config
from app.ingest import DOCUMENTS
from scripts.build_key_figures import CIKS, FACTS_URL, FIGURES, pick_values

CIK_BY_COMPANY = {company: CIKS[doc_name] for doc_name, company, _ in DOCUMENTS}

FIGURES_LOWER = {name.lower(): name for name in FIGURES}
FILLER_WORDS = {"of", "and", "the", "in", "for", "to", "net", "total"}  # too common in labels to count as a match


def words(text: str) -> set[str]:
    """'Short-term Investments' -> {'short', 'term', 'investment'}: lower case, no punctuation, singular."""
    result = set()
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in FILLER_WORDS:
            continue
        if word.endswith("ies"):
            word = word[:-3] + "y"        # inventories -> inventory
        result.add(word.removesuffix("s"))  # investments -> investment
    return result


def matching_tags(facts: dict, figure: str, limit: int = 10, min_score: float = 0.5) -> list[str]:
    """The company's XBRL tags whose label is closest to the figure name, best first.

    Score = shared words / all words (1.0 = same words). Below min_score the match is too weak to trust:
    'Cost of Sales' would otherwise return 'Variable Lease, Cost'. Better an error than a wrong number."""
    wanted = words(figure)
    scored = []
    for tag, fact in facts.items():
        label = words(fact.get("label") or "")
        if wanted & label:
            score = len(wanted & label) / len(wanted | label)
            if score >= min_score:
                scored.append((score, tag))
    scored.sort(reverse=True)
    return [tag for _, tag in scored[:limit]]


mcp = FastMCP("SEC financial facts")

@lru_cache(maxsize=3)  # a company's file is several MB: keep only the last 3 in memory
def company_facts(cik: int) -> dict:
    """Download every XBRL fact the company has filed (all years, all tags)."""
    if not config.SEC_USER_AGENT:
        raise RuntimeError("SEC_USER_AGENT is not set")
    response = requests.get(FACTS_URL.format(cik), headers={"User-Agent": config.SEC_USER_AGENT}, timeout=30)
    response.raise_for_status()
    return response.json()["facts"].get("us-gaap", {})


@mcp.tool()
def get_sec_fact(company: str, figure: str, fiscal_year: int) -> dict:
    """Look up one figure from a company's 10-K in SEC XBRL data, for any 
    fiscal year.

    company: one of Apple, 3M, AES Corporation, Amcor, AMD, American Express, 
    Best Buy, Boeing, CVS Health, Johnson & Johnson, PepsiCo, Verizon.
    figure: a common name (Revenue, Net income, Inventory, Capital expenditure, ...), any line-item name
    as the 10-K writes it (matched to the closest XBRL label, returned as 'label'), or an
    XBRL us-gaap tag such as ResearchAndDevelopmentExpense or EarningsPerShareDiluted.
    fiscal_year: the fiscal year the figure belongs to, e.g. 2021.
    Money values are in millions of USD; per-share values in USD.
    """
    if company not in CIK_BY_COMPANY:
        raise ValueError(f"Unknown company {company!r}. Use one of: {', '.join(CIK_BY_COMPANY)}")
    facts = company_facts(CIK_BY_COMPANY[company])

    # One of our 19 checked figures -> its tags; an exact XBRL tag -> itself;
    # anything else -> the tags whose labels are closest ("Inventories" -> InventoryNet, "Inventory, Net").
    known = FIGURES_LOWER.get(figure.strip().lower())
    if known:
        tags = FIGURES[known]
    elif figure in facts:
        tags = [figure]
    else:
        tags = matching_tags(facts, figure)

    for tag in tags:
        units = facts.get(tag, {}).get("units", {})
        for unit, scale, unit_name in (("USD", 1_000_000, "millions of USD"), ("USD/shares", 1, "USD per share")):
            chosen = pick_values(units.get(unit, []), fiscal_year)
            if chosen:
                entry = chosen[0]  # the fiscal year itself, not the prior year
                return {"company": company, "figure": figure, "tag": tag, "label": facts[tag].get("label"),
                        "fiscal_year": fiscal_year,
                        "value": round(entry["val"] / scale, 2), "unit": unit_name,
                        "period_end": entry["end"], "filed": entry["filed"]}
        
    closest = [facts[tag].get("label") for tag in matching_tags(facts, figure, limit=5, min_score=0)]
    raise ValueError(f"No {figure!r} found for {company} in its FY{fiscal_year} 10-K. "
                     f"Common names: {', '.join(FIGURES)}. "
                     f"Closest labels this company files: {'; '.join(closest) or 'none'}")

if __name__ == "__main__":
    mcp.run()  # stdio: the client starts this file and talks to it through standard input/output