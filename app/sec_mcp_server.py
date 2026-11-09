"""MCP server with one tool: look up a figure in a company's SEC XBRL data, 
  for any fiscal year.
"""
 
from functools import lru_cache

import requests
from fastmcp import FastMCP

from app import config
from app.ingest import DOCUMENTS
from scripts.build_key_figures import CIKS, FACTS_URL, FIGURES, pick_values

CIK_BY_COMPANY = {company: CIKS[doc_name] for doc_name, company, _ in DOCUMENTS}

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
    figure: a common name (Revenue, Net income, Inventory, Capital  expenditure, ...) or an
    XBRL us-gaap tag such as ResearchAndDevelopmentExpense or  EarningsPerShareDiluted.
    fiscal_year: the fiscal year the figure belongs to, e.g. 2021.
    Money values are in millions of USD; per-share values in USD.
    """
    if company not in CIK_BY_COMPANY:
        raise ValueError(f"Unknown company {company!r}. Use one of: {', '.join(CIK_BY_COMPANY)}")
    facts = company_facts(CIK_BY_COMPANY[company])

    # A common name maps to several tags. Anything else: try it as written, and as a tag built from its words
    # ("Research and development expense" -> "ResearchAndDevelopmentExpense").
    
    tags = FIGURES.get(figure) or [figure, "".join(word.capitalize() for word in figure.split())]

    for tag in tags:
        units = facts.get(tag, {}).get("units", {})
        for unit, scale, unit_name in (("USD", 1_000_000, "millions of USD"), ("USD/shares", 1, "USD per share")):
            chosen = pick_values(units.get(unit, []), fiscal_year)
            if chosen:
                entry = chosen[0]  # the fiscal year itself, not the prior year
                return {"company": company, "figure": figure, "tag": tag, "fiscal_year": fiscal_year,
                        "value": round(entry["val"] / scale, 2), "unit": unit_name,
                        "period_end": entry["end"], "filed": entry["filed"]}
        
    raise ValueError(f"No {figure!r} found for {company} in its FY{fiscal_year} 10-K. "
                    f"Common names: {', '.join(FIGURES)}")

if __name__ == "__main__":
    mcp.run()  # stdio: the client starts this file and talks to it through standard input/output