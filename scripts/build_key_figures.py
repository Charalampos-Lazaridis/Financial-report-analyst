"""Download key financial figures for our 12 reports from SEC XBRL 
and save them to data/key_figures.json.
The app will only read the saved file; it never calls the SEC.
"""

import json
from datetime import date
from pathlib import Path

import requests
import time

from app import config
from app.ingest import DOCUMENTS
from collections import Counter

OUTPUT = Path("data/key_figures.json")
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{:010d}.json"
REQUEST_PAUSE_SECONDS = 0.2

# SEC company ID (CIK) for each report. The script prints the SEC's company name so a wrong ID shows up.
CIKS = {
    "AAPL_10K_2025": 320193,
    "3M_2022_10K": 66740,
    "AES_2022_10K": 874761,
    "AMCOR_2023_10K": 1748790,
    "AMD_2022_10K": 2488,
    "AMERICANEXPRESS_2022_10K": 4962,
    "BESTBUY_2023_10K": 764478,
    "BOEING_2022_10K": 12927,
    "CVSHEALTH_2022_10K": 64803,
    "JOHNSON_JOHNSON_2022_10K": 200406,
    "PEPSICO_2022_10K": 77476,
    "VERIZON_2022_10K": 732712,
}

# Figure name -> XBRL tags to try, in order. Companies use different tags for the same thing.
FIGURES = {
    "Revenue": ["RevenuesNetOfInterestExpense", "Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "Cost of revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "Operating income": ["OperatingIncomeLoss"],
    "Net income": ["NetIncomeLoss"],
    "Depreciation and amortization": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
                                      "DepreciationAmortizationAndAccretionNet"],
    "Interest expense": ["InterestExpense", "InterestExpenseNonoperating", "InterestAndDebtExpense"],
    "Total assets": ["Assets"],
    "Current assets": ["AssetsCurrent"],
    "Current liabilities": ["LiabilitiesCurrent"],
    "Cash and cash equivalents": ["CashAndCashEquivalentsAtCarryingValue"],
    "Accounts receivable": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent",
                            "AccountsNotesAndLoansReceivableNetCurrent"],
    "Inventory": ["InventoryNet"],
    "Property, plant and equipment (net)": ["PropertyPlantAndEquipmentNet"],
    "Total liabilities": ["Liabilities"],
    "Shareholders' equity": ["StockholdersEquity",
                             "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "Long-term debt": ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"],
    "Operating cash flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "Capital expenditure": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "Dividends paid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
}

def pick_values(entries: list[dict], year: int) -> list[dict]:
    """From one tag's values, pick fiscal `year` and the year before, as reported in that year's 10-K.

    A 10-K shows the current year next to earlier years, so we keep the two latest period ends,
    newest first. The prior one is kept only if it ends about a year earlier.
    Income and cash-flow figures cover a period (start..end): keep only full years, not quarters.
    Balance-sheet figures are a single date (no start).
    """
    by_end = {}
    for e in entries:
        if e.get("form") != "10-K" or e.get("fy") != year or e.get("fp") != "FY":
            continue
        if "start" in e:
            days = (date.fromisoformat(e["end"]) - date.fromisoformat(e["start"])).days
            if not 350 <= days <= 380:
                continue
        by_end.setdefault(e["end"], e)  # one value per period end

    ends = sorted(by_end, reverse=True)  # ISO dates sort correctly as text
    if len(ends) >= 2:
        gap = (date.fromisoformat(ends[0]) - date.fromisoformat(ends[1])).days
        if 350 <= gap <= 380:
            return [by_end[ends[0]], by_end[ends[1]]]
    return [by_end[ends[0]]] if ends else []


def main() -> None:
    if not config.SEC_USER_AGENT:
        raise RuntimeError("Missing SEC_USER_AGENT in .env (e.g. SEC_USER_AGENT=Name email).")

    headers = {"User-Agent": config.SEC_USER_AGENT}
    result = {}
    for doc_name, company, year in DOCUMENTS:
        time.sleep(REQUEST_PAUSE_SECONDS)
        response = requests.get(FACTS_URL.format(CIKS[doc_name]), headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        us_gaap = data["facts"].get("us-gaap", {})
        print(f"\n{doc_name}  (SEC name: {data['entityName']})")

        figures, missing = {}, []
        for name, tags in FIGURES.items():
            for tag in tags:
                entries = us_gaap.get(tag, {}).get("units", {}).get("USD", [])
                chosen = pick_values(entries, year)
                if chosen:
                    current = chosen[0]
                    prior = chosen[1] if len(chosen) > 1 else None
                    figures[name] = {
                        "value_musd": round(current["val"] / 1_000_000, 1),  # millions of USD
                        "period_end": current["end"],
                        "prior_musd": round(prior["val"] / 1_000_000, 1) if prior else None,
                        "prior_end": prior["end"] if prior else None,
                        "tag": tag,
                    }
                    prior_text = f"{figures[name]['prior_musd']:>14,.1f}" if prior else f"{'-':>14}"
                    print(f"  {name:38} {figures[name]['value_musd']:>14,.1f} {prior_text}"
                          f"  {current['end']}  {tag}")
                    break
            else:  # no tag found a value
                missing.append(name)

        period_end = Counter(f["period_end"] for f in figures.values()).most_common(1)[0][0]
        for name in [n for n, f in figures.items() if f["period_end"] != period_end]:
            print(f"  DROPPED (old period {figures[name]['period_end']}): {name}")
            del figures[name]
            missing.append(name)

        if missing:
            print(f"  MISSING: {', '.join(missing)}")
        result[doc_name] = {"company": company, "fiscal_year": year, "period_end": period_end,
                            "figures": figures, "missing": missing}

    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()