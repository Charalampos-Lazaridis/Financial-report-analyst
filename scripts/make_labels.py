import json
import sys
from pathlib import Path

RESULTS = "results/golden_2026-11-09_final_48.jsonl"
OUT = Path("results/labels_2026-11-09.json")

if OUT.exists():
    sys.exit(f"{OUT} already exists: not overwriting your labels")

with open("data/golden_set.jsonl", encoding="utf-8") as file:
    tests = [json.loads(line) for line in file if line.strip()]
numbers = {test["id"]: n for n, test in enumerate(tests, start=1)}  # id -> golden-set number

with open(RESULTS, encoding="utf-8") as file:
    rows = [json.loads(line) for line in file if line.strip()]

items = []
for row in rows:
    items.append({
        "n": numbers[row["id"]],
        "label": None,  # fill in: true = correct, false = wrong
        "cause": "",    # if wrong: a few words, e.g. "wrong definition", "misread number"
        "note": "",     # optional
        "company": row["company"],
        "question": row["question"],
        "reference": row["reference"],
        "answer": row["answer"],
        "calculations": [f"{c['expression']} = {c['result']}" for c in row["calculations"]],
        "steps": row["steps"],
    })

with open(OUT, "w", encoding="utf-8") as file:
    json.dump(items, file, indent=2, ensure_ascii=False)

print(f"Wrote {len(items)} items to {OUT}")