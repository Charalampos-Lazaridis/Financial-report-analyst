"""Run the 48 FinanceBench golden questions through ask() and grade retrieval and answers."""

import json
from datetime import date

from openai import OpenAI
from pydantic import BaseModel

from app import config
from app.analyst import FALLBACK_ANSWER, PRICES, TOP_K, ask, qdrant

RUN_NAME = "k20"  # change this for each experiment so results files don't overwrite each other
REJUDGE_FROM = None  # None = run ask() again; a path = only re-grade those saved answers
JUDGE_MODEL = "gpt-4o-mini"
JUDGE_INSTRUCTIONS = (
    "You grade answers to questions about companies' 10-K reports. "
    "Compare the candidate answer with the reference answer. Judge meaning, not wording.\n"
    "Correct if the candidate reaches the same conclusion and its key numbers match the reference:\n"
    "- Small rounding differences and different units for the same amount "
    "(e.g. $1.5 billion vs $1,500 million) are fine.\n"
    "- Extra detail, context or numbers never make an answer wrong, unless they contradict the reference.\n"
    "- If the reference says a metric is not meaningful for this company, or that an amount is zero / "
    "none / not outlined, a candidate that says the same thing in other words is correct.\n"
    "- For yes/no questions, the yes or no must match.\n"
    "Incorrect if a key number or the conclusion differs or is missing, "
    "or if the candidate says the information isn't available while the reference gives an answer."
)

class Verdict(BaseModel):
    correct: bool
    reason: str

judge_client = OpenAI(api_key=config.OPENAI_API_KEY, max_retries=3, timeout=45)

def judge(question: str, reference: str, candidate: str) -> tuple[Verdict, float]:
    """Ask the judge model whether the candidate answer matches the reference. Returns (verdict, cost)."""
    response = judge_client.responses.parse(
        model=JUDGE_MODEL,
        instructions=JUDGE_INSTRUCTIONS,
        input=f"Question: {question}\n\nReference answer: {reference}\n\nCandidate answer: {candidate}",
        text_format=Verdict,
    )
    price_in, price_out = PRICES[JUDGE_MODEL]
    usage = response.usage
    cost = (usage.input_tokens * price_in + usage.output_tokens * price_out) / 1_000_000
    verdict = response.output_parsed or Verdict(correct=False, reason="judge gave no verdict")
    return verdict, cost

def first_hit_rank(test: dict, sources) -> int | None:
    """Rank (1 = best) of the first retrieved excerpt that is an evidence page, or None if none is."""
    evidence_pages = {item["page"] for item in test["evidence"]}
    for source in sources:
        if source.company == test["company"] and source.page in evidence_pages:
            return source.id
    return None

with open("data/golden_set.jsonl", encoding="utf-8") as file:
    tests = [json.loads(line) for line in file if line.strip()]

if REJUDGE_FROM:
    # Re-grade saved answers only: the answers stay fixed, so any score change comes from the judge
    with open(REJUDGE_FROM, encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    for n, row in enumerate(rows, start=1):
        verdict, judge_cost = judge(row["question"], row["reference"], row["answer"])
        flipped = "  <- CHANGED" if verdict.correct != row["correct"] else ""
        row["old_correct"] = row["correct"]
        row["correct"] = verdict.correct
        row["judge_reason"] = verdict.reason
        row["judge_cost_usd"] = round(judge_cost, 6)
        print(f"{n:>2}/{len(rows)}  {'CORRECT' if verdict.correct else 'WRONG  '}{flipped}  "
              f"{row['company']}: {row['question'][:60]}")
else:
    rows = []
    for n, test in enumerate(tests, start=1):
        result = ask(test["question"])
        rank = first_hit_rank(test, result.sources)
        verdict, judge_cost = judge(test["question"], test["answer"], result.answer)

        rows.append({
            "id": test["id"],
            "company": test["company"],
            "question_type": test["question_type"],
            "question": test["question"],
            "reference": test["answer"],
            "evidence_pages": [item["page"] for item in test["evidence"]],
            "answer": result.answer,
            "sources": [source.model_dump() for source in result.sources],
            "hit_rank": rank,
            "correct": verdict.correct,
            "judge_reason": verdict.reason,
            "fallback": result is FALLBACK_ANSWER,
            "cost_usd": result.cost_usd,
            "judge_cost_usd": round(judge_cost, 6),
        })
        print(f"{n:>2}/{len(tests)}  {'CORRECT' if verdict.correct else 'WRONG  '}  "
              f"hit={'-' if rank is None else rank}  {test['company']}: {test['question'][:70]}")

qdrant.close()

out_path = f"results/golden_{date.today()}_{RUN_NAME}.jsonl"
with open(out_path, "w", encoding="utf-8") as file:
    for row in rows:
        file.write(json.dumps(row) + "\n")

# Summary: sort every question into one of four boxes
hit_correct = sum(1 for r in rows if r["hit_rank"] and r["correct"])
hit_wrong = sum(1 for r in rows if r["hit_rank"] and not r["correct"])
miss_correct = sum(1 for r in rows if not r["hit_rank"] and r["correct"])
miss_wrong = sum(1 for r in rows if not r["hit_rank"] and not r["correct"])
total = len(rows)

print(f"\nAnswer accuracy:     {hit_correct + miss_correct}/{total}")
print(f"Retrieval hit rate:  {hit_correct + hit_wrong}/{total}  (an evidence page in the top {TOP_K})")
print(f"  evidence retrieved, answer correct:  {hit_correct}")
print(f"  evidence retrieved, answer wrong:    {hit_wrong}   <- model ignored/misread the right chunk")
print(f"  evidence missed,    answer correct:  {miss_correct}   <- found it on another page")
print(f"  evidence missed,    answer wrong:    {miss_wrong}   <- retrieval failure")
print(f"Fallbacks: {sum(r['fallback'] for r in rows)}")
if REJUDGE_FROM:
    print(f"Re-judged {REJUDGE_FROM}: {sum(r['correct'] != r['old_correct'] for r in rows)} verdicts changed "
          f"(answer cost below is from the original run, not spent again)")
print(f"Cost: answers ${sum(r['cost_usd'] for r in rows):.4f} + judge ${sum(r['judge_cost_usd'] for r in rows):.4f}")
print(f"Saved to {out_path}")
