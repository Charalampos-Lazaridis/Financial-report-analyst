# Cost experiment: whole 10-K vs. relevant section, small vs. large model

**Date:** 2026-10-31
**Script:** `scripts/cost_experiment.py` (run with `python -m scripts.cost_experiment`)
**Document:** Apple 10-K for fiscal 2025 (`data/filings/AAPL_10K_2025.pdf`, 78 pages), text extracted with plain `pypdf`, no cleanup

## Why

Until now, every question sent the whole 10-K to the model (~49.6k input tokens), even though the answer sits in a page or two. This experiment measures what that costs, and whether a bigger model is worth paying for. If sending only the relevant section gives the same answers for much less, that is the case for RAG (Retrieval-Augmented Generation): let code retrieve the relevant parts and send only those.

## Setup

- **2 contexts** (the text the model answers from):
  - *whole 10-K*: all 78 pages (~216k characters)
  - *MD&A only*: Item 7, Management's Discussion and Analysis (pages 29–37 counting from 0, 9 pages). Picked by hand because it contains all three answers
- **2 models:** `gpt-4o-mini` and `gpt-4o` (OpenAI Responses API, structured output)
- **3 lookup questions**, same prompt (`INSTRUCTIONS` and `Answer` schema from `app/analyst.py`) for every call:

| Question | Expected | Where in the report |
|---|---|---|
| What were Apple's total net sales in fiscal 2025? | $416,161 million | MD&A, net sales table |
| By what percentage did Greater China net sales change in 2025? | −4% (written `(4)%`) | MD&A, net sales by segment |
| What was Apple's research and development expense in fiscal 2025? | $34,550 million | MD&A, operating expenses |

- **Prices used** (US$ per 1M tokens, input / output): `gpt-4o-mini` 0.15 / 0.60, `gpt-4o` 2.50 / 10.00
- **"correct"** = the expected text (`416,161`, `4%`, `34,550`) appears in the answer. Rough, but every answer was also checked by eye.

## Results

| model | context | input tokens | cached | output tokens | cost $ | secs | correct |
|---|---|---:|---:|---:|---:|---:|---|
| gpt-4o-mini | whole 10-K | 49,618 | 0 | 70 | 0.0075 | 4.6 | True |
| gpt-4o-mini | whole 10-K | 49,620 | 0 | 83 | 0.0075 | 9.0 | True |
| gpt-4o-mini | whole 10-K | 49,619 | 0 | 72 | 0.0075 | 5.0 | True |
| gpt-4o-mini | MD&A only | 4,833 | 0 | 64 | 0.0008 | 3.6 | True |
| gpt-4o-mini | MD&A only | 4,835 | 0 | 80 | 0.0008 | 2.0 | True |
| gpt-4o-mini | MD&A only | 4,834 | 0 | 82 | 0.0008 | 1.7 | True |
| gpt-4o | whole 10-K | 49,618 | 0 | 97 | 0.1250 | 3.4 | True |
| gpt-4o | whole 10-K | 49,620 | 0 | 99 | 0.1250 | 4.4 | True |
| gpt-4o | whole 10-K | 49,619 | 0 | 71 | 0.1248 | 4.0 | True |
| gpt-4o | MD&A only | 4,833 | 0 | 67 | 0.0128 | 1.6 | True |
| gpt-4o | MD&A only | 4,835 | 0 | 72 | 0.0128 | 1.7 | True |
| gpt-4o | MD&A only | 4,834 | 0 | 94 | 0.0130 | 1.6 | True |

Rows are in question order (net sales, Greater China, R&D) within each model/context group.

### Summary per question

| setup | input tokens | cost | time | correct |
|---|---:|---:|---:|---|
| gpt-4o-mini, whole 10-K | ~49.6k | $0.0075 | 4.6–9.0 s | 3/3 |
| gpt-4o-mini, MD&A only | ~4.8k | $0.0008 | 1.7–3.6 s | 3/3 |
| gpt-4o, whole 10-K | ~49.6k | $0.125 | 3.4–4.4 s | 3/3 |
| gpt-4o, MD&A only | ~4.8k | $0.013 | 1.6–1.7 s | 3/3 |

Total cost of the run: about $0.44.

## Takeaways

1. **Only the relevant section: ~10× fewer tokens, ~10× cheaper, faster, just as correct.** The other ~90% of the report added cost and waiting time without improving anything. This motivates RAG (Unit 2).
2. **The large model cost ~17× more and gained nothing on these questions.** For lookups (find a number in a table), `gpt-4o-mini` is enough. We keep it as the default.
3. **Best vs. worst setup:** mini + section ($0.0008) is ~150× cheaper than 4o + whole 10-K ($0.125), with the same result.

## Caveats

- **Tiny, easy sample.** 3 lookup questions, one run each. Harder questions (calculations across tables, "explain why margins changed", questions the report can't answer) may separate the models. The Step 8 test set should include those.
- **The section was hand-picked.** We knew where the answers were. RAG has to find the right section by itself, and a wrong pick means a wrong or missing answer. That is the new risk RAG brings.
- **Timing is noisy.** The same setup took 9.0 s on one run and 5.0 s on the next. Treat the times as rough.
- **Prompt caching didn't happen** (`cached` = 0 everywhere), although questions 2 and 3 had the same long prefix as question 1. OpenAI's caching is best-effort; requests may reach different servers. `prompt_cache_key` might improve hit rates (not tested). The cost column assumes no caching discount.
- **Prices change.** The costs use the prices above. Check https://openai.com/api/pricing before comparing with a later run.

## Next

Rerun this after RAG is in place (Unit 2): same questions, with retrieved chunks instead of a hand-picked section, to see whether retrieval keeps the accuracy at the lower cost.
