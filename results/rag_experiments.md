# RAG experiments (Unit 2)

One lever changed per run. Each run is scored on the 48-question FinanceBench golden set by `scripts/eval_golden.py` and saved as `results/golden_<date>_<run>.jsonl`.

- **Accuracy**: share of answers the judge marks correct.
- **Hit rate**: share of questions where an evidence page is among the retrieved chunks.

## Run log

| Run | Date | Change vs. previous | Accuracy | Hit rate | Hit+correct | Hit+wrong | Miss+correct | Miss+wrong | Cost (answers + judge) |
|---|---|---|---|---|---|---|---|---|---|
| `naive_k5` | 2026-11-02 | Baseline | 9/48 | 21/48 | 7 | 14 | 2 | 25 | $0.021 + $0.004 |
| `naive_k5_judge2` | 2026-11-02 | New judge prompt, same answers (re-graded only) | 14/48 (≈13/48, see #42) | 21/48 | 9 | 12 | 5 | 22 | $0 + $0.0042 |
| `prompt_v2` | 2026-11-03 | New answer prompt (compute, fewer refusals) | 12/48 | 21/48 | 7 | 14 | 5 | 22 | $0.0213 + $0.0037 |
| `prompt_v3` | 2026-11-03 | Answer prompt: support in 2–4 sentences, special case only when shown, no example phrases. **Frozen** | 14/48 | 20/48 | 6 | 14 | 8 | 20 | $0.0218 + $0.0037 |
| `k10` | 2026-11-03 | `TOP_K` 5 → 10 | 15/48 | 25/48 | 10 | 15 | 5 | 18 | $0.0368 + $0.0038 |
| `k20` | 2026-11-03 | `TOP_K` 10 → 20 | 17/48 | 34/48 | 11 | 23 | 6 | 8 | $0.0652 + $0.0038 |

## `naive_k5`: baseline setup

- Corpus: 12 10-Ks (11 FinanceBench + Apple 2025), pypdf page text, no cleanup
- Chunking: 2,000 chars, 200 overlap, never crossing pages → 5,418 chunks
- Embeddings: `text-embedding-3-small` (1536, cosine), local Qdrant
- Retrieval: top 5 chunks from all reports, no filtering
- Answer model: `gpt-4o-mini`, structured output
- Judge: `gpt-4o-mini`, `Verdict{correct, reason}`

Answer prompt (`INSTRUCTIONS` in `app/analyst.py`):

> You are a financial analyst. Answer the question using only the numbered excerpts from 10-K reports. Each excerpt is labelled with its company, fiscal year and page. If the excerpts don't contain the answer, say that the reports don't contain it. Don't guess. Cite the excerpts you use in the answer, like [2]. List every number you use in figures, with source set to the number of the excerpt it comes from. Copy each value exactly as the report writes it (e.g. in millions if the table is in millions). Set is_advice_request to true if the user asks for investment advice.

Judge prompt (`JUDGE_INSTRUCTIONS` in `scripts/eval_golden.py`):

> You grade answers to questions about companies' 10-K reports. Compare the candidate answer with the reference answer. It is correct if it gives the same key facts, numbers and conclusion as the reference. Small rounding differences, different units for the same amount (e.g. $1.5 billion vs $1,500 million) and extra detail are fine. It is incorrect if a key number or conclusion differs or is missing, or if the candidate says the information isn't available.

Findings:

- 25 of the 39 wrong answers are refusals ("the reports do not contain…"), 7 of them with the evidence retrieved. The model doesn't compute ratios (quick ratio, ROA, inventory turnover), and it refuses on "if X isn't a useful metric, say so" questions.
- Every question had at least one chunk from the right company. Only 2 had 3 or more from other companies. Misses are wrong pages within the right report, so company filtering has little room to help.
- By type: domain-relevant 6/37, novel 3/8, metrics 0/3.
- The judge has some false negatives (AES restructuring "0" with the same conclusion; Boeing legal penalised for extra detail). Fix the judge before comparing runs.

## `naive_k5_judge2`: judge fix

Same answers as `naive_k5` (no new `ask()` calls). Only the judge prompt changed, using `REJUDGE_FROM` in `scripts/eval_golden.py`. The hit rate stays 21/48 because retrieval wasn't rerun.

New judge prompt (`JUDGE_INSTRUCTIONS`; used for all later runs):

> You grade answers to questions about companies' 10-K reports. Compare the candidate answer with the reference answer. Judge meaning, not wording.
> Correct if the candidate reaches the same conclusion and its key numbers match the reference:
> - Small rounding differences and different units for the same amount (e.g. $1.5 billion vs $1,500 million) are fine.
> - Extra detail, context or numbers never make an answer wrong, unless they contradict the reference.
> - If the reference says a metric is not meaningful for this company, or that an amount is zero / none / not outlined, a candidate that says the same thing in other words is correct.
> - For yes/no questions, the yes or no must match.
> Incorrect if a key number or the conclusion differs or is missing, or if the candidate says the information isn't available while the reference gives an answer.

Findings:

- 5 verdicts flipped from wrong to correct and none the other way. All 5 were checked by hand: AES restructuring "0", AmEx gross margin, Best Buy acquisitions and Boeing legal are fair.
- **PepsiCo legal (#42) is a false positive**: the answer only says "the reports do not contain" it, while the reference says "No". So the true score is ≈13/48. We kept the judge as is (user's choice) so later runs stay comparable.
- Only the 5 flipped answers were read by hand, not all 48.
- Of the 12 hit+wrong, 7 are refusals with the evidence in hand (a prompt problem). Of the 22 miss+wrong, 15 are refusals (a retrieval problem).

## `prompt_v2`: answer prompt fix

Same retrieval (top 5) and same judge as `naive_k5_judge2`. Only `INSTRUCTIONS` in `app/analyst.py` changed:

> You are a financial analyst. Answer the question using only the numbered excerpts from 10-K reports. Each excerpt is labelled with its company, fiscal year and page.
> - Start with a direct conclusion (yes/no, the number, or the main drivers), then the support.
> - If the question needs a calculation (ratio, margin, turnover, growth, year-over-year change), do it from the numbers in the excerpts and show the formula and the inputs.
> - If the question says what to answer in a special case (e.g. "if not outlined, state 0", "if gross margin isn't a useful metric, say so and explain why"), follow that instruction.
> - Say that the reports don't contain the answer only if the excerpts have nothing relevant. Never make up a number that isn't in the excerpts.
> Cite the excerpts you use in the answer, like [2]. List every number you use in figures, with source set to the number of the excerpt it comes from. Copy each value exactly as the report writes it (e.g. in millions if the table is in millions). Set is_advice_request to true if the user asks for investment advice.

Results: golden set **12/48** (was 14/48), Apple eval **10/11** (was 11/11).

Findings:

- **Refusals fell from 24 to 3** (answers matching "do not contain / not provided / not available…"). Of the 7 hit+refusal answers, only 1 still refuses. The refusal rule worked.
- **Answers got too short**: median length 94 characters (was 321), 9 answers under 40 characters (e.g. just "Yes"). The model gave the conclusion and dropped the support. This lost 4 previously correct answers: #29 Boeing legal (no crashes named), #37 CVS dividends (no $0.55), #33 Boeing production rates (no aircraft named), #22 AmEx tax rate (change given without the 24.6% / 21.6% rates).
- **The special-case examples became the new refusal**: "not a useful/relevant metric" for the quick ratio (#7 Amcor, #11 AMD, #46 Verizon) and JnJ inventory turnover (#40); "state 0" for PepsiCo EBITDA less capex (#44) and AES inventory turnover (#5). Few of the requested calculations actually happened. The one ROA attempt (#6) is wrong (−3.57% vs −2%).
- **Wrong answers instead of refusals**: #18 calls AmEx common shares "debt securities" (reference: none); #30 says Boeing's gross margin didn't improve (it did, 4.8% → 5.3%); #15 says investing activities beat operating activities (a plain error).
- Gained 3: #10 Amcor gross margin, #14 AMD operating margin (Xilinx amortization), #38 JnJ growth. Lost 5 (above).
- **Apple eval, net income per day**: the model set up 112,010 ÷ 365 but answered $306.01 (correct: 306.88), with confidence "high" because the result isn't in `figures`. Asking for calculations makes the model do more mental arithmetic, and that arithmetic is unreliable. The fix is a calculator tool (Unit 3), not the prompt.
- gpt-4o-mini isn't deterministic, so a 2-point gap between runs is partly noise. The patterns above are clear, though.

## `prompt_v3`: final answer prompt (frozen)

Same retrieval and judge. `INSTRUCTIONS` in `app/analyst.py`:

> You are a financial analyst. Answer the question using only the numbered excerpts from 10-K reports. Each excerpt is labelled with its company, fiscal year and page.
> - Start with a direct conclusion (yes/no, the number, or the main drivers), then support it in 2-4 sentences with the specific numbers, names and reasons from the excerpts.
> - If the question needs a calculation (ratio, margin, turnover, growth, year-over-year change), find the inputs in the excerpts, do the calculation, and show the formula and the inputs.
> - Some questions say what to answer in a special case (for example, if a metric doesn't apply). Use that special answer only when the excerpts clearly show that case. If the inputs for the calculation are in the excerpts, calculate instead.
> - Say that the reports don't contain the answer only if the excerpts have nothing relevant. Never make up a number that isn't in the excerpts.
> Cite the excerpts you use in the answer, like [2]. List every number you use in figures, with source set to the number of the excerpt it comes from. Copy each value exactly as the report writes it (e.g. in millions if the table is in millions). Set is_advice_request to true if the user asks for investment advice.

Results: golden set **14/48** (v2 12/48, `naive_k5_judge2` 14/48), Apple eval **10/11** (per-day: 306.74 vs 306.88, now with confidence "low").

| | `naive_k5_judge2` | `prompt_v2` | `prompt_v3` |
|---|---|---|---|
| Correct | 14 | 12 | 14 |
| Refusals | 24 | 3 | 2 |
| "Not a useful metric" / "state 0" excuses | 0 | 6 | 5 |
| Median answer length (chars) | 321 | 94 | 101 |

Findings:

- **Decision: freeze `prompt_v3`** (it beats v2 under the freeze rule). It's the fixed prompt for all retrieval runs. Same score as the old prompt, but the mix differs: refusals are almost gone, and the correct answers come from different questions. Gained vs `naive_k5_judge2`: #10, #14, #20, #27, #38, #47. Lost: #12, #15, #22, #29, #33, #37.
- **"2–4 sentences" didn't work**: answers are still short (median 101). #29 Boeing legal and #37 CVS dividends still lack the detail. The support may be going into `figures` rather than `answer`; not investigated.
- **Quick-ratio excuses remain** (#7 Amcor, #11 AMD, #46 Verizon), and the inputs probably aren't in the top 5 (all three are misses). Retrieval may fix some of this.
- **Bug: #44 PepsiCo EBITDA less capex was answered with the advice decline** (`is_advice_request` false positive). Advice detection needs a look later.
- **#40 JnJ inventory turnover**: computed 4.5 vs 2.7. A wrong calculation, for the calculator tool (Unit 3).
- **Retrieval isn't fully deterministic**: the hit rate went 21 → 20 with the same retrieval code (#36 CVS legal: rank 1 → missed). Most likely the question embeddings vary slightly between calls (not verified). So small hit-rate changes (±1) in later runs are noise.

## `k10`: more chunks per question

Same prompt (`prompt_v3`) and judge. Only `TOP_K = 10` in `app/analyst.py`. Hit rate now means "an evidence page in the top 10".

Results: **15/48** (was 14/48), hit rate **25/48** (was 20/48). Answer cost +70% ($0.0368 vs $0.0218).

Findings:

- New hits at ranks 6–9: #9 Amcor industry (7, now correct), #40 JnJ inventory (7), #45 PepsiCo EBITDA margin (6), #47 Verizon capital intensity (9). #36 CVS legal is back at rank 1 (noise, see `prompt_v3`).
- Accuracy rose by only 1, within noise. More evidence doesn't automatically mean more correct answers: hit+wrong went from 14 to 15.

**Retrieval depth check** (retrieval only, no answers; evidence page rank among the top 100 chunks for each question):

| k | 5 | 10 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| Hit | 21/48 | 26/48 | 35/48 | 36/48 | 40/48 | 46/48 |

- **The evidence is found but ranked too low.** 46/48 evidence pages are in the top 100, and only 2 aren't (#25 Best Buy gross margins, #42 PepsiCo legal). That points to **reranking** (fetch many, reorder with a stronger model, keep the best ~10) more than hybrid search, which mainly helps find chunks that aren't found at all.
- k=20 already reaches 35/48, at about twice the k=10 answer cost.

## `k20`: 20 chunks per question

Same prompt and judge. Only `TOP_K = 20`.

Results: **17/48** (k10 15/48, `prompt_v3` 14/48), hit rate **34/48**. 1 fallback (#45 PepsiCo EBITDA margin; reason not checked). Answer cost ≈ $0.0014/question, 3× k=5.

**How the hit rate can mislead.** It counts a question as a hit if *any* evidence page is retrieved, but 11 of the 48 questions need 2–3 pages (e.g. a ratio from the balance sheet and the income statement). A stricter count, "all evidence pages retrieved", computed afterwards from the saved files:

| | `prompt_v3` (k5) | `k10` | `k20` |
|---|---|---|---|
| Correct | 14 | 15 | 17 |
| Hit (any evidence page) | 20 | 25 | 34 |
| Strict hit (all evidence pages) | 18 | 20 | 29 |
| Answer cost / question | $0.00045 | $0.00077 | $0.00136 |

In k10, 5 of the 6 multi-page hits were partial, and none of the 6 was answered correctly. Hit rate also *understates* in a different way: some answers are correct from other pages, or need no page ("0", "not a useful metric"). Page numbering itself is fine (checked 62/62 in Step 1). From the next run, `eval_golden.py` should log the strict count too.

Findings:

- **Accuracy follows the hit rate only weakly**: +14 hits (k5 → k20) gave +3 correct. **Hit+wrong is now the biggest box (23)**: the model has the evidence but answers wrongly.
- **Lots of churn between runs**: 12 questions changed verdict across v3/k10/k20 (e.g. #14, #26 correct before, wrong at k20). A 1–2 point difference between runs is mostly noise.
- **Calculations still fail with the inputs in hand**: #7 Amcor quick ratio (hit at 11) says "not provided"; #46 Verizon quick ratio and #5 AES inventory turnover say "not relevant / not applicable"; #44 PepsiCo EBITDA less capex gives $15,775M vs $9,068M. The model, not retrieval, is the bottleneck here. Unit 3 tools (calculator, structured financial data) target exactly this.
