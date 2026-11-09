import json
import math
import re
import ast
import asyncio
from functools import cache

from typing import Literal
from pathlib import Path

from openai import APIError, OpenAI
from pydantic import BaseModel
from qdrant_client import models
from fastmcp import Client

from app import config
from app.ingest import COLLECTION, DOCUMENTS, embed, get_qdrant
from app.tools import CALCULATOR_TOOL, calculate
from app.sec_mcp_server import mcp as sec_server
from collections import Counter


NUMBER = r"\d+(?:\.\d+)?"  # a number like 11512 or 21.4
MIN_CHECKED = 1000  # smaller numbers are skipped, so 2, 100 and 365 stay allowed
MAX_STEPS = 4  # most model calls for one question
STATEMENT_K = 3  # chunks from each financial statement
DOC_NAMES = {company: doc_name for doc_name, company, _ in DOCUMENTS}
MAX_QUESTION_LENGTH = 500
TOP_K = 11  # how many chunks the model gets per question
STATEMENT_QUERIES = [
    "consolidated balance sheets total current assets inventories accounts receivable total current liabilities total assets",
    "consolidated statements of operations income revenue cost of sales operating income net income",
    "consolidated statements of cash flows depreciation and amortization capital expenditures net cash provided by operating activities",
]
ADVICE_DECLINE = (
    "I can't give investment advice. I can answer factual questions about the 10-K reports I have, "
    "for example a company's revenue, profit or risks."
)
INSTRUCTIONS = (
    "You are a financial analyst. Answer the question using only the numbered excerpts from 10-K reports. "
    "Each excerpt is labelled with its company, fiscal year and page.\n"
    "- The last excerpt may be key figures from the company's SEC XBRL data, in millions of USD, for the "
    "report's fiscal year and the year before. Prefer its values for those figures and cite it like any excerpt.\n"
    "- If a number you need isn't in the excerpts (for example an earlier year, or a figure missing from the "
    "key figures), look it up with get_sec_fact, then cite the excerpt number it returns.\n"
    "- Start with a direct conclusion (yes/no, the number, or the main drivers), then support it in 2-4 sentences "
    "with the specific numbers, names and reasons from the excerpts.\n"
    "- If the question needs a calculation (ratio, margin, turnover, growth, year-over-year change), "
    "find the inputs in the excerpts, call the calculate tool, and show the formula and the inputs. "
    "Never do arithmetic yourself. In figures, give a calculated value with source 0 and copy the tool's "
    "result exactly; you may round it in the answer text.\n"
    "- Some questions say what to answer in a special case (for example, if a metric doesn't apply). "
    "Use that special answer only when the excerpts clearly show that case. "
    "If the inputs for the calculation are in the excerpts, calculate instead.\n"
    "- Say that the reports don't contain the answer only if the excerpts have nothing relevant. "
    "Never replace a missing input with a different number.\n"
    "- Use these definitions unless the question gives its own: "
    "inventory turnover = cost of sales / average inventory; "
    "quick ratio = (cash + short-term investments + receivables) / current liabilities; "
    "return on assets = net income / average total assets; "
    "EBITDA = operating income + depreciation and amortization; a net loss is a negative number.\n"
    "- Do each calculation once. When you have the results you need, write your answer.\n"
    "Cite the excerpts you use in the answer, like [2]. "
    "List every number you use in figures, with source set to the number of the excerpt it comes from. "
    "Copy each value exactly as the report writes it (e.g. in millions if the table is in millions). "
    "Set is_advice_request to true only if the user asks for a recommendation or opinion on what to do "
    "with an investment (buy, sell, hold, or whether a stock is a good investment). "
    "Questions that ask for facts, metrics or calculations are not advice, "
    "even if they mention investors or an analyst's perspective."
)
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}
KEY_FIGURES = json.loads(Path("data/key_figures.json").read_text(encoding="utf-8"))

class Figure(BaseModel):
    name: str
    value: float
    unit: str
    period: str
    source: int  # excerpt number, e.g. 2 for [2]

class Answer(BaseModel):
    """What the model fills in."""
    answer: str
    figures: list[Figure]
    confidence: Literal["high", "medium", "low"]
    is_advice_request: bool

class Source(BaseModel):
    """One retrieved excerpt, so a reader can find it in the PDF."""
    id: int
    company: str
    year: int
    page: int | None # 0-indexed pypdf page, same as FinanceBench
    score: float # cosine similarity to the question

class Calculation(BaseModel):
    expression: str
    result: float

class AskResponse(Answer):
    """What the API returns: the model's answer plus our own checks."""
    sources: list[Source]
    warnings: list[str]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    calculations: list[Calculation] = []
    steps: list[str] = []  # what the agent did, in order, for the UI

FALLBACK_ANSWER = AskResponse(
    answer="Sorry, I couldn't get an answer right now. Please try again in a moment.",
    figures=[],
    confidence="low",
    is_advice_request=False,
    sources=[],
    warnings=[],
    input_tokens=0,
    output_tokens=0,
    cost_usd=0.0,
)

def number_in_text(value: float, text: str) -> bool:
    """Check if a number appears in the text, written the way 10-Ks write numbers."""
    number = abs(value)
    candidates = {
        f"{number:,.0f}",  # 416161.0 -> "416,161"
        f"{number:,.1f}",  # 46.2     -> "46.2"
        f"{number:,.2f}",  # 6.08     -> "6.08"
    }
    if value < 0:
        # 10-Ks write negatives in parentheses: -4 -> "(4)"
        candidates = {f"({candidate})" for candidate in candidates}
    return any(candidate in text for candidate in candidates)


def calculator_values(calculations: list[Calculation]) -> list[float]:
    """Every value the calculator produced, including the parts of each expression:
    (11512 + 2763) / 86392 gives 0.1652, and also 14275 from the part in brackets."""
    values = [c.result for c in calculations]
    for c in calculations:
        for node in ast.walk(ast.parse(c.expression, mode="eval")):
            if isinstance(node, ast.BinOp):  # one operation, e.g. 11512 + 2763
                try:
                    values.append(calculate(ast.unparse(node)))
                except (ValueError, SyntaxError, ZeroDivisionError):
                    pass
    return values


def matches_calculation(value: float, results: list[float]) -> bool:
    """True if value is a calculator result, also when rounded (0.165 for 0.16524) or as a percent (16.52)."""
    text = f"{value}"
    decimals = len(text.split(".")[1]) if "." in text else 0  # 0.165 -> 3
    for result in results:
        for candidate in (result, result * 100):
            if math.isclose(value, candidate, rel_tol=0.001) or round(candidate, decimals) == value:
                return True
    return False


def check_figures(result: Answer, chunks: list[dict], calculations: list[Calculation]) -> list[str]:
    """Return a warning for every figure that doesn't match its excerpt or a calculator result."""
    results = calculator_values(calculations)
    warnings = []
    for figure in result.figures:
        if figure.source == 0:
            if not matches_calculation(figure.value, results):
                warnings.append(f"'{figure.name}' ({figure.value}) doesn't match any calculator result")
        elif not 1 <= figure.source <= len(chunks):
            warnings.append(f"'{figure.name}' cites excerpt [{figure.source}], which doesn't exist")
        elif (not number_in_text(figure.value, chunks[figure.source - 1]["text"])
              and not matches_calculation(figure.value, results)):
            warnings.append(f"'{figure.name}' ({figure.value}) not found in excerpt [{figure.source}]")
    return warnings



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


# Done once, when the server starts, not on every question
qdrant = get_qdrant()
client = OpenAI(api_key=config.OPENAI_API_KEY, max_retries=3, timeout=45)

# --- SEC lookups through MCP: our agent is a client of app/sec_mcp_server.py ---
# Client(sec_server) speaks MCP to the server in memory. Client("https://.../mcp") would speak the same
# protocol over HTTP to a server elsewhere: only that argument changes.

async def _list_sec_tools() -> list:
    async with Client(sec_server) as mcp_client:
        return await mcp_client.list_tools()


async def _call_sec_tool(name: str, arguments: dict):
    async with Client(sec_server) as mcp_client:
        return await mcp_client.call_tool(name, arguments, raise_on_error=False)


@cache
def sec_tools() -> list[dict]:
    """Ask the MCP server which tools it has (tool discovery), once, in OpenAI's tool format."""
    return [
        {"type": "function", "name": tool.name, "description": tool.description,
         "parameters": tool.input_schema, "strict": False}
        for tool in asyncio.run(_list_sec_tools())
    ]


def run_sec_tool(call, chunks: list[dict], steps: list[str]) -> str:
    """Send the model's SEC lookup to the MCP server. The result becomes a new excerpt the model can cite."""
    arguments = json.loads(call.arguments)
    result = asyncio.run(_call_sec_tool(call.name, arguments))
    text = result.content[0].text if result.content else "no result"
    if result.is_error:
        steps.append(f"SEC lookup {arguments} failed: {text}")
        return f"Error: {text}"  # the model sees the reason, e.g. the list of valid names, and can retry

    fact = result.data
    chunks.append({
        "doc_name": "", "company": fact["company"], "year": fact["fiscal_year"], "page": None,
        "chunk": f"sec-{fact['tag']}-{fact['fiscal_year']}", "score": 0.0,
        "text": (f"SEC XBRL lookup: {fact['figure']} [XBRL tag {fact['tag']}], fiscal year {fact['fiscal_year']} "
                 f"(period ending {fact['period_end']}): {format_musd(fact['value'])} {fact['unit']}"),
    })
    steps.append(f"SEC lookup via MCP: {fact['company']} {fact['figure']} FY{fact['fiscal_year']} = "
                 f"{format_musd(fact['value'])} {fact['unit']}, added as excerpt [{len(chunks)}]")
    return f"Excerpt [{len(chunks)}]: {chunks[-1]['text']}"


def unknown_inputs(expression: str, chunks: list[dict], calculations: list[Calculation]) -> list[str]:
    """Return the numbers in a calculation that aren't in the excerpts or earlier calculator results."""
    text = " ".join(chunk["text"] for chunk in chunks).replace(",", "")  # 11,512 -> 11512
    known = {float(number) for number in re.findall(NUMBER, text)}
    unknown = []
    for number in re.findall(NUMBER, expression):
        value = float(number)
        if value < MIN_CHECKED or value in known:
            continue
        if any(math.isclose(value, c.result, rel_tol=0.001) for c in calculations):
            continue  # it comes from an earlier calculation
        unknown.append(number)
    return unknown


def run_tool(call, chunks: list[dict], calculations: list[Calculation], steps: list[str]) -> str:
    """Run one tool the model asked for, note it in steps, and return the text the model gets back."""
    if call.name != "calculate":
        return run_sec_tool(call, chunks, steps)
    expression = json.loads(call.arguments)["expression"]
    unknown = unknown_inputs(expression, chunks, calculations)
    if unknown:
        print(f"rejected: {expression} ({', '.join(unknown)} not found)")
        steps.append(f"Rejected `{expression}`: {', '.join(unknown)} isn't in the excerpts, so the model must recheck")
        return (f"Error: {', '.join(unknown)} isn't in the excerpts or your earlier results. "
                "Check the number and use the value exactly as the report writes it.")
    try:
        value = calculate(expression)
    except (ValueError, SyntaxError, ZeroDivisionError) as error:
        steps.append(f"Calculator error on `{expression}`: {error}")
        return f"Error: {error}"  # the model sees this and can fix its expression
    calculations.append(Calculation(expression=expression, result=value))
    print(f"calculate: {expression} = {value}")
    steps.append(f"Calculated `{expression}` = {round(value, 4):,}")
    return str(value)


def run_agent(question: str, chunks: list[dict], steps: list[str]):
    """Call the model, running the tools it asks for, until it answers or MAX_STEPS runs out."""
    messages = [{"role": "user", "content": f"Excerpts:\n\n{format_excerpts(chunks)}\n\nQuestion: {question}"}]
    calculations = []
    input_tokens = output_tokens = 0

    for step in range(MAX_STEPS):
        last_step = step == MAX_STEPS - 1
        response = client.responses.parse(
            model=config.OPENAI_MODEL,
            instructions=INSTRUCTIONS,
            input=messages,
            tools=[CALCULATOR_TOOL, *sec_tools()],
            tool_choice="none" if last_step else "auto",  # last step: must answer now
            text_format=Answer,
        )

        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens

        tool_calls = [item for item in response.output if item.type == "function_call"]
        if not tool_calls:  # the model is done
            result = response.output_parsed
            if last_step and result:
                print("Forced an answer on the last step")
                result.confidence = "low"
                steps.append(f"Step {step + 1}: last step, tools off, so the model had to answer now "
                             "(confidence set to low)")
            else:
                steps.append(f"Step {step + 1}: the model wrote its answer")
            return result, calculations, input_tokens, output_tokens

        steps.append(f"Step {step + 1}: the model asked for {len(tool_calls)} tool call(s)")

        for call in tool_calls:
            messages.append({"type": "function_call", "call_id": call.call_id,
                             "name": call.name, "arguments": call.arguments})
            messages.append({"type": "function_call_output", "call_id": call.call_id,
                             "output": run_tool(call, chunks, calculations, steps)})

    print(f"Agent stopped: no answer after {MAX_STEPS} steps")
    return None, calculations, input_tokens, output_tokens


def ask(question: str) -> AskResponse:
    """Answer one question from the 10-K reports. Always returns an AskResponse, even if a call fails."""
    result = None
    chunks = []
    steps = []
    try:
        chunks = retrieve_context(question)
        pages = [chunk for chunk in chunks if chunk["page"] is not None]
        retrieved = f"Searched the 12 reports: {len(pages)} page excerpts"
        if len(pages) < len(chunks):  # the last excerpt is the SEC table
            retrieved += f", plus SEC key figures for {chunks[-1]['company']} {chunks[-1]['year']}"
        steps.append(retrieved)
        result, calculations, input_tokens, output_tokens = run_agent(question, chunks, steps)
    except APIError as error:
        print("API call failed: ", type(error).__name__)

    if result is None:  # a call failed, the model gave no usable answer, or it ran out of steps
        return FALLBACK_ANSWER

    # The question's embedding also costs money, but ~20 tokens × $0.02/1M ≈ $0.0000004, so we leave it out.
    price_in, price_out = PRICES[config.OPENAI_MODEL]
    cost = (input_tokens * price_in + output_tokens * price_out) / 1_000_000

    if result.is_advice_request:
        result.answer = ADVICE_DECLINE
        result.figures = []
        chunks = []  # no sources for a declined question
        steps.append("Recognised as an investment-advice request: answer replaced with a polite decline")

    warnings = check_figures(result, chunks, calculations)
    if warnings:
        result.confidence = "low"

    if result.figures:
        steps.append(f"Checked {len(result.figures)} figure(s) against their excerpts and calculator results: "
                     + (f"{len(warnings)} warning(s), confidence set to low" if warnings else "all match"))

    sources = [
        Source(id=i, company=chunk["company"], year=chunk["year"], page=chunk["page"],
               score=round(chunk["score"], 3))
        for i, chunk in enumerate(chunks, start=1)
    ]
    return AskResponse(
        **result.model_dump(),
        sources=sources,
        warnings=warnings,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(cost, 6),
        calculations=calculations,
        steps=steps
    )