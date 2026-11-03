from typing import Literal

from openai import APIError, OpenAI
from pydantic import BaseModel

from app import config
from app.ingest import COLLECTION, embed, get_qdrant

MAX_QUESTION_LENGTH = 500
TOP_K = 20  # how many chunks the model gets per question
ADVICE_DECLINE = (
    "I can't give investment advice. I can answer factual questions about the 10-K reports I have, "
    "for example a company's revenue, profit or risks."
)
INSTRUCTIONS = (
    "You are a financial analyst. Answer the question using only the numbered excerpts from 10-K reports. "
    "Each excerpt is labelled with its company, fiscal year and page.\n"
    "- Start with a direct conclusion (yes/no, the number, or the main drivers), then support it in 2-4 sentences "
    "with the specific numbers, names and reasons from the excerpts.\n"
    "- If the question needs a calculation (ratio, margin, turnover, growth, year-over-year change), "
    "find the inputs in the excerpts, do the calculation, and show the formula and the inputs.\n"
    "- Some questions say what to answer in a special case (for example, if a metric doesn't apply). "
    "Use that special answer only when the excerpts clearly show that case. "
    "If the inputs for the calculation are in the excerpts, calculate instead.\n"
    "- Say that the reports don't contain the answer only if the excerpts have nothing relevant. "
    "Never make up a number that isn't in the excerpts.\n"
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
    page: int  # 0-indexed pypdf page, same as FinanceBench
    score: float # cosine similarity to the question

class AskResponse(Answer):
    """What the API returns: the model's answer plus our own checks."""
    sources: list[Source]
    warnings: list[str]
    input_tokens: int
    output_tokens: int
    cost_usd: float

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


def check_figures(result: Answer, chunks: list[dict]) -> list[str]:
    """Return a warning for every figure whose excerpt doesn't exist or doesn't contain its number."""
    warnings = []
    for figure in result.figures:
        if not 1 <= figure.source <= len(chunks):
            warnings.append(f"'{figure.name}' cites excerpt [{figure.source}], which doesn't exist")
        elif not number_in_text(figure.value, chunks[figure.source - 1]["text"]):
            warnings.append(f"'{figure.name}' ({figure.value}) not found in excerpt [{figure.source}]")
    return warnings


def retrieve(question: str) -> list[dict]:
    """Return the TOP_K chunks closest in meaning to the question, from all reports."""
    vector = embed([question])[0]
    hits = qdrant.query_points(COLLECTION, query=vector, limit=TOP_K).points
    return [{**hit.payload, "score": hit.score} for hit in hits]


def format_excerpts(chunks: list[dict]) -> str:
    """Number the chunks like footnotes, each with its company, year and page."""
    return "\n\n".join(
        f"[{i}] {chunk['company']} {chunk['year']}, page {chunk['page']}\n{chunk['text']}"
        for i, chunk in enumerate(chunks, start=1)
    )

# Done once, when the server starts, not on every question
qdrant = get_qdrant()
client = OpenAI(api_key=config.OPENAI_API_KEY, max_retries=3, timeout=45)

def ask(question: str) -> AskResponse:
    """Answer one question from the 10-K reports. Always returns an AskResponse, even if a call fails."""
    result = None
    chunks = []
    try:
        chunks = retrieve(question)
        response = client.responses.parse(
            model=config.OPENAI_MODEL,
            instructions=INSTRUCTIONS,
            input=f"Excerpts:\n\n{format_excerpts(chunks)}\n\nQuestion: {question}",
            text_format=Answer,
        )
        result = response.output_parsed
    except APIError as error:
        print("API call failed: ", type(error).__name__)

    if result is None:  # a call failed, or the model gave no usable answer
        return FALLBACK_ANSWER

    # The question's embedding also costs money, but ~20 tokens × $0.02/1M ≈ $0.0000004, so we leave it out.
    price_in, price_out = PRICES[config.OPENAI_MODEL]
    usage = response.usage
    cost = (usage.input_tokens * price_in + usage.output_tokens * price_out) / 1_000_000

    if result.is_advice_request:
        result.answer = ADVICE_DECLINE
        result.figures = []
        chunks = []  # no sources for a declined question

    warnings = check_figures(result, chunks)
    if warnings:
        result.confidence = "low"

    sources = [
        Source(id=i, company=chunk["company"], year=chunk["year"], page=chunk["page"],
               score=round(chunk["score"], 3))
        for i, chunk in enumerate(chunks, start=1)
    ]
    return AskResponse(
        **result.model_dump(),
        sources=sources,
        warnings=warnings,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=round(cost, 6),
    )