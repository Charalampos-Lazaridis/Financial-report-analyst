from typing import Literal

from openai import APIError, OpenAI
from pydantic import BaseModel
from pypdf import PdfReader

from app import config

MAX_QUESTION_LENGTH = 500
ADVICE_DECLINE = (
    "I can't give investment advice. I can answer factual questions about Apple's 10-K, "
    "for example its revenue, profit or risks."
)
INSTRUCTIONS = (
    "You are a financial analyst. Answer the question using only the 10-K report provided. "
    "If the answer is not in the report, say so. "
    "List every number you use in figures, with the section of the report it comes from. "
    "Copy each value exactly as the report writes it (e.g. in millions if the table is in millions). "
    "Set is_advice_request to true if the user asks for investment advice."
)

class Figure(BaseModel):
    name: str
    value: float
    unit: str
    period: str
    source: str

class Answer(BaseModel):
    """What the model fills in."""
    answer: str
    figures: list[Figure]
    confidence: Literal["high", "medium", "low"]
    is_advice_request: bool

class AskResponse(Answer):
    """What the API returns: the model's answer plus our own checks."""
    warnings: list[str]

FALLBACK_ANSWER = AskResponse(
    answer="Sorry, I couldn't get an answer right now. Please try again in a moment.",
    figures=[],
    confidence="low",
    is_advice_request=False,
    warnings=[],
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

def check_figures(result: Answer, text: str) -> list[str]:
    """Return a warning for every figure without a source or not found in the report."""
    warnings = []
    for figure in result.figures:
        if not figure.source.strip():
            warnings.append(f"'{figure.name}' has no source")
        if not number_in_text(figure.value, text):
            warnings.append(f"'{figure.name}' ({figure.value}) not found in the report")
    return warnings

def load_10k(path: str) -> str:
    reader = PdfReader(path)
    return "".join(page.extract_text() for page in reader.pages)

# Done once, when the server starts, not on every question
REPORT_TEXT = load_10k("data/filings/AAPL_10K_2025.pdf")
client = OpenAI(api_key=config.OPENAI_API_KEY, max_retries=3, timeout=45)

def ask(question: str) -> AskResponse:
    """"Answer one question about the 10-K. Always returns an AskResponse, even if the call fails."""
    result = None
    try:
        response = client.responses.parse(
            model=config.OPENAI_MODEL,
            instructions=INSTRUCTIONS,
            input=f"10-K report:\n{REPORT_TEXT}\n\nQuestion: {question}",
            text_format=Answer
        )
        result = response.output_parsed
    except APIError as error:
        print("API call failed: ", type(error).__name__)

    if result is None:  # the call failed, or the model gave no usable answer
        return FALLBACK_ANSWER

    if result.is_advice_request:
        result.answer = ADVICE_DECLINE
        result.figures = []

    warnings = check_figures(result, REPORT_TEXT)
    if warnings:
        result.confidence = "low"

    return AskResponse(**result.model_dump(), warnings=warnings)