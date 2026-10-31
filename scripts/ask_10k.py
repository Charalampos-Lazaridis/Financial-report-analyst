#UNIT 1: Steps 1-4

from typing import Literal

from openai import OpenAI, APIError
from pydantic import BaseModel
from pypdf import PdfReader

from app import config

MAX_QUESTION_LENGTH = 500
ADVICE_DECLINE = (
    "I can't give investment advice. I can answer factual questions about Apple's 10-K, "
    "for example its revenue, profit or risks."
)

class Figure(BaseModel):
    name: str
    value: float
    unit: str
    period: str
    source: str

class Answer(BaseModel):
    answer: str
    figures: list[Figure]
    confidence: Literal["high", "medium", "low"]
    is_advice_request: bool

FALLBACK_ANSWER = Answer(
    answer="Sorry, I couldn't get an answer right now. Please try again in a moment.",
    figures=[],
    confidence="low",
    is_advice_request=False,
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


reader = PdfReader("data/filings/AAPL_10K_2025.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()

client = OpenAI(api_key=config.OPENAI_API_KEY, max_retries=3, timeout=45)

questions = [
    # "What were Apple's total net sales in fiscal 2025?",
    "By what percentage did Greater China net sales change in 2025?",
    # "What were Orange's total net sales in fiscal 2025?",
    # "    ",    # empty -> rejected, no tokens spent
    # "x" * 600, # too long -> rejected, no tokens spent
    # f"Given the 6% sales growth, is Apple a good investment?",
]

for question in questions:
    print("Question:", question)

    question = question.strip()
    if not question:
        print("Please type a question")
        print("-" * 40)
        continue
    
    if len(question) > MAX_QUESTION_LENGTH:
        print(f"Question too long (max {MAX_QUESTION_LENGTH} characters)")
        print("-" * 40)
        continue

    result = None
    try: 
        response = client.responses.parse(
            model = config.OPENAI_MODEL,
            instructions=(
                "You are a financial analyst. Answer the question using only the 10-K report provided. "
                "If the answer is not in the report, say so. "
                "List every number you use in figures, with the section of the report it comes from. "
                "Copy each value exactly as the report writes it (e.g. in millions if the table is in millions). "
                "Set is_advice_request to true if the user asks for investment advice."
            ),
            input=f"10-K report:\n{text}\n\nQuestion: {question}",
            text_format=Answer
        )
        result = response.output_parsed
    except APIError as error:
        print("API call failed:", type(error).__name__)

    if result is None:
        print(FALLBACK_ANSWER.model_dump_json(indent=2))
        print("-" * 40)
        continue

    if result.is_advice_request:
        result.answer = ADVICE_DECLINE
        result.figures = []

    warnings = check_figures(result, text)
    if warnings:
        result.confidence = "low"
    
    print(result.model_dump_json(indent=2))
    for warning in warnings:
        print("Warning:", warning)
    print("Tokens used", response.usage.input_tokens, "in /", response.usage.output_tokens, "out")
    print("-" * 40)  # a separator line between answers