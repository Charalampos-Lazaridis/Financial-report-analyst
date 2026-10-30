from typing import Literal

from openai import OpenAI
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

reader = PdfReader("data/filings/AAPL_10K_2025.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()

client = OpenAI(api_key=config.OPENAI_API_KEY)

questions = [
    "What were Apple's total net sales in fiscal 2025?",
    # "By what percentage did Greater China net sales change in 2025?",
    # "What were Orange's total net sales in fiscal 2025?",
    "    ",    # empty -> rejected, no tokens spent
    "x" * 600, # too long -> rejected, no tokens spent
    f"Given the 6% sales growth, is Apple a good investment?",
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

    response = client.responses.parse(
        model = config.OPENAI_MODEL,
        instructions=(
            "You are a financial analyst. Answer the question using only the 10-K report provided. "
            "If the answer is not in the report, say so. "
            "List every number you use in figures, with the section of the report it comes from. "
            "Set is_advice_request to true if the user asks for investment advice."
        ),
        input=f"10-K report:\n{text}\n\nQuestion: {question}",
        text_format=Answer
    )

    result = response.output_parsed

    if result.is_advice_request:
        result.answer = ADVICE_DECLINE
        result.figures = []
    
    print(result.model_dump_json(indent=2))
    print("Tokens used", response.usage.input_tokens, "in /", response.usage.output_tokens, "out")
    print("-" * 40)  # a separator line between answers