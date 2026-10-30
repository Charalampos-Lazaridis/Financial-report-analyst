from typing import Literal

from openai import OpenAI
from pydantic import BaseModel
from pypdf import PdfReader

from app import config

class Figure(BaseModel):
    name: str
    value: float
    unit: str
    period: str
    surce: str

class Answer(BaseModel):
    answer: str
    figure: list[Figure]
    confidence: Literal["high", "medioum", "low"]
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
    "Should I buy Apple stock?",
]

for question in questions:

    response = client.responses.parse(
        model = config.OPENAI_MODEL,
        instructions=(
            "You are a financial analyst. Answer the question using only the 10-K report provided. "
            "If the answer is not in the report, say so."
            "List every number you use in figures, with the section of the report it comes from. "
            "Set is_advice_request to true if the user asks for investment advice."
        ),
        input=f"10-K report:\n{text}\n\nQuestion: {question}",
        text_format=Answer
    )

    result = response.output_parsed
    print("question", question)
    print(result.model_dump_json(indent=2))
    print("Tokens used", response.usage.input_tokens, "in /", response.usage.output_tokens, "out")
    print("-" * 40)  # a separator line between answers