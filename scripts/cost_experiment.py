import time

from openai import OpenAI
from pypdf import PdfReader

from app import config
from app.analyst import INSTRUCTIONS, Answer

# US$ per 1 million tokens: (input, output). Check https://openai.com/api/pricing before running.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}

# All three answers are in Item 7 (MD&A). Second item = text we expect in a correct answer.
QUESTIONS = [
    ("What were Apple's total net sales in fiscal 2025?", "416,161"),
    ("By what percentage did Greater China net sales change in 2025?", "4%"),
    ("What was Apple's research and development expense in fiscal 2025?", "34,550"),
]

pages = [page.extract_text() for page in PdfReader("data/filings/AAPL_10K_2025.pdf").pages]
CONTEXTS = {
    "whole 10-K": "".join(pages),
    "MD&A only": "".join(pages[29:38]),  # Item 7 starts on page 29, Item 7A on page 38
}

client = OpenAI(api_key=config.OPENAI_API_KEY)

print(f"{'model':<12} {'context':<11} {'in':>7} {'cached':>7} {'out':>5} {'cost $':>8} {'secs':>5}  correct")
for model in PRICES:
    for context_name, context in CONTEXTS.items():
        for question, expected in QUESTIONS:
            start = time.perf_counter()
            response = client.responses.parse(
                model=model,
                instructions=INSTRUCTIONS,
                input=f"10-K report:\n{context}\n\nQuestion: {question}",
                text_format=Answer,
            )
            seconds = time.perf_counter() - start

            usage = response.usage
            cached = usage.input_tokens_details.cached_tokens
            price_in, price_out = PRICES[model]
            cost = (usage.input_tokens * price_in + usage.output_tokens * price_out) / 1_000_000
            correct = expected in response.output_parsed.answer

            print(f"{model:<12} {context_name:<11} {usage.input_tokens:>7} {cached:>7} "
                  f"{usage.output_tokens:>5} {cost:>8.4f} {seconds:>5.1f}  {correct}")