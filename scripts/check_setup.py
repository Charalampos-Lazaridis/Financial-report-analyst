"""Step 0 smoke test: can we load settings and reach OpenAI?

Run from the project root:  python -m scripts.check_setup
"""

from openai import OpenAI

from app import config

client = OpenAI(api_key=config.OPENAI_API_KEY)

response = client.responses.create(
    model=config.OPENAI_MODEL,
    input="Reply with exactly: setup works",
)

print("Model:", config.OPENAI_MODEL)
print("Reply:", response.output_text)
print("Tokens used:", response.usage.input_tokens, "in /", response.usage.output_tokens, "out")
