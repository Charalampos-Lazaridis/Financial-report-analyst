"""ask(): answer one question. Runs retrieval, memory recall, the agent and the checks, in that order.

The parts live in retrieval.py, agent.py, checks.py, memory.py and schemas.py. Names that scripts import
from here (TOP_K, qdrant, INSTRUCTIONS, Answer, ...) are re-exported below so those imports keep working.
"""

from openai import APIError

from app import config
from app.agent import INSTRUCTIONS, MAX_STEPS, run_agent  # noqa: F401 (INSTRUCTIONS, MAX_STEPS re-exported)
from app.checks import check_figures, fix_sources
from app.ingest import embed
from app.memory import ensure_memories, format_memories, recall, remember, should_remember
from app.retrieval import TOP_K, qdrant, retrieve_context  # noqa: F401 (TOP_K re-exported)
from app.schemas import Answer, AskResponse, Source  # noqa: F401 (Answer re-exported)

MAX_QUESTION_LENGTH = 500

ADVICE_DECLINE = (
    "I can't give investment advice. I can answer factual questions about the 10-K reports I have, "
    "for example a company's revenue, profit or risks."
)

PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}

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

ensure_memories(qdrant)

def ask(question: str, user: str | None = None) -> AskResponse:
    """Answer one question from the 10-K reports. Always returns an AskResponse, even if a call fails.
    With a user name, recall that user's similar past analyses and remember this one if it passes every check."""
    result = None
    chunks = []
    steps = []
    memories = []
    vector = None
    user = (user or "").strip().lower() or None  # "Lampis " and "lampis" are the same person; "" = no memory
    try:
        chunks = retrieve_context(question)
        pages = [chunk for chunk in chunks if chunk["page"] is not None]
        retrieved = f"Searched the 12 reports: {len(pages)} page excerpts"
        if len(pages) < len(chunks):  # the last excerpt is the SEC table
            retrieved += f", plus SEC key figures for {chunks[-1]['company']} {chunks[-1]['year']}"
        steps.append(retrieved)
        if user:
            vector = embed([question])[0]
            memories = recall(qdrant, user, vector)
            steps.append(f"Recalled {len(memories)} similar earlier analysis(es) for '{user}'"
                         + (" (given to the model as background, not as a source)" if memories else ""))
        background = format_memories(memories) if memories else ""
        result, calculations, input_tokens, output_tokens = run_agent(question, chunks, steps, background)
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

    fix_sources(result, chunks, steps)
    warnings = check_figures(result, chunks, calculations, question)
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
    response = AskResponse(
        **result.model_dump(),
        sources=sources,
        warnings=warnings,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(cost, 6),
        calculations=calculations,
        steps=steps,
        memories=memories,
    )
    if user:  # the write gate: only answers that passed every check are remembered
        if should_remember(response):
            remember(qdrant, user, question, vector, response)
            response.remembered = True
            response.steps.append("Passed every check, so saved to memory")
        else:
            response.steps.append("Not saved to memory: only answers that pass every check are remembered")
    return response
