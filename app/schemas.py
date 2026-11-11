"""The data shapes: what the model fills in (Answer) and what the API returns (AskResponse)."""

from typing import Literal

from pydantic import BaseModel


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

class Memory(BaseModel):
    """One recalled past analysis (Pydantic ignores the extra 'user' key in the stored payload)."""
    question: str
    answer: str
    figures: str
    date: str

class AskResponse(Answer):
    """What the API returns: the model's answer plus our own checks."""
    sources: list[Source]
    warnings: list[str]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    calculations: list[Calculation] = []
    steps: list[str] = []  # what the agent did, in order, for the UI
    memories: list[Memory] = []  # the user's similar past analyses, shown as background
    remembered: bool = False  # True if this answer passed the write gate and was saved
