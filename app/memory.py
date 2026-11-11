"""Episodic memory: each user's past analyses that passed every check, recalled by meaning."""

import uuid
from datetime import date

from qdrant_client import models

from app.ingest import EMBEDDING_SIZE

MEMORIES = "memories"
RECALL_K = 3
MIN_SCORE = 0.45  # cosine similarity; below this a past question is about something else


def ensure_memories(qdrant) -> None:
    """Create the memories collection and its 'user' index if they don't exist yet."""
    if not qdrant.collection_exists(MEMORIES):
        qdrant.create_collection(MEMORIES, vectors_config=models.VectorParams(
            size=EMBEDDING_SIZE, distance=models.Distance.COSINE))
    # Qdrant Cloud needs an index on a field before we can filter on it; the local folder mode ignores it.
    qdrant.create_payload_index(MEMORIES, field_name="user", field_schema=models.PayloadSchemaType.KEYWORD)


def recall(qdrant, user: str, vector: list[float]) -> list[dict]:
    """The user's most similar past analyses, most similar first."""
    only_this_user = models.Filter(must=[models.FieldCondition(key="user", match=models.MatchValue(value=user))])
    hits = qdrant.query_points(MEMORIES, query=vector, limit=RECALL_K,
                               score_threshold=MIN_SCORE, query_filter=only_this_user).points
    return [hit.payload for hit in hits]


def should_remember(response) -> bool:
    """The write gate: remember only answers that passed every check."""
    return (not response.warnings and response.confidence != "low"
            and not response.is_advice_request and bool(response.sources))


def remember(qdrant, user: str, question: str, vector: list[float], response) -> None:
    """Save one past analysis: the question, a short answer, the key figures and today's date."""
    figures = "; ".join(f"{f.name} {f.value:,g} {f.unit}" for f in response.figures[:5])
    payload = {"user": user, "question": question, "answer": response.answer[:300],
               "figures": figures, "date": date.today().isoformat()}
    qdrant.upsert(MEMORIES, points=[models.PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)])


def format_memories(memories: list[dict]) -> str:
    """Recalled analyses as background text for the model: no excerpt numbers, so they can't be cited."""
    lines = [f"- {m['date']}: {m['question']} -> {m['answer']}" for m in memories]
    return ("Earlier analyses for this user (background only, not a source: don't cite them or take "
            "numbers from them; use the excerpts):\n" + "\n".join(lines))
