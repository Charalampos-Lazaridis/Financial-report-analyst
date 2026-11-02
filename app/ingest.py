import uuid
from pathlib import Path

from openai import OpenAI
from pypdf import PdfReader
from qdrant_client import QdrantClient, models

from app import config

FILINGS = Path("data/filings")
COLLECTION = "filings"
EMBEDDING_SIZE = 1536   # text-embedding-3-small turns each text into 1536 numbers
CHUNK_SIZE = 2000       # characters, about 500 tokens
CHUNK_OVERLAP = 200     # characters shared by neighbouring chunks
BATCH_SIZE = 100        # chunks per embeddings call

# (file name in data/filings without .pdf, company, fiscal year)
DOCUMENTS = [
    ("AAPL_10K_2025", "Apple", 2025),
    ("3M_2022_10K", "3M", 2022),
    ("AES_2022_10K", "AES Corporation", 2022),
    ("AMCOR_2023_10K", "Amcor", 2023),
    ("AMD_2022_10K", "AMD", 2022),
    ("AMERICANEXPRESS_2022_10K", "American Express", 2022),
    ("BESTBUY_2023_10K", "Best Buy", 2023),
    ("BOEING_2022_10K", "Boeing", 2022),
    ("CVSHEALTH_2022_10K", "CVS Health", 2022),
    ("JOHNSON_JOHNSON_2022_10K", "Johnson & Johnson", 2022),
    ("PEPSICO_2022_10K", "PepsiCo", 2022),
    ("VERIZON_2022_10K", "Verizon", 2022),
]

openai_client = OpenAI(api_key=config.OPENAI_API_KEY, max_retries=3, timeout=60)

def get_qdrant():
    """Connect to Qdrant Cloud if a URL is set, otherwise use the local folder."""
    if config.QDRANT_URL:
        return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    return QdrantClient(path=config.QDRANT_PATH)


def embed(texts):
    """Turn a list of texts into a list of vectors (one list of 1536 numbers per text)."""
    response = openai_client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def chunk_page(text):
    """Split one page's text into overlapping pieces of CHUNK_SIZE characters."""
    text=text.strip()
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def make_chunks(doc_name, company, year):
    """Read a PDF page by page and return its chunks, each with its metadata."""
    reader = PdfReader(FILINGS / f"{doc_name}.pdf")
    chunks = []
    for page_number, page in enumerate(reader.pages):
        for i, text in enumerate(chunk_page(page.extract_text())):
            chunks.append({
                "doc_name": doc_name,
                "company": company,
                "year": year,
                "page": page_number,
                "chunk": i,
                "text": text,
            })
    return chunks


def ensure_collection(client):
    """Create the collection (Qdrant's word for a table of vectors) if it doesn't exist yet."""
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=EMBEDDING_SIZE, distance=models.Distance.COSINE),
        )


def ingest_document(client, doc_name, company, year):
    """Replace one document's chunks in Qdrant. Safe to run again."""
    chunks = make_chunks(doc_name, company, year)
    # Delete this document's old chunks first, so re-ingesting never leaves duplicates.
    client.delete(
        collection_name=COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=[
                models.FieldCondition(key="doc_name", match=models.MatchValue(value=doc_name)),
            ])
        ),
    )
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        vectors = embed([chunk["text"] for chunk in batch])
        client.upsert(
            collection_name = COLLECTION,
            points=[
                models.PointStruct(id=str(uuid.uuid4()), vector=vector, payload=chunk)
                for chunk, vector in zip(batch, vectors)
            ],
        )
    return chunks


if __name__ == "__main__":
    client = get_qdrant()
    ensure_collection(client)

    total_chars = 0
    for doc_name, company, year in DOCUMENTS:
        chunks = ingest_document(client, doc_name, company, year)
        total_chars += sum(len(chunk["text"]) for chunk in chunks)
        print(f"{doc_name}: {len(chunks)} chunks")

    tokens = total_chars / 4  # rough rule: 1 token ≈ 4 characters
    print(f"points in Qdrant: {client.count(COLLECTION).count}")
    print(f"embedded ~{tokens / 1e6:.1f}M tokens ≈ ${tokens * 0.02 / 1e6:.3f}")

    # Smoke test: does a search find sensible chunks?
    question = "What was AMD's revenue in 2022?"
    hits = client.query_points(COLLECTION, query=embed([question])[0], limit=3).points
    print(f"\nTop 3 chunks for: {question}")
    for hit in hits:
        p = hit.payload
        print(f"  {hit.score:.3f}  {p['company']} {p['year']} page {p['page']}: {p['text'][:80]!r}")

    client.close()