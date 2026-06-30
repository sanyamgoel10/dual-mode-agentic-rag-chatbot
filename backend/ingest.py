import os
import fitz  # PyMuPDF
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from tools.search_docs import embed_text
from config import settings

DATABASE_DIR = Path(__file__).parent.parent / "database"
CHUNK_SIZE_WORDS = settings.CHUNK_SIZE_WORDS
CHUNK_OVERLAP_WORDS = settings.CHUNK_OVERLAP_WORDS


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Returns list of (page_num, text) for each page with content."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + CHUNK_SIZE_WORDS])
        chunks.append(chunk)
        i += CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS
    return chunks


def get_or_create_index(pc: Pinecone) -> object:
    existing = [idx.name for idx in pc.list_indexes()]
    if settings.PINECONE_INDEX not in existing:
        pc.create_index(
            name=settings.PINECONE_INDEX,
            dimension=settings.EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Created Pinecone index: {settings.PINECONE_INDEX}")
    return pc.Index(settings.PINECONE_INDEX)


def ingest_all():
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = get_or_create_index(pc)

    pdf_files = list(DATABASE_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files: {[f.name for f in pdf_files]}")

    vectors = []
    for pdf_path in pdf_files:
        pages = extract_pages(pdf_path)
        for page_num, page_text in pages:
            chunks = chunk_text(page_text)
            for chunk_idx, chunk in enumerate(chunks):
                embedding = embed_text(chunk, task_type="RETRIEVAL_DOCUMENT")  # gemini-embedding-001, 768 dims
                vector_id = f"{pdf_path.stem}-p{page_num}-c{chunk_idx}"
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page_num,
                        "text": chunk,
                    },
                })
                print(f"  Embedded: {vector_id}")

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i : i + batch_size])
        print(f"Upserted batch {i // batch_size + 1}")

    print(f"\nDone. {len(vectors)} vectors indexed to '{settings.PINECONE_INDEX}'.")


if __name__ == "__main__":
    ingest_all()
