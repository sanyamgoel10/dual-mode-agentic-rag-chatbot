from google import genai
from google.genai import types
from pinecone import Pinecone
from langsmith import traceable
from config import settings

_genai_client = None
_pinecone_index = None


def _get_genai_client():
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=settings.GOOGLE_API_KEY)
    return _genai_client


def _get_index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        _pinecone_index = pc.Index(settings.PINECONE_INDEX)
    return _pinecone_index


def embed_text(text: str, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
    client = _get_genai_client()
    result = client.models.embed_content(
        model=settings.EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=settings.EMBEDDING_DIMENSIONS,
        ),
    )
    return result.embeddings[0].values


@traceable(name="search_docs")
def search_docs(query: str) -> list[dict]:
    query_embedding = embed_text(query, task_type="RETRIEVAL_QUERY")
    index = _get_index()
    response = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True,
    )
    return [
        {
            "text": match.metadata.get("text", ""),
            "source": match.metadata.get("source", "unknown"),
            "page": match.metadata.get("page", 1),
        }
        for match in response.matches
    ]
