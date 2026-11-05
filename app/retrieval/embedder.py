"""Embedding interface + implementations (local sentence-transformers, OpenAI)."""

from typing import Any, Protocol, runtime_checkable

from app.config import EmbeddingProvider, Settings

# bge-*-v1.5 models are trained with this instruction on the *query* side only.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@runtime_checkable
class Embedder(Protocol):
    """Synchronous (CPU-bound) embedder; callers run it in a worker thread."""

    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Local embeddings via sentence-transformers (default: BAAI/bge-small-en-v1.5)."""

    def __init__(self, model_name: str, model: Any | None = None) -> None:
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, device="cpu")
        self._model = model
        dim = model.get_sentence_embedding_dimension()
        if dim is None:
            raise ValueError(f"Cannot determine embedding dimension of {model_name}")
        self.dim = int(dim)
        self._query_prefix = BGE_QUERY_INSTRUCTION if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [list(map(float, v)) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode([self._query_prefix + text], normalize_embeddings=True)[0]
        return list(map(float, vector))


class OpenAIEmbedder:
    """OpenAI embeddings (e.g. text-embedding-3-small, truncated to `dim` dimensions)."""

    def __init__(self, model: str, dim: int, api_key: str, client: Any | None = None) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        self._client = client
        self._model = model
        self.dim = dim

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self._model, input=texts, dimensions=self.dim
        )
        return [list(item.embedding) for item in response.data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v for i in range(0, len(texts), 256) for v in self._embed(texts[i : i + 256])]

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def build_embedder(settings: Settings) -> Embedder:
    """Construct the configured embedder and check its dimension matches the DB schema."""
    embedder: Embedder
    if settings.embedding_provider is EmbeddingProvider.OPENAI:
        embedder = OpenAIEmbedder(
            settings.openai_embedding_model,
            settings.embedding_dim,
            settings.openai_api_key.get_secret_value(),
        )
    elif settings.embedding_provider is EmbeddingProvider.FAKE:
        from app.testing import HashingEmbedder

        embedder = HashingEmbedder(settings.embedding_dim)
    else:
        embedder = SentenceTransformerEmbedder(settings.embedding_model)
    if embedder.dim != settings.embedding_dim:
        raise ValueError(
            f"Embedding model produces {embedder.dim}-dim vectors but EMBEDDING_DIM="
            f"{settings.embedding_dim}; they must match the pgvector column."
        )
    return embedder
