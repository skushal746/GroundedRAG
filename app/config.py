"""Application settings, loaded from environment variables (and `.env`)."""

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalMode(StrEnum):
    """Retrieval strategy. `hybrid_rerank` is the production default; the others exist
    so the eval harness can run ablations through the exact same code path."""

    VECTOR = "vector"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class LLMProvider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    FAKE = "fake"


class EmbeddingProvider(StrEnum):
    LOCAL = "local"
    OPENAI = "openai"
    FAKE = "fake"


DEFAULT_LLM_MODELS: dict[LLMProvider, str] = {
    LLMProvider.OPENAI: "gpt-4.1-mini",
    LLMProvider.ANTHROPIC: "claude-sonnet-5-5",
    LLMProvider.OLLAMA: "Qwen3-Coder:latest",
    LLMProvider.FAKE: "fake-llm",
}


class Settings(BaseSettings):
    """All runtime configuration. Every field maps to an upper-case env var."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- service ---
    app_name: str = "GroundedRAG"
    log_level: str = "INFO"
    api_keys: str = Field(default="", description="Comma-separated list of valid X-API-Key values")
    max_upload_mb: int = 20

    # --- database ---
    database_url: str = "postgresql+asyncpg://groundedrag:groundedrag@localhost:5433/groundedrag"
    db_pool_size: int = 5

    # --- embeddings ---
    embedding_provider: EmbeddingProvider = EmbeddingProvider.LOCAL
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    openai_embedding_model: str = "text-embedding-3-small"

    # --- chunking ---
    chunk_max_tokens: int = 500
    chunk_overlap_tokens: int = 80

    # --- retrieval ---
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID_RERANK
    vector_top_k: int = 20
    keyword_top_k: int = 20
    rrf_k: int = 60
    final_top_k: int = 5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    min_rerank_score: float = 1e-4

    # --- generation ---
    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_model: str = ""
    llm_temperature: float | None = 0.0
    llm_max_tokens: int = 800
    llm_timeout_s: float = 30.0
    openai_api_key: SecretStr = SecretStr("")
    anthropic_api_key: SecretStr = SecretStr("")
    # Local Ollama server (OpenAI-compatible API). Used when LLM_PROVIDER=ollama, and as an
    # automatic fallback when LLM_PROVIDER is openai/anthropic but that provider's key is empty.
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = DEFAULT_LLM_MODELS[LLMProvider.OLLAMA]
    ollama_timeout_s: float = 120.0

    # --- conversation ---
    history_turns: int = 3
    fallback_message: str = (
        "I'm sorry, I couldn't find a reliable answer to that in our help documentation. "
        "Would you like me to connect you with a member of our support team?"
    )

    @field_validator("llm_temperature", mode="before")
    @classmethod
    def _empty_temperature_is_none(cls, value: object) -> object:
        return None if value == "" else value

    @property
    def effective_llm_provider(self) -> LLMProvider:
        """The provider actually used: openai/anthropic without an API key fall back to Ollama."""
        if self.llm_provider is LLMProvider.OPENAI and not self.openai_api_key.get_secret_value():
            return LLMProvider.OLLAMA
        if (
            self.llm_provider is LLMProvider.ANTHROPIC
            and not self.anthropic_api_key.get_secret_value()
        ):
            return LLMProvider.OLLAMA
        return self.llm_provider

    @property
    def llm_fell_back_to_ollama(self) -> bool:
        return self.effective_llm_provider is not self.llm_provider

    @property
    def resolved_llm_model(self) -> str:
        """The configured model, or the provider's sensible default.

        On the Ollama fallback, LLM_MODEL (meant for the cloud provider) is ignored in favour
        of OLLAMA_MODEL.
        """
        provider = self.effective_llm_provider
        if provider is LLMProvider.OLLAMA:
            if self.llm_provider is LLMProvider.OLLAMA and self.llm_model:
                return self.llm_model
            return self.ollama_model
        return self.llm_model or DEFAULT_LLM_MODELS[provider]

    @property
    def api_key_set(self) -> frozenset[str]:
        return frozenset(k.strip() for k in self.api_keys.split(",") if k.strip())


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
