"""Cross-encoder reranking. Scores are passed through a sigmoid so they live in [0, 1],
which makes the confidence-gate threshold readable and stable."""

from typing import Any, Protocol


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder (default: ms-marco-MiniLM-L-6-v2)."""

    def __init__(self, model_name: str, model: Any | None = None) -> None:
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name, device="cpu", max_length=512)
        self._model = model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        import torch

        scores = self._model.predict(
            [(query, p) for p in passages], activation_fn=torch.nn.Sigmoid(), batch_size=32
        )
        return [float(s) for s in scores]
