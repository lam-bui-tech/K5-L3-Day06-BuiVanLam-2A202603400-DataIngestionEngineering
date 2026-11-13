"""Choose a real embedding provider (API or local) and implement this contract."""
from typing import Protocol
import numpy as np


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Real local embedding adapter using SentenceTransformers.

    Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384 dimensions).
    Strictly forbids silent fallback to random/mock vectors per RULES.md.
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> None:
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            # Determine actual dimensions
            if hasattr(self._model, "get_embedding_dimension"):
                dim = self._model.get_embedding_dimension()
            elif hasattr(self._model, "get_sentence_embedding_dimension"):
                dim = self._model.get_sentence_embedding_dimension()
            else:
                test_emb = self._model.encode(["test"], convert_to_numpy=True)
                dim = int(test_emb.shape[1])
            self.dimensions = int(dim)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model '{model_name}'. Real embeddings are required per RULES.md. "
                f"Original error: {exc}"
            ) from exc

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors: np.ndarray = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return vectors.tolist()
        except Exception as exc:
            raise RuntimeError(f"Error generating document embeddings: {exc}") from exc

    def embed_query(self, question: str) -> list[float]:
        if not question or not question.strip():
            raise ValueError("Query string cannot be empty.")
        try:
            vector: np.ndarray = self._model.encode(
                question,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return vector.tolist()
        except Exception as exc:
            raise RuntimeError(f"Error generating query embedding: {exc}") from exc
