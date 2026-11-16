"""Optional transformer-based semantic embeddings for source retrieval."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math
from typing import Any


DEFAULT_SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_BATCH_SIZE = 16
_DEFAULT_MAX_LENGTH = 256


def _validate_model_name(model_name: str) -> None:
    """Reject model names that cannot identify an embedding model."""
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")


def _normalise_vector(vector: Sequence[float]) -> list[float]:
    """Validate and L2-normalise one embedding vector."""
    values: list[float] = []
    for value in vector:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("Embedding vectors must contain finite numbers")
        values.append(float(value))

    if not values:
        raise ValueError("Embedding vectors must not be empty")

    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding vectors must have a non-zero norm")
    return [value / norm for value in values]


class _TransformerEncoder:
    """Lazily load a transformer tokenizer and encoder model."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_length: int = _DEFAULT_MAX_LENGTH,
    ) -> None:
        _validate_model_name(model_name)
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if type(max_length) is not int or max_length <= 0:
            raise ValueError("max_length must be a positive integer")

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def _load_components(self) -> tuple[Any, Any, Any]:
        """Load the model only when the first embedding is requested."""
        if (
            self._tokenizer is not None
            and self._model is not None
            and self._torch is not None
        ):
            return self._tokenizer, self._model, self._torch

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
            model.eval()
        except Exception as error:
            raise RuntimeError(
                "Unable to load semantic model "
                f"{self.model_name}; download its weights first"
            ) from error

        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        return tokenizer, model, torch

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode texts with mean pooling and unit-length normalisation."""
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("texts must contain only strings")
        if not texts:
            return []

        tokenizer, model, torch = self._load_components()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            try:
                inputs = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                with torch.inference_mode():
                    outputs = model(**inputs)
                hidden_states = getattr(outputs, "last_hidden_state", None)
                attention_mask = inputs.get("attention_mask")
                if hidden_states is None or attention_mask is None:
                    raise RuntimeError(
                        "semantic model did not return token embeddings"
                    )
                mask = attention_mask.unsqueeze(-1).to(
                    hidden_states.dtype,
                )
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(
                    dim=1,
                ).clamp(min=1e-9)
                normalised = torch.nn.functional.normalize(
                    pooled,
                    p=2,
                    dim=1,
                )
                raw_vectors = normalised.detach().cpu().tolist()
            except Exception as error:
                raise RuntimeError(
                    "Unable to encode text with semantic model "
                    f"{self.model_name}"
                ) from error

            if not isinstance(raw_vectors, list):
                raise RuntimeError("semantic model returned invalid vectors")
            for raw_vector in raw_vectors:
                if not isinstance(raw_vector, list):
                    raise RuntimeError(
                        "semantic model returned invalid vectors"
                    )
                vectors.append(_normalise_vector(raw_vector))

        return vectors


_ENCODER_CACHE: dict[str, _TransformerEncoder] = {}


def get_semantic_encoder(model_name: str) -> _TransformerEncoder:
    """Return the process-local lazy encoder for a model name."""
    _validate_model_name(model_name)
    encoder = _ENCODER_CACHE.get(model_name)
    if encoder is None:
        encoder = _TransformerEncoder(model_name)
        _ENCODER_CACHE[model_name] = encoder
    return encoder


@dataclass(frozen=True)
class SemanticIndex:
    """Store normalised document embeddings and their model identity."""

    embeddings: list[list[float]]
    model_name: str
    dimension: int

    def __post_init__(self) -> None:
        _validate_model_name(self.model_name)
        if type(self.dimension) is not int or self.dimension < 0:
            raise ValueError("dimension must be a non-negative integer")
        if not isinstance(self.embeddings, list):
            raise ValueError("embeddings must be a list")
        if self.embeddings and self.dimension <= 0:
            raise ValueError("non-empty embeddings need a positive dimension")
        for embedding in self.embeddings:
            if not isinstance(embedding, list):
                raise ValueError("embeddings must contain lists")
            if len(embedding) != self.dimension:
                raise ValueError("embedding dimensions do not match")
            for value in embedding:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(
                        "embeddings must contain finite numbers"
                    )

    @classmethod
    def from_embeddings(
        cls,
        embeddings: Sequence[Sequence[float]],
        model_name: str,
    ) -> "SemanticIndex":
        """Create an index from vectors, normalising them for cosine search."""
        _validate_model_name(model_name)
        normalised = [_normalise_vector(embedding) for embedding in embeddings]
        dimension = len(normalised[0]) if normalised else 0
        return cls(
            embeddings=normalised,
            model_name=model_name,
            dimension=dimension,
        )

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[str],
        model_name: str = DEFAULT_SEMANTIC_MODEL_NAME,
        reused_embeddings: Sequence[Sequence[float] | None] | None = None,
    ) -> "SemanticIndex":
        """Encode documents, reusing unchanged vectors when given."""
        _validate_model_name(model_name)
        document_list = list(documents)
        if not all(isinstance(document, str) for document in document_list):
            raise ValueError("documents must contain only strings")
        if reused_embeddings is not None and len(reused_embeddings) != len(
            document_list
        ):
            raise ValueError(
                "reused_embeddings must match the number of documents"
            )

        vectors: list[list[float] | None] = []
        missing_documents: list[str] = []
        missing_positions: list[int] = []
        for position, document in enumerate(document_list):
            reused = (
                None
                if reused_embeddings is None
                else reused_embeddings[position]
            )
            if reused is None:
                vectors.append(None)
                missing_documents.append(document)
                missing_positions.append(position)
            else:
                vectors.append(_normalise_vector(reused))

        if missing_documents:
            encoded = get_semantic_encoder(model_name).encode(
                missing_documents,
            )
            if len(encoded) != len(missing_positions):
                raise RuntimeError(
                    "semantic model returned the wrong number of vectors"
                )
            for position, vector in zip(missing_positions, encoded):
                vectors[position] = _normalise_vector(vector)

        complete_vectors = [
            vector for vector in vectors if vector is not None
        ]
        if len(complete_vectors) != len(document_list):
            raise RuntimeError("semantic embeddings are incomplete")
        dimension = len(complete_vectors[0]) if complete_vectors else 0
        return cls(
            embeddings=complete_vectors,
            model_name=model_name,
            dimension=dimension,
        )

    def score_vector(self, query_vector: Sequence[float]) -> list[float]:
        """Return cosine similarities for a supplied query vector."""
        if not self.embeddings:
            return []
        query = _normalise_vector(query_vector)
        if len(query) != self.dimension:
            raise ValueError("query embedding dimension does not match index")
        return [
            sum(value * query_value for value, query_value in zip(
                embedding,
                query,
            ))
            for embedding in self.embeddings
        ]

    def scores_for(self, query: str) -> list[float]:
        """Encode a query and return cosine similarities to all chunks."""
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if not self.embeddings:
            return []
        vectors = get_semantic_encoder(self.model_name).encode([query])
        if len(vectors) != 1:
            raise RuntimeError(
                "semantic model returned an invalid query vector"
            )
        return self.score_vector(vectors[0])
