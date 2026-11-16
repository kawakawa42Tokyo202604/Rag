"""Tests for optional semantic embedding persistence and scoring."""

import pytest

from src.bonus.semantic import SemanticIndex


def test_semantic_index_scores_normalised_vectors() -> None:
    """Score a query vector with cosine similarity."""
    index = SemanticIndex.from_embeddings(
        [[3.0, 0.0], [0.0, 2.0]],
        model_name="test-model",
    )

    scores = index.score_vector([10.0, 0.0])

    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)


def test_semantic_index_reuses_all_embeddings_without_loading_model() -> None:
    """Allow incremental builds to skip model loading for unchanged chunks."""
    index = SemanticIndex.from_documents(
        ["first", "second"],
        model_name="test-model",
        reused_embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )

    assert index.dimension == 2
    assert index.embeddings == [[1.0, 0.0], [0.0, 1.0]]


def test_semantic_index_rejects_invalid_reused_embedding_count() -> None:
    """Reject vectors that are not aligned with the document list."""
    with pytest.raises(ValueError, match="match the number"):
        SemanticIndex.from_documents(
            ["one", "two"],
            model_name="test-model",
            reused_embeddings=[[1.0, 0.0]],
        )


def test_semantic_index_rejects_zero_vectors() -> None:
    """Reject vectors that cannot be normalised for cosine search."""
    with pytest.raises(ValueError, match="non-zero norm"):
        SemanticIndex.from_embeddings(
            [[0.0, 0.0]],
            model_name="test-model",
        )
