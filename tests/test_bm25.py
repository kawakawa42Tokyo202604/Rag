"""Tests for the BM25 implementation."""

import math
from typing import cast

import pytest

from src.mandatory.bm25 import BM25Index, tokenize


def test_tokenizer_preserves_code_identifiers() -> None:
    """Keep underscore-separated identifiers as searchable tokens."""
    assert "trust_remote_code" in tokenize("trust_remote_code = False")


def test_bm25_ranks_matching_document_first() -> None:
    """Rank the document containing the queried identifier highest."""
    index = BM25Index.from_documents(
        [
            "trust_remote_code is disabled by default",
            "tensor parallel size controls distribution",
        ]
    )

    scores = index.get_scores("default trust_remote_code")

    assert scores[0] > scores[1]


def test_empty_bm25_index_returns_empty_scores() -> None:
    """Handle an empty corpus without raising an exception."""
    assert BM25Index.from_documents([]).get_scores("query") == []


def test_bm25_uses_tuned_default_parameters() -> None:
    """Keep the tuned BM25 defaults stable."""
    index = BM25Index.from_documents(["term"])

    assert index.k1 == pytest.approx(1.2)
    assert index.b == pytest.approx(1.0)


@pytest.mark.parametrize(
    "k1",
    [
        math.nan,
        math.inf,
        -math.inf,
        cast(float, True),
        cast(float, "1.2"),
    ],
)
def test_bm25_rejects_non_finite_k1(k1: float) -> None:
    """Reject invalid BM25 saturation parameters."""
    with pytest.raises(ValueError, match="finite"):
        BM25Index.from_documents(["term"], k1=k1)


@pytest.mark.parametrize(
    "b",
    [
        math.nan,
        math.inf,
        -math.inf,
        -0.1,
        1.1,
        cast(float, True),
        cast(float, "1.0"),
    ],
)
def test_bm25_rejects_invalid_length_normalization(b: float) -> None:
    """Reject invalid BM25 length-normalization parameters."""
    with pytest.raises(ValueError, match="finite"):
        BM25Index.from_documents(["term"], b=b)
