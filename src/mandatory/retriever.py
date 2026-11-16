"""Retrieve source locations from a validated BM25 index."""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any

from .bm25 import BM25Index
from .chunking import Chunk
from .indexer import INDEX_FILENAME, MAX_CHUNK_SIZE
from .models import MinimalSource


_ALLOWED_PICKLE_GLOBALS: dict[tuple[str, str], Any] = {
    ("src.mandatory.bm25", "BM25Index"): BM25Index,
    ("src.bm25", "BM25Index"): BM25Index,
    ("src.mandatory.chunking", "Chunk"): Chunk,
    ("src.chunking", "Chunk"): Chunk,
}


class _RestrictedUnpickler(pickle.Unpickler):
    """Load only the project classes required by a BM25 index."""

    def find_class(self, module: str, name: str) -> Any:
        """Reject globals that could execute arbitrary imported code."""
        try:
            return _ALLOWED_PICKLE_GLOBALS[(module, name)]
        except KeyError as error:
            raise pickle.UnpicklingError(
                f"Disallowed pickle global: {module}.{name}"
            ) from error


def _resolve_index_path(index_path: Path) -> Path:
    """Resolve either an index file or an index directory."""
    if index_path.is_dir():
        return index_path / INDEX_FILENAME
    return index_path


def _validate_bm25_index(bm25: BM25Index, chunk_count: int) -> None:
    """Reject BM25 statistics that cannot score the indexed chunks."""
    if type(bm25.total_documents) is not int or bm25.total_documents < 0:
        raise ValueError("Index contains invalid BM25 document count")
    if bm25.total_documents != chunk_count:
        raise ValueError("Index chunks and BM25 statistics do not match")
    if (
        not isinstance(bm25.document_term_frequencies, list)
        or not isinstance(bm25.document_lengths, list)
        or len(bm25.document_term_frequencies) != chunk_count
        or len(bm25.document_lengths) != chunk_count
    ):
        raise ValueError("Index BM25 arrays do not match chunks")
    if not isinstance(bm25.document_frequencies, dict):
        raise ValueError("Index contains invalid BM25 statistics")
    if not all(
        isinstance(frequencies, dict)
        and all(
            isinstance(term, str)
            and type(frequency) is int
            and frequency > 0
            for term, frequency in frequencies.items()
        )
        for frequencies in bm25.document_term_frequencies
    ):
        raise ValueError("Index contains invalid BM25 statistics")
    if not all(
        type(document_length) is int and document_length >= 0
        for document_length in bm25.document_lengths
    ):
        raise ValueError("Index contains invalid BM25 statistics")
    if not all(
        isinstance(term, str)
        and type(document_frequency) is int
        and 0 <= document_frequency <= chunk_count
        for term, document_frequency in bm25.document_frequencies.items()
    ):
        raise ValueError("Index contains invalid BM25 statistics")

    expected_document_frequencies: dict[str, int] = {}
    for frequencies, document_length in zip(
        bm25.document_term_frequencies,
        bm25.document_lengths,
    ):
        if sum(frequencies.values()) != document_length:
            raise ValueError(
                "Index BM25 document lengths do not match term frequencies"
            )
        for term in frequencies:
            expected_document_frequencies[term] = (
                expected_document_frequencies.get(term, 0) + 1
            )
    if expected_document_frequencies != bm25.document_frequencies:
        raise ValueError(
            "Index BM25 document frequencies do not match term frequencies"
        )

    if (
        type(bm25.k1) not in (int, float)
        or not math.isfinite(bm25.k1)
        or bm25.k1 < 0
        or type(bm25.b) not in (int, float)
        or not math.isfinite(bm25.b)
        or not 0 <= bm25.b <= 1
        or type(bm25.average_document_length) not in (int, float)
        or not math.isfinite(bm25.average_document_length)
        or bm25.average_document_length < 0
    ):
        raise ValueError("Index contains invalid BM25 parameters")
    expected_average_document_length = (
        sum(bm25.document_lengths) / chunk_count
        if chunk_count
        else 0.0
    )
    if not math.isclose(
        bm25.average_document_length,
        expected_average_document_length,
    ):
        raise ValueError(
            "Index BM25 average document length does not match "
            "document lengths"
        )


def _validate_chunks(chunks: list[Chunk]) -> None:
    """Reject chunks with unusable source metadata or text ranges."""
    for chunk in chunks:
        try:
            text = chunk.text
            file_path = chunk.file_path
            first = chunk.first_character_index
            last = chunk.last_character_index
        except AttributeError as error:
            raise ValueError("Index contains invalid chunks") from error

        if (
            not isinstance(text, str)
            or not isinstance(file_path, str)
            or not file_path
            or type(first) is not int
            or type(last) is not int
            or first < 0
            or last <= first
            or last - first > MAX_CHUNK_SIZE
            or len(text) != last - first
        ):
            raise ValueError("Index contains invalid chunks")


def load_index(index_path: Path) -> tuple[list[Chunk], BM25Index]:
    """Load and validate chunks and BM25 statistics from an index."""
    saved_index_path = _resolve_index_path(index_path)
    if not saved_index_path.exists():
        raise FileNotFoundError(
            f"Index does not exist: {saved_index_path}"
        )

    try:
        with saved_index_path.open("rb") as index_file:
            index_data = _RestrictedUnpickler(index_file).load()
    except (
        AttributeError,
        EOFError,
        ImportError,
        IndexError,
        OSError,
        OverflowError,
        TypeError,
        ValueError,
        pickle.UnpicklingError,
    ) as error:
        raise ValueError(
            f"Unable to load index: {saved_index_path}"
        ) from error

    if not isinstance(index_data, dict):
        raise ValueError("Index must contain a dictionary")
    chunks = index_data.get("chunks")
    bm25 = index_data.get("bm25")
    if not isinstance(chunks, list) or not all(
        isinstance(chunk, Chunk) for chunk in chunks
    ):
        raise ValueError("Index contains invalid chunks")
    if not isinstance(bm25, BM25Index):
        raise ValueError("Index is not a BM25 index")
    _validate_chunks(chunks)
    try:
        _validate_bm25_index(bm25, len(chunks))
    except AttributeError as error:
        raise ValueError("Index contains invalid BM25 statistics") from error
    return chunks, bm25


def retrieve_sources(
    query: str,
    k: int = 5,
    index_path: Path = Path("data/processed"),
) -> list[MinimalSource]:
    """Return the highest-scoring BM25 source locations for a query."""
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    if type(k) is not int:
        raise ValueError("k must be an integer")
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0 or not query.strip():
        return []

    chunks, bm25 = load_index(index_path)
    scores = bm25.get_scores(query)
    indices = [
        index
        for index in sorted(
            range(len(chunks)),
            key=lambda index: (-scores[index], index),
        )
        if scores[index] > 0
    ][:k]
    return [
        MinimalSource(
            file_path=chunks[index].file_path,
            first_character_index=chunks[index].first_character_index,
            last_character_index=chunks[index].last_character_index,
        )
        for index in indices
    ]
