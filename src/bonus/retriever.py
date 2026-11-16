"""Retrieve source locations from a validated persisted search index."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..mandatory.bm25 import BM25Index
from ..mandatory.chunking import Chunk
from ..mandatory.models import MinimalSource
from .indexer import INDEX_FILENAME, MAX_CHUNK_SIZE
from .semantic import SemanticIndex


_ALLOWED_PICKLE_GLOBALS: dict[tuple[str, str], Any] = {
    ("src.mandatory.bm25", "BM25Index"): BM25Index,
    ("src.bm25", "BM25Index"): BM25Index,
    ("src.mandatory.chunking", "Chunk"): Chunk,
    ("src.chunking", "Chunk"): Chunk,
    ("src.bonus.semantic", "SemanticIndex"): SemanticIndex,
    ("src.semantic", "SemanticIndex"): SemanticIndex,
}
_CACHE_VERSION = 1
_QUERY_CACHE_FILENAME = "query_cache.json"
_MAX_QUERY_CACHE_ENTRIES = 256
_RETRIEVAL_MODES = frozenset({"bm25", "semantic", "hybrid"})


@dataclass(frozen=True)
class IndexDetails:
    """Validated data loaded from one persisted index."""

    chunks: list[Chunk]
    bm25: BM25Index
    semantic: SemanticIndex | None
    source_metadata: dict[str, tuple[int, int]] | None
    max_chunk_size: int | None


class _RestrictedUnpickler(pickle.Unpickler):
    """Load only the project classes required by a persisted index."""

    def find_class(self, module: str, name: str) -> Any:
        """Reject globals that could execute arbitrary imported code."""
        try:
            return _ALLOWED_PICKLE_GLOBALS[(module, name)]
        except KeyError as error:
            raise pickle.UnpicklingError(
                f"Disallowed pickle global: {module}.{name}"
            ) from error


_INDEX_CACHE: dict[tuple[str, int, int], IndexDetails] = {}


def _resolve_index_path(index_path: Path) -> Path:
    """Resolve either an index file or an index directory."""
    if index_path.is_dir():
        return index_path / INDEX_FILENAME
    return index_path


def _index_signature(index_path: Path) -> tuple[int, int]:
    """Return the metadata used to invalidate index and query caches."""
    stat = index_path.stat()
    return stat.st_mtime_ns, stat.st_size


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


def _validate_semantic_index(
    semantic: SemanticIndex,
    chunk_count: int,
) -> None:
    """Reject semantic vectors that cannot score the indexed chunks."""
    try:
        model_name = semantic.model_name
        dimension = semantic.dimension
        embeddings = semantic.embeddings
    except AttributeError as error:
        raise ValueError("Index contains invalid semantic metadata") from error

    if (
        not isinstance(model_name, str)
        or not model_name.strip()
        or type(dimension) is not int
        or dimension < 0
        or not isinstance(embeddings, list)
        or len(embeddings) != chunk_count
    ):
        raise ValueError("Index contains invalid semantic metadata")
    if embeddings and dimension <= 0:
        raise ValueError("Index contains invalid semantic metadata")
    for embedding in embeddings:
        if (
            not isinstance(embedding, list)
            or len(embedding) != dimension
            or not all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
                for value in embedding
            )
        ):
            raise ValueError("Index contains invalid semantic metadata")
        norm = math.sqrt(sum(float(value) ** 2 for value in embedding))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("Index contains invalid semantic metadata")


def _validate_source_metadata(
    source_metadata: dict[str, tuple[int, int]],
) -> None:
    """Validate file signatures used by incremental indexing."""
    if not isinstance(source_metadata, dict):
        raise ValueError("Index contains invalid source metadata")
    for file_path, signature in source_metadata.items():
        if (
            not isinstance(file_path, str)
            or not file_path
            or not isinstance(signature, tuple)
            or len(signature) != 2
            or type(signature[0]) is not int
            or type(signature[1]) is not int
            or signature[0] < 0
            or signature[1] < 0
        ):
            raise ValueError("Index contains invalid source metadata")


def load_index_details(index_path: Path) -> IndexDetails:
    """Load, validate, and cache all data from a persisted index."""
    saved_index_path = _resolve_index_path(index_path)
    if not saved_index_path.exists():
        raise FileNotFoundError(
            f"Index does not exist: {saved_index_path}"
        )
    try:
        stat_signature = _index_signature(saved_index_path)
    except OSError as error:
        raise ValueError(
            f"Unable to inspect index: {saved_index_path}"
        ) from error

    cache_key = (
        str(saved_index_path.resolve()),
        stat_signature[0],
        stat_signature[1],
    )
    cached = _INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

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

    semantic = index_data.get("semantic")
    if semantic is not None:
        if not isinstance(semantic, SemanticIndex):
            raise ValueError("Index contains invalid semantic metadata")
        _validate_semantic_index(semantic, len(chunks))

    source_metadata = index_data.get("source_metadata")
    if source_metadata is not None:
        _validate_source_metadata(source_metadata)

    max_chunk_size = index_data.get("max_chunk_size")
    if max_chunk_size is not None and (
        type(max_chunk_size) is not int
        or not 1 <= max_chunk_size <= MAX_CHUNK_SIZE
    ):
        raise ValueError("Index contains invalid max chunk size")

    details = IndexDetails(
        chunks=chunks,
        bm25=bm25,
        semantic=semantic,
        source_metadata=source_metadata,
        max_chunk_size=max_chunk_size,
    )
    for old_key in list(_INDEX_CACHE):
        if old_key[0] == cache_key[0] and old_key != cache_key:
            del _INDEX_CACHE[old_key]
    _INDEX_CACHE[cache_key] = details
    return details


def load_index(index_path: Path) -> tuple[list[Chunk], BM25Index]:
    """Load chunks and BM25 statistics from a persisted index."""
    details = load_index_details(index_path)
    return details.chunks, details.bm25


def _validate_retrieval_options(
    retrieval_mode: str,
    semantic_weight: float,
    use_cache: bool,
) -> None:
    """Validate retrieval and cache options before loading the index."""
    if (
        not isinstance(retrieval_mode, str)
        or retrieval_mode not in _RETRIEVAL_MODES
    ):
        raise ValueError(
            "retrieval_mode must be one of bm25, semantic, or hybrid"
        )
    if (
        isinstance(semantic_weight, bool)
        or not isinstance(semantic_weight, (int, float))
        or not math.isfinite(semantic_weight)
        or not 0 <= semantic_weight <= 1
    ):
        raise ValueError("semantic_weight must be finite and between 0 and 1")
    if type(use_cache) is not bool:
        raise ValueError("use_cache must be a boolean")


def _query_cache_path(index_path: Path) -> Path:
    """Return the cache file next to the persisted index."""
    return index_path.parent / _QUERY_CACHE_FILENAME


def _query_cache_key(
    query: str,
    k: int,
    retrieval_mode: str,
    semantic_weight: float,
) -> str:
    """Create a stable cache key for one retrieval request."""
    request = json.dumps(
        [query, k, retrieval_mode, semantic_weight],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(request).hexdigest()


def _read_query_cache(
    saved_index_path: Path,
    query: str,
    k: int,
    retrieval_mode: str,
    semantic_weight: float,
) -> list[MinimalSource] | None:
    """Return a validated cached result, ignoring stale or bad cache data."""
    cache_path = _query_cache_path(saved_index_path)
    try:
        with cache_path.open("r", encoding="utf-8") as cache_file:
            cache_data = json.load(cache_file)
        if not isinstance(cache_data, dict):
            return None
        if cache_data.get("version") != _CACHE_VERSION:
            return None
        if cache_data.get("index_signature") != list(
            _index_signature(saved_index_path)
        ):
            return None
        entries = cache_data.get("entries")
        if not isinstance(entries, dict):
            return None
        cached_sources = entries.get(
            _query_cache_key(query, k, retrieval_mode, semantic_weight)
        )
        if not isinstance(cached_sources, list):
            return None
        return [
            MinimalSource.model_validate(source)
            for source in cached_sources
        ]
    except Exception:
        return None


def _write_query_cache(
    saved_index_path: Path,
    query: str,
    k: int,
    retrieval_mode: str,
    semantic_weight: float,
    sources: list[MinimalSource],
) -> None:
    """Persist one result atomically, without making caching mandatory."""
    cache_path = _query_cache_path(saved_index_path)
    key = _query_cache_key(query, k, retrieval_mode, semantic_weight)
    try:
        current_signature = list(_index_signature(saved_index_path))
    except OSError:
        return
    entries: dict[str, Any] = {}
    try:
        with cache_path.open("r", encoding="utf-8") as cache_file:
            cache_data = json.load(cache_file)
        if (
            isinstance(cache_data, dict)
            and cache_data.get("version") == _CACHE_VERSION
            and cache_data.get("index_signature") == current_signature
            and isinstance(cache_data.get("entries"), dict)
        ):
            entries = cache_data["entries"]
    except Exception:
        entries = {}

    entries.pop(key, None)
    entries[key] = [source.model_dump(mode="json") for source in sources]
    while len(entries) > _MAX_QUERY_CACHE_ENTRIES:
        entries.pop(next(iter(entries)))

    cache_data = {
        "version": _CACHE_VERSION,
        "index_signature": current_signature,
        "entries": entries,
    }
    temporary_path = cache_path.with_name(
        f".{cache_path.name}.{os.getpid()}.tmp"
    )
    try:
        temporary_path.write_text(
            json.dumps(cache_data, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_path.replace(cache_path)
    except OSError:
        return


def _normalise_scores(scores: list[float]) -> list[float]:
    """Scale non-negative BM25 scores to the range 0–1."""
    maximum = max(scores, default=0.0)
    if maximum <= 0:
        return [0.0] * len(scores)
    return [max(score, 0.0) / maximum for score in scores]


def _normalise_semantic_scores(scores: list[float]) -> list[float]:
    """Scale semantic similarities to the range 0–1."""
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if math.isclose(minimum, maximum):
        return [1.0] * len(scores)
    return [
        (score - minimum) / (maximum - minimum)
        for score in scores
    ]


def _sources_from_indices(
    chunks: list[Chunk],
    indices: list[int],
) -> list[MinimalSource]:
    """Convert ranked chunk indices to evaluator-compatible source models."""
    return [
        MinimalSource(
            file_path=chunks[index].file_path,
            first_character_index=chunks[index].first_character_index,
            last_character_index=chunks[index].last_character_index,
        )
        for index in indices
    ]


def _rank_bm25_indices(scores: list[float], k: int) -> list[int]:
    """Return positive BM25 results in deterministic score order."""
    return [
        index
        for index in sorted(
            range(len(scores)),
            key=lambda index: (-scores[index], index),
        )
        if scores[index] > 0
    ][:k]


def retrieve_sources(
    query: str,
    k: int = 5,
    index_path: Path = Path("data/processed"),
    retrieval_mode: str = "bm25",
    semantic_weight: float = 0.5,
    use_cache: bool = True,
) -> list[MinimalSource]:
    """Return the highest-scoring source locations for a query."""
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    if type(k) is not int:
        raise ValueError("k must be an integer")
    if k < 0:
        raise ValueError("k must be non-negative")
    _validate_retrieval_options(
        retrieval_mode,
        semantic_weight,
        use_cache,
    )
    if k == 0 or not query.strip():
        return []

    saved_index_path = _resolve_index_path(index_path)
    if use_cache:
        cached_sources = _read_query_cache(
            saved_index_path,
            query,
            k,
            retrieval_mode,
            semantic_weight,
        )
        if cached_sources is not None:
            return cached_sources

    details = load_index_details(saved_index_path)
    chunks = details.chunks
    bm25_scores = details.bm25.get_scores(query)

    if retrieval_mode == "bm25" or (
        retrieval_mode == "hybrid" and semantic_weight == 0
    ):
        ranked_indices = _rank_bm25_indices(bm25_scores, k)
    else:
        semantic = details.semantic
        if semantic is None:
            raise ValueError(
                "This index has no semantic vectors; rebuild it with "
                "semantic_model_name"
            )
        semantic_scores = semantic.scores_for(query)
        if len(semantic_scores) != len(chunks):
            raise ValueError("Semantic scores do not match indexed chunks")

        if retrieval_mode == "semantic":
            ranked_indices = sorted(
                range(len(chunks)),
                key=lambda index: (-semantic_scores[index], index),
            )[:k]
        else:
            bm25_normalised = _normalise_scores(bm25_scores)
            semantic_normalised = _normalise_semantic_scores(
                semantic_scores,
            )
            combined_scores = [
                (1.0 - semantic_weight) * bm25_normalised[index]
                + semantic_weight * semantic_normalised[index]
                for index in range(len(chunks))
            ]
            ranked_indices = sorted(
                range(len(chunks)),
                key=lambda index: (-combined_scores[index], index),
            )[:k]

    sources = _sources_from_indices(chunks, ranked_indices)
    if use_cache:
        _write_query_cache(
            saved_index_path,
            query,
            k,
            retrieval_mode,
            semantic_weight,
            sources,
        )
    return sources
