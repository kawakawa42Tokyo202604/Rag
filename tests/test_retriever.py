"""Tests for BM25 source retrieval."""

import math
import pickle
from pathlib import Path
from typing import cast

import pytest

from src.mandatory.bm25 import BM25Index
from src.mandatory.chunking import Chunk
from src.mandatory.indexer import build_index
from src.mandatory.retriever import load_index, retrieve_sources


def test_retrieve_sources_returns_matching_locations(tmp_path: Path) -> None:
    """Return source metadata ranked by BM25 relevance."""
    source_file = tmp_path / "example.py"
    source_file.write_text("trust_remote_code = False\n", encoding="utf-8")
    processed_directory = tmp_path / "processed"
    build_index(source_file, output_directory=processed_directory)

    sources = retrieve_sources(
        "trust_remote_code",
        k=1,
        index_path=processed_directory,
    )

    assert len(sources) == 1
    assert sources[0].file_path == str(source_file)


def test_retrieve_sources_handles_zero_k_and_empty_query(
    tmp_path: Path,
) -> None:
    """Return no results for queries that cannot retrieve anything."""
    assert retrieve_sources("query", k=0, index_path=tmp_path) == []
    assert retrieve_sources("", k=5, index_path=tmp_path) == []


@pytest.mark.parametrize("query", [cast(str, 123), cast(str, None)])
def test_retrieve_sources_rejects_non_string_query(
    tmp_path: Path,
    query: str,
) -> None:
    """Convert Fire's non-string positional values into a clear error."""
    with pytest.raises(ValueError, match="query must be a string"):
        retrieve_sources(query, k=1, index_path=tmp_path)


@pytest.mark.parametrize("k", [-1, cast(int, 1.5), cast(int, True)])
def test_retrieve_sources_rejects_invalid_k(
    tmp_path: Path,
    k: int,
) -> None:
    """Reject negative and non-integer retrieval counts before indexing."""
    with pytest.raises(ValueError, match="k must"):
        retrieve_sources("query", k=k, index_path=tmp_path)


def test_retrieve_sources_returns_empty_for_unknown_query(
    tmp_path: Path,
) -> None:
    """Return no sources when the query has no indexed terms."""
    source_file = tmp_path / "example.py"
    source_file.write_text("trust_remote_code = False\n", encoding="utf-8")
    processed_directory = tmp_path / "processed"
    build_index(source_file, output_directory=processed_directory)

    assert retrieve_sources(
        "term_not_in_index",
        k=5,
        index_path=processed_directory,
    ) == []


def test_retrieve_sources_handles_k_larger_than_index(
    tmp_path: Path,
) -> None:
    """Return all matching chunks when k exceeds the index size."""
    source_file = tmp_path / "example.py"
    source_file.write_text("trust_remote_code = False\n", encoding="utf-8")
    processed_directory = tmp_path / "processed"
    build_index(source_file, output_directory=processed_directory)

    sources = retrieve_sources(
        "trust_remote_code",
        k=100,
        index_path=processed_directory,
    )

    assert len(sources) == 1
    assert all(
        source.last_character_index - source.first_character_index <= 2000
        for source in sources
    )


def test_retrieve_sources_orders_equal_scores_deterministically(
    tmp_path: Path,
) -> None:
    """Use source traversal order as a stable tie-breaker."""
    first_file = tmp_path / "a.py"
    second_file = tmp_path / "b.py"
    first_file.write_text("shared = True\n", encoding="utf-8")
    second_file.write_text("shared = True\n", encoding="utf-8")
    processed_directory = tmp_path / "processed"
    build_index(tmp_path, output_directory=processed_directory)

    sources = retrieve_sources(
        "shared",
        k=2,
        index_path=processed_directory,
    )

    assert [source.file_path for source in sources] == [
        str(first_file),
        str(second_file),
    ]


def test_load_index_rejects_corrupt_pickle(tmp_path: Path) -> None:
    """Convert an invalid pickle into a clear validation error."""
    index_path = tmp_path / "index.pkl"
    index_path.write_bytes(b"not a pickle")

    with pytest.raises(ValueError, match="Unable to load index"):
        load_index(index_path)


def test_load_index_rejects_mismatched_bm25_arrays(tmp_path: Path) -> None:
    """Reject an index whose BM25 arrays cannot score its chunks."""
    index_path = tmp_path / "index.pkl"
    index_data = {
        "chunks": [Chunk("term", "example.py", 0, 4)],
        "bm25": BM25Index(
            document_term_frequencies=[{"term": 1}],
            document_frequencies={"term": 1},
            document_lengths=[],
            average_document_length=1.0,
            total_documents=1,
        ),
    }
    with index_path.open("wb") as index_file:
        pickle.dump(index_data, index_file)

    with pytest.raises(ValueError, match="BM25 arrays"):
        load_index(index_path)


def test_load_index_rejects_inconsistent_bm25_statistics(
    tmp_path: Path,
) -> None:
    """Reject BM25 statistics whose document frequencies are inconsistent."""
    index_path = tmp_path / "index.pkl"
    index_data = {
        "chunks": [Chunk("term", "example.py", 0, 4)],
        "bm25": BM25Index(
            document_term_frequencies=[{"term": 1}],
            document_frequencies={},
            document_lengths=[1],
            average_document_length=1.0,
            total_documents=1,
        ),
    }
    with index_path.open("wb") as index_file:
        pickle.dump(index_data, index_file)

    with pytest.raises(ValueError, match="document frequencies"):
        load_index(index_path)


def test_load_index_rejects_disallowed_pickle_globals(tmp_path: Path) -> None:
    """Reject pickle globals outside the classes used by the index format."""
    index_path = tmp_path / "index.pkl"
    with index_path.open("wb") as index_file:
        pickle.dump(Path("not-an-index"), index_file)

    with pytest.raises(ValueError, match="Unable to load index"):
        load_index(index_path)


@pytest.mark.parametrize(
    "chunk",
    [
        Chunk("term", "example.py", 0, 5),
        Chunk("x" * 2001, "example.py", 0, 2001),
    ],
)
def test_load_index_rejects_invalid_chunk_metadata(
    tmp_path: Path,
    chunk: Chunk,
) -> None:
    """Reject inconsistent or evaluator-incompatible source ranges."""
    index_path = tmp_path / "index.pkl"
    index_data = {
        "chunks": [chunk],
        "bm25": BM25Index.from_documents([chunk.text]),
    }
    with index_path.open("wb") as index_file:
        pickle.dump(index_data, index_file)

    with pytest.raises(ValueError, match="invalid chunks"):
        load_index(index_path)


@pytest.mark.parametrize("total_documents", [cast(int, 1.0), cast(int, True)])
def test_load_index_rejects_non_integer_document_count(
    tmp_path: Path,
    total_documents: int,
) -> None:
    """Reject document counts that cannot be used for score arrays."""
    index_path = tmp_path / "index.pkl"
    index_data = {
        "chunks": [Chunk("term", "example.py", 0, 4)],
        "bm25": BM25Index(
            document_term_frequencies=[{"term": 1}],
            document_frequencies={"term": 1},
            document_lengths=[1],
            average_document_length=1.0,
            total_documents=total_documents,
        ),
    }
    with index_path.open("wb") as index_file:
        pickle.dump(index_data, index_file)

    with pytest.raises(ValueError, match="document count"):
        load_index(index_path)


@pytest.mark.parametrize(
    ("k1", "b", "average_document_length"),
    [
        (math.nan, 1.0, 1.0),
        (1.2, math.nan, 1.0),
        (1.2, 1.1, 1.0),
        (1.2, 1.0, math.nan),
        (1.2, 1.0, 2.0),
    ],
)
def test_load_index_rejects_invalid_bm25_parameters(
    tmp_path: Path,
    k1: float,
    b: float,
    average_document_length: float,
) -> None:
    """Reject invalid scalar BM25 statistics from persisted indexes."""
    index_path = tmp_path / "index.pkl"
    index_data = {
        "chunks": [Chunk("term", "example.py", 0, 4)],
        "bm25": BM25Index(
            document_term_frequencies=[{"term": 1}],
            document_frequencies={"term": 1},
            document_lengths=[1],
            average_document_length=average_document_length,
            total_documents=1,
            k1=k1,
            b=b,
        ),
    }
    with index_path.open("wb") as index_file:
        pickle.dump(index_data, index_file)

    with pytest.raises(ValueError, match="BM25"):
        load_index(index_path)
