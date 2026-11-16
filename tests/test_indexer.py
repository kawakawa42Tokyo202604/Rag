"""Tests for BM25 index creation and persistence."""

import pickle
from pathlib import Path

import pytest

from src.mandatory.bm25 import BM25Index
from src.mandatory.indexer import _iter_source_files, build_index
from src.mandatory.retriever import load_index


def test_build_index_persists_bm25_data(tmp_path: Path) -> None:
    """Persist chunks and BM25 statistics in the expected index file."""
    source_file = tmp_path / "example.py"
    source_file.write_text("trust_remote_code = False\n", encoding="utf-8")

    index_path = build_index(
        source_file,
        output_directory=tmp_path / "processed",
    )

    with index_path.open("rb") as index_file:
        index_data = pickle.load(index_file)

    assert isinstance(index_data["bm25"], BM25Index)
    assert index_data["chunks"]


def test_indexer_filters_binary_and_excluded_files(tmp_path: Path) -> None:
    """Index supported source files but skip binaries and cache directories."""
    source_file = tmp_path / "example.py"
    source_file.write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"not source text")
    cache_file = tmp_path / ".git" / "ignored.py"
    cache_file.parent.mkdir()
    cache_file.write_text("ignored = True\n", encoding="utf-8")

    files = list(_iter_source_files(tmp_path))

    assert files == [source_file]


def test_indexer_keeps_files_when_corpus_root_has_excluded_name(
    tmp_path: Path,
) -> None:
    """Apply excluded-directory rules relative to the corpus root."""
    corpus = tmp_path / "build"
    corpus.mkdir()
    source_file = corpus / "example.py"
    source_file.write_text("value = 1\n", encoding="utf-8")
    nested_excluded_file = corpus / "build" / "ignored.py"
    nested_excluded_file.parent.mkdir()
    nested_excluded_file.write_text("ignored = True\n", encoding="utf-8")

    files = list(_iter_source_files(corpus))

    assert files == [source_file]


def test_build_index_handles_empty_directory(tmp_path: Path) -> None:
    """Persist and reload an empty index without raising an exception."""
    corpus = tmp_path / "empty"
    corpus.mkdir()

    index_path = build_index(
        corpus,
        output_directory=tmp_path / "processed",
    )

    chunks, bm25 = load_index(index_path)
    assert chunks == []
    assert bm25.total_documents == 0


def test_build_index_rejects_chunks_larger_than_evaluator_limit(
    tmp_path: Path,
) -> None:
    """Prevent indexes that could emit source spans over 2000 characters."""
    source_file = tmp_path / "example.py"
    source_file.write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="between 1 and 2000"):
        build_index(
            source_file,
            output_directory=tmp_path / "processed",
            max_chunk_size=2001,
        )
