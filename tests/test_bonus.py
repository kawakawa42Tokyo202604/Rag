"""Regression tests for bonus retrieval and incremental indexing features."""

from pathlib import Path
import pickle

import pytest

from src.mandatory.bm25 import BM25Index
from src.mandatory.chunking import Chunk, chunk_file as original_chunk_file
from src.bonus.indexer import build_index, _iter_source_files
from src.bonus.retriever import retrieve_sources
from src.bonus.semantic import SemanticIndex


def test_incremental_index_reuses_unchanged_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-chunk only changed or newly added files."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    unchanged = corpus / "unchanged.py"
    changed = corpus / "changed.py"
    unchanged.write_text("unchanged = True\n", encoding="utf-8")
    changed.write_text("changed = 1\n", encoding="utf-8")
    processed = tmp_path / "processed"
    build_index(corpus, output_directory=processed)

    changed.write_text("changed_value = 12345\n", encoding="utf-8")
    new_file = corpus / "new.py"
    new_file.write_text("new_value = True\n", encoding="utf-8")
    chunked_files: list[Path] = []

    def tracking_chunk_file(
        file_path: Path,
        max_chunk_size: int,
    ) -> list[Chunk]:
        chunked_files.append(file_path)
        return original_chunk_file(file_path, max_chunk_size)

    monkeypatch.setattr(
        "src.bonus.indexer.chunk_file",
        tracking_chunk_file,
    )
    index_path = build_index(
        corpus,
        output_directory=processed,
        incremental=True,
    )

    assert unchanged not in chunked_files
    assert changed in chunked_files
    assert new_file in chunked_files
    with index_path.open("rb") as index_file:
        index_data = pickle.load(index_file)
    indexed_paths = {chunk.file_path for chunk in index_data["chunks"]}
    assert str(unchanged) in indexed_paths
    assert str(changed) in indexed_paths
    assert str(new_file) in indexed_paths

    unchanged.unlink()
    build_index(corpus, output_directory=processed, incremental=True)
    with index_path.open("rb") as index_file:
        index_data = pickle.load(index_file)
    indexed_paths = {chunk.file_path for chunk in index_data["chunks"]}
    assert str(unchanged) not in indexed_paths


def test_bonus_indexer_keeps_corpus_root_named_build(tmp_path: Path) -> None:
    """Do not treat the corpus root as an excluded child directory."""
    corpus = tmp_path / "build"
    corpus.mkdir()
    source_file = corpus / "example.py"
    source_file.write_text("value = 1\n", encoding="utf-8")

    assert list(_iter_source_files(corpus)) == [source_file]


def _write_semantic_index(directory: Path) -> Path:
    """Write a small deterministic semantic index for ranking tests."""
    chunks = [
        Chunk("cat", "cat.py", 0, 3),
        Chunk("dog", "dog.py", 0, 3),
    ]
    index_data = {
        "chunks": chunks,
        "bm25": BM25Index.from_documents(chunk.text for chunk in chunks),
        "semantic": SemanticIndex.from_embeddings(
            [[1.0, 0.0], [0.0, 1.0]],
            model_name="test-model",
        ),
        "max_chunk_size": 2000,
        "source_metadata": {},
    }
    directory.mkdir(parents=True, exist_ok=True)
    index_path = directory / "index.pkl"
    with index_path.open("wb") as index_file:
        pickle.dump(index_data, index_file)
    return index_path


def test_semantic_and_hybrid_retrieval_use_semantic_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rank an out-of-vocabulary query with the persisted embeddings."""
    index_path = _write_semantic_index(tmp_path / "processed")

    def fake_scores(self: SemanticIndex, query: str) -> list[float]:
        assert query == "feline"
        return self.score_vector([1.0, 0.0])

    monkeypatch.setattr(SemanticIndex, "scores_for", fake_scores)

    semantic_sources = retrieve_sources(
        "feline",
        k=1,
        index_path=index_path,
        retrieval_mode="semantic",
        use_cache=False,
    )
    hybrid_sources = retrieve_sources(
        "feline",
        k=1,
        index_path=index_path,
        retrieval_mode="hybrid",
        semantic_weight=0.8,
        use_cache=False,
    )

    assert semantic_sources[0].file_path == "cat.py"
    assert hybrid_sources[0].file_path == "cat.py"


def test_hybrid_zero_weight_uses_bm25_without_semantic_index(
    tmp_path: Path,
) -> None:
    """Treat a zero semantic weight as a BM25-only retrieval request."""
    source = tmp_path / "example.py"
    source.write_text("trust_remote_code = False\n", encoding="utf-8")
    processed = tmp_path / "processed"
    build_index(source, output_directory=processed)

    sources = retrieve_sources(
        "trust_remote_code",
        index_path=processed,
        retrieval_mode="hybrid",
        semantic_weight=0,
        use_cache=False,
    )

    assert sources[0].file_path == str(source)


def test_retrieval_writes_and_reads_query_cache(tmp_path: Path) -> None:
    """Persist normal retrieval results for repeated requests."""
    source = tmp_path / "example.py"
    source.write_text("trust_remote_code = False\n", encoding="utf-8")
    processed = tmp_path / "processed"
    build_index(source, output_directory=processed)

    first = retrieve_sources("trust_remote_code", index_path=processed)
    second = retrieve_sources("trust_remote_code", index_path=processed)

    assert first == second
    assert (processed / "query_cache.json").exists()


def test_semantic_retrieval_requires_semantic_index(tmp_path: Path) -> None:
    """Explain how to enable semantic retrieval for a BM25-only index."""
    source = tmp_path / "example.py"
    source.write_text("value = 1\n", encoding="utf-8")
    processed = tmp_path / "processed"
    build_index(source, output_directory=processed)

    with pytest.raises(ValueError, match="no semantic vectors"):
        retrieve_sources(
            "value",
            index_path=processed,
            retrieval_mode="semantic",
            use_cache=False,
        )
