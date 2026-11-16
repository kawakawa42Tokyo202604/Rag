"""Build and persist a BM25 index for the source corpus."""

from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

from ..mandatory.bm25 import BM25Index
from ..mandatory.chunking import Chunk, chunk_file
from .semantic import SemanticIndex


INDEX_FILENAME = "index.pkl"
MAX_CHUNK_SIZE = 2000
_SOURCE_SUFFIXES = frozenset(
    {
        ".bash",
        ".c",
        ".cfg",
        ".cc",
        ".cjs",
        ".cmake",
        ".cpp",
        ".css",
        ".cu",
        ".cuh",
        ".h",
        ".hpp",
        ".hxx",
        ".html",
        ".in",
        ".ini",
        ".js",
        ".json",
        ".jsonl",
        ".jinja",
        ".markdown",
        ".md",
        ".patch",
        ".py",
        ".pyi",
        ".rst",
        ".sh",
        ".toml",
        ".ts",
        ".txt",
        ".tsx",
        ".yaml",
        ".yml",
    }
)
_SOURCE_FILENAMES = frozenset(
    {
        "DCO",
        "Dockerfile",
        "LICENSE",
        "Makefile",
    }
)
_EXCLUDED_FILE_NAMES = frozenset({"index.pkl", "query_cache.json"})
_EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
        ".uv-cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    }
)


@dataclass(frozen=True)
class _PreviousIndex:
    """Subset of a previous index needed for incremental rebuilding."""

    chunks_by_file: dict[str, list[Chunk]]
    embeddings_by_chunk: dict[tuple[str, int, int, str], list[float]]
    source_metadata: dict[str, tuple[int, int]]
    semantic_model_name: str | None


def _is_source_file(
    file_path: Path,
    corpus_root: Path | None = None,
) -> bool:
    """Return whether a file should be included in a corpus index."""
    if file_path.name in _EXCLUDED_FILE_NAMES:
        return False
    path_to_check = file_path
    if corpus_root is not None:
        try:
            path_to_check = file_path.relative_to(corpus_root)
        except ValueError:
            pass
    if any(
        part in _EXCLUDED_DIRECTORY_NAMES for part in path_to_check.parts
    ):
        return False
    return (
        file_path.name in _SOURCE_FILENAMES
        or file_path.suffix.lower() in _SOURCE_SUFFIXES
    )


def _iter_source_files(corpus_path: Path) -> Iterator[Path]:
    """Yield files in a corpus in deterministic path order."""
    if corpus_path.is_file():
        yield corpus_path
        return

    files = sorted(
        path
        for path in corpus_path.rglob("*")
        if path.is_file() and _is_source_file(path, corpus_path)
    )
    yield from files


def _project_relative_path(file_path: Path) -> str:
    """Return a source path in the format expected by the evaluator."""
    if not file_path.is_absolute():
        return file_path.as_posix()
    try:
        return file_path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return file_path.as_posix()


def _source_signature(file_path: Path) -> tuple[int, int]:
    """Return the file signature used to detect unchanged source files."""
    stat = file_path.stat()
    return stat.st_mtime_ns, stat.st_size


def _chunk_key(chunk: Chunk) -> tuple[str, int, int, str]:
    """Return a content-aware key for reusing one semantic embedding."""
    return (
        chunk.file_path,
        chunk.first_character_index,
        chunk.last_character_index,
        chunk.text,
    )


def _relative_chunks(
    chunks: list[Chunk],
    relative_path: str,
) -> list[Chunk]:
    """Set the persisted path on chunks returned by the file chunker."""
    return [
        Chunk(
            text=chunk.text,
            file_path=relative_path,
            first_character_index=chunk.first_character_index,
            last_character_index=chunk.last_character_index,
        )
        for chunk in chunks
    ]


def _load_previous_index(
    index_path: Path,
    max_chunk_size: int,
) -> _PreviousIndex | None:
    """Load reusable data, or return ``None`` for an incompatible index."""
    if not index_path.exists():
        return None

    try:
        from .retriever import load_index_details

        previous = load_index_details(index_path)
    except (OSError, ValueError):
        return None

    if (
        previous.max_chunk_size != max_chunk_size
        or previous.source_metadata is None
    ):
        return None

    chunks_by_file: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in previous.chunks:
        chunks_by_file[chunk.file_path].append(chunk)

    embeddings_by_chunk: dict[tuple[str, int, int, str], list[float]] = {}
    semantic_model_name: str | None = None
    if (
        previous.semantic is not None
        and len(previous.semantic.embeddings) == len(previous.chunks)
    ):
        semantic_model_name = previous.semantic.model_name
        for chunk, embedding in zip(
            previous.chunks,
            previous.semantic.embeddings,
        ):
            embeddings_by_chunk[_chunk_key(chunk)] = embedding

    return _PreviousIndex(
        chunks_by_file=dict(chunks_by_file),
        embeddings_by_chunk=embeddings_by_chunk,
        source_metadata=previous.source_metadata,
        semantic_model_name=semantic_model_name,
    )


def build_index(
    corpus_path: Path,
    output_directory: Path = Path("data/processed"),
    max_chunk_size: int = MAX_CHUNK_SIZE,
    semantic_model_name: str | None = None,
    incremental: bool = False,
) -> Path:
    """Chunk a corpus, build BM25 statistics, and save the resulting index.

    Args:
        corpus_path: File or directory containing the source corpus.
        output_directory: Directory in which to save the index.
        max_chunk_size: Maximum number of characters in one chunk.
        semantic_model_name: Optional transformer model for semantic search.
        incremental: Reuse chunks for files whose size and mtime are unchanged.

    Returns:
        Path to the saved pickle index.

    Raises:
        FileNotFoundError: If ``corpus_path`` does not exist.
        ValueError: If ``max_chunk_size`` is outside the supported range.
    """
    if type(max_chunk_size) is not int or not (
        1 <= max_chunk_size <= MAX_CHUNK_SIZE
    ):
        raise ValueError(
            f"max_chunk_size must be an integer between 1 and "
            f"{MAX_CHUNK_SIZE}"
        )
    if type(incremental) is not bool:
        raise ValueError("incremental must be a boolean")
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus does not exist: {corpus_path}")
    if not corpus_path.is_file() and not corpus_path.is_dir():
        raise ValueError(f"Corpus is not a file or directory: {corpus_path}")

    output_directory.mkdir(parents=True, exist_ok=True)
    index_path = output_directory / INDEX_FILENAME
    previous = (
        _load_previous_index(index_path, max_chunk_size)
        if incremental
        else None
    )

    chunks: list[Chunk] = []
    source_metadata: dict[str, tuple[int, int]] = {}
    reused_embeddings: list[list[float] | None] = []
    source_files = list(_iter_source_files(corpus_path))
    for source_file in tqdm(source_files, desc="Indexing files"):
        relative_path = _project_relative_path(source_file)
        try:
            signature = _source_signature(source_file)
        except OSError as error:
            tqdm.write(f"Skipping {source_file}: {error}")
            continue

        reuse_file = bool(
            previous is not None
            and previous.source_metadata.get(relative_path) == signature
            and relative_path in previous.source_metadata
        )
        if reuse_file and previous is not None:
            file_chunks = previous.chunks_by_file.get(relative_path, [])
        else:
            try:
                file_chunks = _relative_chunks(
                    chunk_file(source_file, max_chunk_size),
                    relative_path,
                )
            except (OSError, UnicodeError) as error:
                tqdm.write(f"Skipping {source_file}: {error}")
                continue

        try:
            source_metadata[relative_path] = _source_signature(source_file)
        except OSError:
            pass

        chunks.extend(file_chunks)
        if semantic_model_name is not None:
            can_reuse_embeddings = bool(
                reuse_file
                and previous is not None
                and previous.semantic_model_name == semantic_model_name
            )
            reused_embeddings.extend(
                previous.embeddings_by_chunk.get(_chunk_key(chunk))
                if can_reuse_embeddings and previous is not None
                else None
                for chunk in file_chunks
            )

    bm25 = BM25Index.from_documents(chunk.text for chunk in chunks)

    semantic = (
        None
        if semantic_model_name is None
        else SemanticIndex.from_documents(
            (chunk.text for chunk in chunks),
            model_name=semantic_model_name,
            reused_embeddings=reused_embeddings,
        )
    )

    index_data = {
        "chunks": chunks,
        "bm25": bm25,
        "max_chunk_size": max_chunk_size,
        "semantic": semantic,
        "source_metadata": source_metadata,
    }
    with index_path.open("wb") as index_file:
        pickle.dump(index_data, index_file)

    return index_path
