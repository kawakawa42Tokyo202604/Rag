"""Build and persist the mandatory BM25 index."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

from .bm25 import BM25Index
from .chunking import Chunk, chunk_file


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
_SOURCE_FILENAMES = frozenset({"DCO", "Dockerfile", "LICENSE", "Makefile"})
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
        part in _EXCLUDED_DIRECTORY_NAMES
        for part in path_to_check.parts
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


def build_index(
    corpus_path: Path,
    output_directory: Path = Path("data/processed"),
    max_chunk_size: int = MAX_CHUNK_SIZE,
) -> Path:
    """Chunk a corpus, build BM25 statistics, and save the resulting index."""
    if type(max_chunk_size) is not int or not (
        1 <= max_chunk_size <= MAX_CHUNK_SIZE
    ):
        raise ValueError(
            f"max_chunk_size must be an integer between 1 and "
            f"{MAX_CHUNK_SIZE}"
        )
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus does not exist: {corpus_path}")
    if not corpus_path.is_file() and not corpus_path.is_dir():
        raise ValueError(f"Corpus is not a file or directory: {corpus_path}")

    output_directory.mkdir(parents=True, exist_ok=True)
    chunks: list[Chunk] = []
    for source_file in tqdm(
        _iter_source_files(corpus_path),
        desc="Indexing files",
    ):
        relative_path = _project_relative_path(source_file)
        try:
            file_chunks = _relative_chunks(
                chunk_file(source_file, max_chunk_size),
                relative_path,
            )
        except (OSError, UnicodeError) as error:
            tqdm.write(f"Skipping {source_file}: {error}")
            continue
        chunks.extend(file_chunks)

    index_path = output_directory / INDEX_FILENAME
    index_data = {
        "chunks": chunks,
        "bm25": BM25Index.from_documents(chunk.text for chunk in chunks),
        "max_chunk_size": max_chunk_size,
    }
    with index_path.open("wb") as index_file:
        pickle.dump(index_data, index_file)
    return index_path
