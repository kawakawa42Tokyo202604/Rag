"""Source-code and document chunking utilities for the RAG index."""

import ast
from dataclasses import dataclass
from pathlib import Path


_PYTHON_SUFFIXES = frozenset({".py", ".pyi"})


@dataclass(frozen=True)
class Chunk:
    """Represent one searchable fragment of a source file."""

    text: str
    file_path: str
    first_character_index: int
    last_character_index: int


def _line_starts(source: str) -> list[int]:
    """Return the character offset at the start of every source line."""
    starts = [0]
    for index, character in enumerate(source):
        if character == "\n":
            starts.append(index + 1)
        elif character == "\r" and (
            index + 1 >= len(source) or source[index + 1] != "\n"
        ):
            starts.append(index + 1)
    return starts


def _fence_run(stripped_line: str) -> str | None:
    """Return a Markdown fence run at the start of a stripped line."""
    if not stripped_line or stripped_line[0] not in ("`", "~"):
        return None

    fence_character = stripped_line[0]
    fence_end = 0
    while (
        fence_end < len(stripped_line)
        and stripped_line[fence_end] == fence_character
    ):
        fence_end += 1

    if fence_end < 3:
        return None
    return stripped_line[:fence_end]


def _node_end_offset(
    source: str,
    line_starts: list[int],
    node: ast.AST,
) -> int | None:
    """Convert an AST node's end position to a Python string offset."""
    end_line = getattr(node, "end_lineno", None)
    end_column = getattr(node, "end_col_offset", None)
    if not isinstance(end_line, int) or not isinstance(end_column, int):
        return None

    line_start = line_starts[end_line - 1]
    line_end = (
        line_starts[end_line]
        if end_line < len(line_starts)
        else len(source)
    )
    line = source[line_start:line_end]
    prefix = line.encode("utf-8")[:end_column].decode("utf-8")
    return line_start + len(prefix)


def _fixed_chunks(
    source: str,
    file_path: str,
    start: int,
    end: int,
    max_chunk_size: int,
) -> list[Chunk]:
    """Split one source range into fixed-size chunks."""
    chunks: list[Chunk] = []
    for chunk_start in range(start, end, max_chunk_size):
        chunk_end = min(chunk_start + max_chunk_size, end)
        chunks.append(
            Chunk(
                text=source[chunk_start:chunk_end],
                file_path=file_path,
                first_character_index=chunk_start,
                last_character_index=chunk_end,
            )
        )
    return chunks


def _pack_source_ranges(
    source: str,
    file_path: str,
    ranges: list[tuple[int, int]],
    max_chunk_size: int,
) -> list[Chunk]:
    """Pack adjacent AST ranges without exceeding the maximum size."""
    chunks: list[Chunk] = []
    current_start: int | None = None
    current_end: int | None = None

    for range_start, range_end in ranges:
        if range_start == range_end:
            continue
        range_size = range_end - range_start

        if range_size > max_chunk_size:
            if current_start is not None and current_end is not None:
                chunks.append(
                    Chunk(
                        text=source[current_start:current_end],
                        file_path=file_path,
                        first_character_index=current_start,
                        last_character_index=current_end,
                    )
                )
                current_start = None
                current_end = None
            chunks.extend(
                _fixed_chunks(
                    source,
                    file_path,
                    range_start,
                    range_end,
                    max_chunk_size,
                )
            )
            continue

        if current_start is None or current_end is None:
            current_start = range_start
            current_end = range_end
        elif range_end - current_start <= max_chunk_size:
            current_end = range_end
        else:
            chunks.append(
                Chunk(
                    text=source[current_start:current_end],
                    file_path=file_path,
                    first_character_index=current_start,
                    last_character_index=current_end,
                )
            )
            current_start = range_start
            current_end = range_end

    if current_start is not None and current_end is not None:
        chunks.append(
            Chunk(
                text=source[current_start:current_end],
                file_path=file_path,
                first_character_index=current_start,
                last_character_index=current_end,
            )
        )
    return chunks


def chunk_python_source(
    txt: str,
    file_path: str,
    max_chunk_size: int = 2000,
) -> list[Chunk]:
    """Split Python source code into searchable chunks.

    Args:
        source: Complete Python source code.
        file_path: Project-relative path of the source file.
        max_chunk_size: Maximum number of characters in one chunk.

    Returns:
        Chunks containing the source text and its character range.

    Raises:
        ValueError: If ``max_chunk_size`` is not a positive integer.
    """
    if type(max_chunk_size) is not int or max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be a positive integer")

    try:
        tree = ast.parse(txt, filename=file_path)
    except (SyntaxError, ValueError):
        return _fixed_chunks(
            txt,
            file_path,
            0,
            len(txt),
            max_chunk_size,
        )

    line_starts = _line_starts(txt)
    ranges: list[tuple[int, int]] = []
    previous_end = 0

    for node in tree.body:
        end = _node_end_offset(txt, line_starts, node)
        if end is None:
            continue
        ranges.append((previous_end, end))
        previous_end = end

    if previous_end < len(txt):
        ranges.append((previous_end, len(txt)))

    return _pack_source_ranges(
        txt,
        file_path,
        ranges,
        max_chunk_size,
    )


def chunk_text_source(
    txt: str,
    file_path: str,
    max_chunk_size: int = 2000,
) -> list[Chunk]:
    """Split Markdown or plain-text source into searchable chunks.

    Args:
        txt: Complete Markdown or plain-text content.
        file_path: Project-relative path of the source file.
        max_chunk_size: Maximum number of characters in one chunk.

    Returns:
        Chunks containing the source text and its character range.

    Raises:
        ValueError: If ``max_chunk_size`` is not a positive integer.
    """
    if type(max_chunk_size) is not int or max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be a positive integer")
    if not txt:
        return []

    ranges: list[tuple[int, int]] = []
    block_start = 0
    block_end = 0
    line_start = 0
    previous_line_blank = False
    block_has_content = False
    fence_marker: str | None = None

    for line in txt.splitlines(keepends=True):
        line_end = line_start + len(line)
        stripped = line.lstrip()
        hash_count = len(stripped) - len(stripped.lstrip("#"))
        is_heading = (
            1 <= hash_count <= 6
            and (
                len(stripped) == hash_count
                or stripped[hash_count].isspace()
            )
        )
        is_blank = not line.strip()
        fence_run = _fence_run(stripped)

        starts_new_block = (
            block_has_content
            and fence_marker is None
            and not is_blank
            and (previous_line_blank or is_heading)
        )
        if starts_new_block:
            ranges.append((block_start, block_end))
            block_start = line_start

        block_end = line_end
        if not is_blank:
            block_has_content = True

        if fence_marker is None:
            if fence_run is not None:
                fence_marker = fence_run
        elif (
            fence_run is not None
            and fence_run[0] == fence_marker[0]
            and len(fence_run) >= len(fence_marker)
            and not stripped[len(fence_run):].strip()
        ):
            fence_marker = None
        previous_line_blank = is_blank and fence_marker is None
        line_start = line_end

    if block_end > block_start:
        ranges.append((block_start, block_end))

    return _pack_source_ranges(
        txt,
        file_path,
        ranges,
        max_chunk_size,
    )


def chunk_file(
    file_path: Path,
    max_chunk_size: int = 2000,
) -> list[Chunk]:
    """Read a source file and dispatch it to the appropriate chunker.

    Args:
        file_path: Path to a Python, Markdown, or text file.
        max_chunk_size: Maximum number of characters in one chunk.

    Returns:
        Chunks produced by the Python or text chunking strategy.

    """
    with file_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as source_file:
        txt = source_file.read()
    if file_path.suffix.lower() in _PYTHON_SUFFIXES:
        return chunk_python_source(
            txt,
            str(file_path),
            max_chunk_size,
        )
    return chunk_text_source(
        txt,
        str(file_path),
        max_chunk_size,
    )
