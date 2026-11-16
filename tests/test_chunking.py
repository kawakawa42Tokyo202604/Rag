"""Tests for source chunking utilities."""

from pathlib import Path
from typing import cast

import pytest

from src.mandatory.chunking import (
    chunk_file,
    chunk_python_source,
    chunk_text_source,
)


def test_python_chunks_preserve_source_and_respect_size() -> None:
    """Preserve comments and blank lines while respecting the limit."""
    source = (
        "# header comment\n\n"
        "import os\n\n\n"
        "def first() -> int:\n"
        "    return 1\n\n\n"
        "def second() -> int:\n"
        "    return 2\n"
    )

    chunks = chunk_python_source(source, "example.py", max_chunk_size=50)

    assert chunks
    assert all(len(chunk.text) <= 50 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == source
    for chunk in chunks:
        assert chunk.text == source[
            chunk.first_character_index:chunk.last_character_index
        ]


def test_python_chunks_fallback_for_invalid_source() -> None:
    """Split syntactically invalid Python into bounded source ranges."""
    source = "def broken(:\n    return 1\n"

    chunks = chunk_python_source(source, "broken.py", max_chunk_size=10)

    assert chunks
    assert all(len(chunk.text) <= 10 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == source


def test_python_chunks_fallback_for_null_byte_source() -> None:
    """Fallback instead of raising when AST parsing sees a null byte."""
    source = "value = 1\n\x00value = 2\n"

    chunks = chunk_python_source(source, "invalid.py", max_chunk_size=10)

    assert chunks
    assert all(len(chunk.text) <= 10 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == source


def test_python_chunks_support_carriage_return_line_endings() -> None:
    """Calculate AST offsets for source files using CR line endings."""
    source = "first = 1\rsecond = 2\r"

    chunks = chunk_python_source(source, "carriage-return.py")

    assert "".join(chunk.text for chunk in chunks) == source
    for chunk in chunks:
        assert chunk.text == source[
            chunk.first_character_index:chunk.last_character_index
        ]


@pytest.mark.parametrize(
    "max_chunk_size",
    [0, -1, cast(int, True), cast(int, 1.5)],
)
def test_chunkers_reject_invalid_size(max_chunk_size: int) -> None:
    """Reject non-positive and non-integer maximum chunk sizes."""
    with pytest.raises(ValueError, match="positive integer"):
        chunk_python_source(
            "x = 1\n",
            "example.py",
            max_chunk_size=max_chunk_size,
        )
    with pytest.raises(ValueError, match="positive integer"):
        chunk_text_source(
            "text",
            "example.md",
            max_chunk_size=max_chunk_size,
        )


def test_python_chunks_preserve_unicode_character_offsets() -> None:
    """Use character offsets, not UTF-8 byte offsets, for Python source."""
    source = "# こんにちは\n\nvalue = '世界'\n"

    chunks = chunk_python_source(source, "unicode.py", max_chunk_size=15)

    assert "".join(chunk.text for chunk in chunks) == source
    for chunk in chunks:
        assert chunk.text == source[
            chunk.first_character_index:chunk.last_character_index
        ]


@pytest.mark.parametrize("suffix", [".PY", ".pyi"])
def test_chunk_file_uses_python_strategy_for_python_suffixes(
    tmp_path: Path,
    suffix: str,
) -> None:
    """Use AST chunking for case variants and Python stub files."""
    source = "x = 1\ndef second_function():\n    return 2\n"
    source_path = tmp_path / f"example{suffix}"
    source_path.write_text(source, encoding="utf-8")

    chunks = chunk_file(source_path, max_chunk_size=20)

    assert chunks == chunk_python_source(
        source,
        str(source_path),
        max_chunk_size=20,
    )


def test_chunk_file_preserves_carriage_return_offsets(
    tmp_path: Path,
) -> None:
    """Keep CRLF characters when indexing source ranges from a file."""
    source = "first = 1\r\nsecond = 2\r\n"
    source_path = tmp_path / "example.py"
    source_path.write_bytes(source.encode("utf-8"))

    chunks = chunk_file(source_path)

    assert "".join(chunk.text for chunk in chunks) == source
    for chunk in chunks:
        assert chunk.text == source[
            chunk.first_character_index:chunk.last_character_index
        ]


def test_text_chunks_preserve_markdown_blocks_and_ranges() -> None:
    """Keep headings, paragraphs, and fenced code in separate blocks."""
    source = (
        "# Title\n\n"
        + "A" * 40
        + "\n\n"
        + "```python\n"
        + "print('こんにちは')\n"
        + "```\n\n"
        + "## Next\n\n"
        + "C" * 80
    )

    chunks = chunk_text_source(source, "example.md", max_chunk_size=60)

    assert chunks[0].text.startswith("# Title")
    fenced_chunks = [
        chunk for chunk in chunks if "```python" in chunk.text
    ]
    assert len(fenced_chunks) == 1
    assert fenced_chunks[0].text.count("```") == 2
    assert any("## Next" in chunk.text for chunk in chunks)
    for chunk in chunks:
        assert len(chunk.text) <= 60
        assert chunk.text == source[
            chunk.first_character_index:chunk.last_character_index
        ]


def test_text_chunks_match_fence_marker_when_closing_code_blocks() -> None:
    """Keep a tilde line inside backtick fences as code content."""
    source = (
        "```python\n"
        "~~~ body text\n"
        "```\n\n"
        "## Heading after fence\n"
    )

    chunks = chunk_text_source(source, "example.md", max_chunk_size=40)

    assert all(
        not ("```python" in chunk.text and "## Heading" in chunk.text)
        for chunk in chunks
    )
    heading_start = source.index("## Heading")
    assert any(
        chunk.first_character_index == heading_start
        and chunk.text.startswith("## Heading")
        for chunk in chunks
    )


def test_text_chunks_handle_empty_source_and_invalid_size() -> None:
    """Return no chunks for empty text."""
    assert chunk_text_source("", "empty.md") == []
