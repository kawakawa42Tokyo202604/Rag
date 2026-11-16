"""Optional checks using the real answer model, enabled by make test-model."""

import os
from pathlib import Path

import pytest

from src.mandatory.answerer import AnswerGenerator, NO_CONTEXT_ANSWER
from src.mandatory.models import MinimalSource


@pytest.mark.model
def test_real_model_generates_from_source(tmp_path: Path) -> None:
    """Generate an answer from a source range using real model weights."""
    source = tmp_path / "example.py"
    text = "def add(a, b):\n    return a + b\n"
    source.write_text(text, encoding="utf-8")
    generator = AnswerGenerator(
        model_name=os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B"),
        max_new_tokens=64,
    )
    answer = generator.generate(
        "What does the add function return?",
        [MinimalSource(file_path=str(source), first_character_index=0,
                       last_character_index=len(text))],
    )
    assert answer.strip()
    assert answer != NO_CONTEXT_ANSWER
