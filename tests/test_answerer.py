"""Tests for source-grounded answer generation."""

from pathlib import Path
from typing import Any

import pytest

from src.mandatory.answerer import (
    AnswerGenerator,
    NO_CONTEXT_ANSWER,
    build_context,
)
from src.mandatory.models import MinimalSource


class _FakeTokenizer:
    """Small tokenizer double for generation tests."""

    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []
        self.call_kwargs: dict[str, Any] = {}
        self.model_max_length = 1_000_000_000

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        del kwargs
        self.messages = messages
        return "prompt"

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        del prompt
        self.call_kwargs = kwargs
        return {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}

    def batch_decode(
        self,
        token_batches: list[list[int]],
        skip_special_tokens: bool,
    ) -> list[str]:
        del token_batches, skip_special_tokens
        return ["The answer is grounded."]


class _FakeModel:
    """Small causal-language-model double for generation tests."""

    def __init__(self) -> None:
        self.generated = False

    def parameters(self) -> Any:
        return iter(())

    def eval(self) -> None:
        return None

    def generate(self, **kwargs: Any) -> list[list[int]]:
        del kwargs
        self.generated = True
        return [[1, 2, 3]]


def test_build_context_restores_source_ranges(tmp_path: Path) -> None:
    """Format the indexed source text with its location metadata."""
    source_path = tmp_path / "example.py"
    source_path.write_text("before\nanswer = 42\nafter\n", encoding="utf-8")
    source = MinimalSource(
        file_path=str(source_path),
        first_character_index=7,
        last_character_index=18,
    )

    context = build_context([source])

    assert "answer = 42" in context
    assert str(source_path) in context
    assert "7:18" in context


def test_build_context_skips_a_header_that_does_not_fit(
    tmp_path: Path,
) -> None:
    """Try later sources when an earlier source header exceeds the limit."""
    long_path = tmp_path / ("long-name-" * 20 + ".py")
    short_path = tmp_path / "x.py"
    long_path.write_text("a", encoding="utf-8")
    short_path.write_text("b", encoding="utf-8")
    sources = [
        MinimalSource(
            file_path=str(long_path),
            first_character_index=0,
            last_character_index=1,
        ),
        MinimalSource(
            file_path=str(short_path),
            first_character_index=0,
            last_character_index=1,
        ),
    ]
    short_block_length = len(f"[Source 2: {short_path} 0:1]\n") + 1

    context = build_context(sources, short_block_length)

    assert str(long_path) not in context
    assert str(short_path) in context
    assert context.endswith("b")


def test_answer_generator_returns_safe_fallback_without_context() -> None:
    """Avoid loading the model when no usable source was retrieved."""
    generator = AnswerGenerator()

    assert generator.generate("What?", []) == NO_CONTEXT_ANSWER


def test_answer_generator_rejects_boolean_limits() -> None:
    """Reject booleans even though Python treats them as integers."""
    with pytest.raises(ValueError, match="positive integer"):
        AnswerGenerator(max_new_tokens=True)
    with pytest.raises(ValueError, match="positive integer"):
        build_context([], max_context_characters=True)


def test_answer_generator_uses_fake_model_and_context(tmp_path: Path) -> None:
    """Generate an answer from a prompt containing the retrieved context."""
    source_path = tmp_path / "example.py"
    source_path.write_text("answer = 42\n", encoding="utf-8")
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    generator = AnswerGenerator(tokenizer=tokenizer, model=model)

    answer = generator.generate(
        "What is the answer?",
        [
            MinimalSource(
                file_path=str(source_path),
                first_character_index=0,
                last_character_index=12,
            )
        ],
    )

    assert answer == "The answer is grounded."
    assert model.generated
    assert "What is the answer?" in tokenizer.messages[1]["content"]
    assert "answer = 42" in tokenizer.messages[1]["content"]


def test_answer_generator_reserves_model_tokens_for_generation(
    tmp_path: Path,
) -> None:
    """Truncate prompts to leave room inside the model context window."""
    source_path = tmp_path / "example.py"
    source_path.write_text("answer = 42\n", encoding="utf-8")
    tokenizer = _FakeTokenizer()
    tokenizer.model_max_length = 10
    generator = AnswerGenerator(
        tokenizer=tokenizer,
        model=_FakeModel(),
        max_new_tokens=3,
    )

    generator.generate(
        "What is the answer?",
        [
            MinimalSource(
                file_path=str(source_path),
                first_character_index=0,
                last_character_index=12,
            )
        ],
    )

    assert tokenizer.call_kwargs == {
        "return_tensors": "pt",
        "truncation": True,
        "max_length": 7,
    }
