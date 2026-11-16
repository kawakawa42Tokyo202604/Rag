"""Source-grounded answer generation with the local Qwen model."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

from .models import MinimalSource


DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_MAX_CONTEXT_CHARACTERS = 12_000
DEFAULT_MAX_NEW_TOKENS = 256
NO_CONTEXT_ANSWER = "I don't know based on the retrieved context."

_SYSTEM_PROMPT = (
    "You answer questions about a source codebase. Use only the provided "
    "context. If the context is insufficient, say that you do not know. "
    "Answer the question directly and concisely."
)

__all__ = [
    "AnswerGenerator",
    "DEFAULT_MAX_CONTEXT_CHARACTERS",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_MODEL_NAME",
    "NO_CONTEXT_ANSWER",
    "build_context",
]


def _read_source_snippet(source: MinimalSource) -> str | None:
    """Read one indexed source range, returning ``None`` when unavailable."""
    first = source.first_character_index
    last = source.last_character_index
    if type(first) is not int or type(last) is not int:
        return None
    if first < 0 or last <= first:
        return None

    try:
        with Path(source.file_path).open(
            "r",
            encoding="utf-8",
            newline="",
        ) as source_file:
            source_text = source_file.read()
    except (OSError, TypeError, UnicodeError, ValueError):
        return None

    if last > len(source_text):
        return None
    return source_text[first:last]


def build_context(
    sources: Sequence[MinimalSource],
    max_context_characters: int = DEFAULT_MAX_CONTEXT_CHARACTERS,
) -> str:
    """Restore and format retrieved source snippets for the model prompt."""
    if (
        type(max_context_characters) is not int
        or max_context_characters <= 0
    ):
        raise ValueError("max_context_characters must be a positive integer")

    blocks: list[str] = []
    used_characters = 0
    for source_number, source in enumerate(sources, start=1):
        snippet = _read_source_snippet(source)
        if not snippet:
            continue

        header = (
            f"[Source {source_number}: {source.file_path} "
            f"{source.first_character_index}:{source.last_character_index}]\n"
        )
        separator_length = 2 if blocks else 0
        remaining = max_context_characters - used_characters
        if remaining <= separator_length + len(header):
            continue

        available_snippet_length = (
            remaining - separator_length - len(header)
        )
        block = header + snippet[:available_snippet_length]
        blocks.append(block)
        used_characters += separator_length + len(block)

        if len(snippet) > available_snippet_length:
            break

    return "\n\n".join(blocks)


def _prompt_for_question(tokenizer: Any, question: str, context: str) -> Any:
    """Build a chat prompt, supporting tokenizers without Qwen3 options."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nContext:\n{context}",
        },
    ]
    chat_template = getattr(tokenizer, "apply_chat_template", None)
    if chat_template is None:
        return (
            f"System: {_SYSTEM_PROMPT}\n\n"
            f"Question:\n{question}\n\nContext:\n{context}\n\n"
            "Answer:\n"
        )

    try:
        return chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _model_device(model: Any) -> Any:
    """Return the device of a model when it exposes one."""
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return getattr(model, "device", None)


def _move_inputs_to_device(inputs: Any, device: Any) -> Any:
    """Move tokenizer output to the model device when possible."""
    if device is None:
        return inputs

    move = getattr(inputs, "to", None)
    if callable(move):
        return move(device)

    if isinstance(inputs, dict):
        return {
            name: value.to(device) if hasattr(value, "to") else value
            for name, value in inputs.items()
        }
    return inputs


def _input_sequence_length(inputs: Any) -> int:
    """Get the prompt token count from tokenizer output."""
    input_ids = inputs["input_ids"]
    shape = getattr(input_ids, "shape", None)
    if shape is not None:
        return int(shape[-1])
    return len(input_ids[0])


def _prompt_token_limit(
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
) -> int | None:
    """Return the safe prompt length advertised by the model components."""
    limits: list[int] = []
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    model_config = getattr(model, "config", None)
    model_limit = getattr(model_config, "max_position_embeddings", None)

    for limit in (tokenizer_limit, model_limit):
        if type(limit) is int and 0 < limit < 1_000_000_000:
            limits.append(limit)

    if not limits:
        return None

    context_window = min(limits)
    if max_new_tokens >= context_window:
        raise ValueError(
            "max_new_tokens must be smaller than the model context window"
        )
    return context_window - max_new_tokens


class AnswerGenerator:
    """Lazily load Qwen and generate answers from retrieved context."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        max_context_characters: int = DEFAULT_MAX_CONTEXT_CHARACTERS,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        """Configure generation, optionally accepting test doubles."""
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if (
            type(max_context_characters) is not int
            or max_context_characters <= 0
        ):
            raise ValueError(
                "max_context_characters must be a positive integer"
            )
        if (tokenizer is None) != (model is None):
            raise ValueError("tokenizer and model must be provided together")

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_context_characters = max_context_characters
        self._tokenizer = tokenizer
        self._model = model

    def _load_components(self) -> tuple[Any, Any]:
        """Load tokenizer and model only when a non-empty context is used."""
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
            )
            model_eval = cast(Callable[[], Any], getattr(self._model, "eval"))
            model_eval()
        except Exception as error:
            raise RuntimeError(
                f"Unable to load answer model: {self.model_name}"
            ) from error

        return self._tokenizer, self._model

    def generate(
        self,
        question: str,
        sources: Sequence[MinimalSource],
    ) -> str:
        """Generate a grounded answer for one question."""
        if not isinstance(question, str):
            raise ValueError("question must be a string")

        context = build_context(sources, self.max_context_characters)
        if not context:
            return NO_CONTEXT_ANSWER

        tokenizer, model = self._load_components()
        try:
            prompt = _prompt_for_question(tokenizer, question, context)
            tokenizer_options: dict[str, Any] = {"return_tensors": "pt"}
            prompt_token_limit = _prompt_token_limit(
                tokenizer,
                model,
                self.max_new_tokens,
            )
            if prompt_token_limit is not None:
                tokenizer_options.update(
                    truncation=True,
                    max_length=prompt_token_limit,
                )
            inputs = tokenizer(prompt, **tokenizer_options)
            inputs = _move_inputs_to_device(inputs, _model_device(model))
            prompt_length = _input_sequence_length(inputs)

            try:
                import torch
            except ImportError:
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                )
            else:
                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                    )

            generated_tokens = generated_ids[0][prompt_length:]
            batch_decode = getattr(tokenizer, "batch_decode", None)
            if callable(batch_decode):
                decoded = batch_decode(
                    [generated_tokens],
                    skip_special_tokens=True,
                )
                answer = decoded[0] if decoded else ""
            else:
                answer = tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True,
                )
        except Exception as error:
            raise RuntimeError("Unable to generate an answer") from error

        answer = answer.strip()
        return answer or NO_CONTEXT_ANSWER
