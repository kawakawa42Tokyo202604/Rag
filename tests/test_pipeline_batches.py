"""Tests for dataset search and answer-generation pipeline commands."""

import json
from pathlib import Path
from typing import Any

import pytest

from src.mandatory.generator import RAGPipeline
from src.mandatory.io import (
    load_student_search_results,
    load_student_search_results_and_answer,
    save_student_search_results,
)
from src.mandatory.models import (
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
)


def _source() -> MinimalSource:
    return MinimalSource(
        file_path="example.py",
        first_character_index=0,
        last_character_index=10,
    )


def _write_unanswered_dataset(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "rag_questions": [
                    {
                        "question_id": "question-1",
                        "question": "What is this?",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_search_dataset_saves_one_result_per_question(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve dataset question fields and save retrieved sources."""
    dataset_path = tmp_path / "questions.json"
    _write_unanswered_dataset(dataset_path)
    output_directory = tmp_path / "search-results"

    def fake_retrieve_sources(
        query: str,
        k: int,
        index_path: Path,
    ) -> list[MinimalSource]:
        assert query == "What is this?"
        assert k == 3
        assert index_path == tmp_path / "index"
        return [_source()]

    monkeypatch.setattr(
        "src.mandatory.generator.retrieve_sources",
        fake_retrieve_sources,
    )

    RAGPipeline().search_dataset(
        dataset_path=str(dataset_path),
        k=3,
        save_directory=str(output_directory),
        index_path=str(tmp_path / "index"),
    )

    results = load_student_search_results(output_directory / dataset_path.name)
    assert results.k == 3
    assert results.search_results[0].question_id == "question-1"
    assert results.search_results[0].retrieved_sources == [_source()]


class _FakeAnswerGenerator:
    """Answer-generator double for batch processing."""

    def __init__(self, **kwargs: Any) -> None:
        del kwargs

    def generate(
        self,
        question: str,
        sources: list[MinimalSource],
    ) -> str:
        return f"answer:{question}:{len(sources)}"


def test_answer_dataset_preserves_search_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Add answers without changing IDs, questions, sources, or k."""
    input_path = tmp_path / "search-results.json"
    save_student_search_results(
        StudentSearchResults(
            search_results=[
                MinimalSearchResults(
                    question_id="question-1",
                    question="What is this?",
                    retrieved_sources=[_source()],
                )
            ],
            k=5,
        ),
        input_path,
    )
    output_directory = tmp_path / "answers"
    monkeypatch.setattr(
        "src.mandatory.generator.AnswerGenerator",
        _FakeAnswerGenerator,
    )

    RAGPipeline().answer_dataset(
        student_search_results_path=str(input_path),
        save_directory=str(output_directory),
    )

    results = load_student_search_results_and_answer(
        output_directory / input_path.name
    )
    result = results.search_results[0]
    assert results.k == 5
    assert result.question_id == "question-1"
    assert result.question == "What is this?"
    assert result.retrieved_sources == [_source()]
    assert result.answer == "answer:What is this?:1"
