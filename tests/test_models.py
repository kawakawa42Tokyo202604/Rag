"""Tests for the Pydantic models shared by the RAG pipeline."""

from uuid import UUID

import pytest
from pydantic import ValidationError

from src.mandatory.models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)


def test_unanswered_question_generates_unique_uuid_strings() -> None:
    """Generate a valid, distinct UUID when question_id is omitted."""
    first = UnansweredQuestion(question="First question?")
    second = UnansweredQuestion(question="Second question?")

    assert UUID(first.question_id)
    assert UUID(second.question_id)
    assert first.question_id != second.question_id


def test_answered_question_contains_sources_and_answer() -> None:
    """Preserve inherited question fields and reference answer data."""
    source = MinimalSource(
        file_path="data/raw/vllm-0.10.1/vllm/example.py",
        first_character_index=10,
        last_character_index=42,
    )

    question = AnsweredQuestion(
        question_id="question-1",
        question="Where is the example?",
        sources=[source],
        answer="It is in example.py.",
    )

    assert question.sources == [source]
    assert question.answer == "It is in example.py."
    assert question.question == "Where is the example?"


def test_rag_dataset_parses_answered_and_unanswered_questions() -> None:
    """Parse both supported question variants from dataset-shaped input."""
    dataset = RagDataset.model_validate(
        {
            "rag_questions": [
                {
                    "question_id": "answered-1",
                    "question": "Answered?",
                    "sources": [
                        {
                            "file_path": "data/raw/vllm-0.10.1/README.md",
                            "first_character_index": 0,
                            "last_character_index": 20,
                        }
                    ],
                    "answer": "Yes.",
                    "difficulty": "synthetic",
                    "is_valid": True,
                },
                {
                    "question_id": "unanswered-1",
                    "question": "Not answered yet?",
                },
            ]
        }
    )

    assert isinstance(dataset.rag_questions[0], AnsweredQuestion)
    assert isinstance(dataset.rag_questions[1], UnansweredQuestion)


def test_student_search_results_match_required_output_shape() -> None:
    """Serialize batch retrieval output with the required field names."""
    source = MinimalSource(
        file_path="data/raw/vllm-0.10.1/README.md",
        first_character_index=0,
        last_character_index=100,
    )
    result = MinimalSearchResults(
        question_id="question-1",
        question="What is vLLM?",
        retrieved_sources=[source],
    )

    output = StudentSearchResults(search_results=[result], k=5).model_dump()

    assert output == {
        "search_results": [
            {
                "question_id": "question-1",
                "question": "What is vLLM?",
                "retrieved_sources": [
                    {
                        "file_path": "data/raw/vllm-0.10.1/README.md",
                        "first_character_index": 0,
                        "last_character_index": 100,
                    }
                ],
            }
        ],
        "k": 5,
    }


def test_student_answers_match_required_output_shape() -> None:
    """Serialize batch answer output with retrieved sources and answer text."""
    answer = MinimalAnswer(
        question_id="question-1",
        question="What is vLLM?",
        retrieved_sources=[],
        answer="A library for LLM inference.",
    )

    output = StudentSearchResultsAndAnswer(
        search_results=[answer],
        k=0,
    ).model_dump()

    assert output["k"] == 0
    assert output["search_results"][0]["answer"] == (
        "A library for LLM inference."
    )


def test_required_fields_are_validated() -> None:
    """Reject source data when a mandatory location field is missing."""
    with pytest.raises(ValidationError):
        MinimalSource.model_validate(
            {
                "file_path": "data/raw/vllm-0.10.1/README.md",
                "first_character_index": 0,
            }
        )


@pytest.mark.parametrize(
    ("first_character_index", "last_character_index"),
    [(-1, 0), (0, -1), (0, 0), (2, 1)],
)
def test_source_rejects_invalid_character_ranges(
    first_character_index: int,
    last_character_index: int,
) -> None:
    """Reject source locations outside a valid half-open range."""
    with pytest.raises(ValidationError):
        MinimalSource(
            file_path="example.py",
            first_character_index=first_character_index,
            last_character_index=last_character_index,
        )


@pytest.mark.parametrize(
    ("first_character_index", "last_character_index"),
    [(True, 2), (0, True), ("0", 1)],
)
def test_source_rejects_non_integer_character_ranges(
    first_character_index: object,
    last_character_index: object,
) -> None:
    """Reject coercible values at the persisted model boundary."""
    with pytest.raises(ValidationError):
        MinimalSource.model_validate(
            {
                "file_path": "example.py",
                "first_character_index": first_character_index,
                "last_character_index": last_character_index,
            }
        )


def test_student_results_reject_negative_k() -> None:
    """Reject a negative retrieval count in both output models."""
    with pytest.raises(ValidationError):
        StudentSearchResults(search_results=[], k=-1)
    with pytest.raises(ValidationError):
        StudentSearchResultsAndAnswer(search_results=[], k=-1)


@pytest.mark.parametrize("k", [True, "1"])
def test_student_results_reject_non_integer_k(k: object) -> None:
    """Reject coercible values for the persisted result count."""
    with pytest.raises(ValidationError):
        StudentSearchResults.model_validate({"search_results": [], "k": k})
    with pytest.raises(ValidationError):
        StudentSearchResultsAndAnswer.model_validate(
            {"search_results": [], "k": k}
        )


@pytest.mark.parametrize("missing_field", ["sources", "answer"])
def test_rag_dataset_rejects_partial_answered_questions(
    missing_field: str,
) -> None:
    """Reject records that contain only one answered-question field."""
    question = {
        "question_id": "question-1",
        "question": "What is this?",
        "sources": [],
        "answer": "A test answer.",
    }
    question.pop(missing_field)

    with pytest.raises(
        ValidationError,
        match="both sources and answer",
    ):
        RagDataset.model_validate({"rag_questions": [question]})
