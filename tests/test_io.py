"""Tests for JSON input and output helpers."""

import json
from pathlib import Path

import pytest

from src.mandatory.io import (
    load_dataset,
    load_student_search_results,
    save_student_search_results,
)
from src.mandatory.models import (
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
)


def test_load_dataset_validates_json_file(tmp_path: Path) -> None:
    """Load an unanswered dataset through its Pydantic model."""
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
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

    dataset = load_dataset(dataset_path)

    assert dataset.rag_questions[0].question_id == "question-1"


def test_search_results_round_trip_creates_parent_directory(
    tmp_path: Path,
) -> None:
    """Save and reload the required student search result shape."""
    source = MinimalSource(
        file_path="data/raw/vllm-0.10.1/example.py",
        first_character_index=0,
        last_character_index=10,
    )
    results = StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id="question-1",
                question="What is this?",
                retrieved_sources=[source],
            )
        ],
        k=5,
    )
    output_path = tmp_path / "nested" / "results.json"

    assert save_student_search_results(results, output_path) == output_path
    loaded = load_student_search_results(output_path)

    assert loaded == results


def test_load_dataset_rejects_invalid_json(tmp_path: Path) -> None:
    """Convert malformed JSON into a concise value error."""
    dataset_path = tmp_path / "invalid.json"
    dataset_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_dataset(dataset_path)


def test_load_dataset_rejects_invalid_model_data(tmp_path: Path) -> None:
    """Reject JSON that does not match the RAG dataset model."""
    dataset_path = tmp_path / "invalid-shape.json"
    dataset_path.write_text(
        json.dumps({"rag_questions": [{}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid RagDataset data"):
        load_dataset(dataset_path)


def test_load_dataset_reports_missing_file(tmp_path: Path) -> None:
    """Preserve a clear missing-file error for the CLI boundary."""
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.json")
