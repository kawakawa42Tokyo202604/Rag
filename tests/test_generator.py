"""Tests for CLI pipeline orchestration."""

from pathlib import Path

import pytest

from src.mandatory.io import save_student_search_results
from src.mandatory.generator import RAGPipeline
from src.mandatory.models import MinimalSearchResults, StudentSearchResults


def test_index_accepts_output_directory(
    tmp_path: Path,
) -> None:
    """Save an index to the directory selected through the CLI method."""
    source_file = tmp_path / "example.py"
    output_directory = tmp_path / "custom-index"
    source_file.write_text("value = 1\n", encoding="utf-8")

    RAGPipeline().index(
        file_path=str(source_file),
        output_directory=str(output_directory),
    )

    assert (output_directory / "index.pkl").exists()


def test_evaluate_prints_recall_for_saved_results(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose Recall@k evaluation through the CLI pipeline."""
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        '{"rag_questions": [{'
        '"question_id": "question-1", '
        '"question": "Question?", '
        '"sources": [{"file_path": "example.py", '
        '"first_character_index": 0, "last_character_index": 100}], '
        '"answer": "Answer"}]}',
        encoding="utf-8",
    )
    results_path = tmp_path / "results.json"
    save_student_search_results(
        StudentSearchResults(
            search_results=[
                MinimalSearchResults(
                    question_id="question-1",
                    question="Question?",
                    retrieved_sources=[],
                )
            ],
            k=5,
        ),
        results_path,
    )

    RAGPipeline().evaluate(
        student_search_results_path=str(results_path),
        dataset_path=str(dataset_path),
    )

    assert "[evaluate] recall=0.0000" in capsys.readouterr().out
