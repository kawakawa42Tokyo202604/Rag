"""Tests for Recall@k evaluation."""

import json
from pathlib import Path

import pytest

from src.mandatory.evaluator import (
    calculate_iou,
    calculate_recall,
    evaluate,
    evaluate_recall,
    sources_match,
)
from src.mandatory.io import save_student_search_results
from src.mandatory.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    UnansweredQuestion,
)


def _source(
    file_path: str = "example.py",
    first: int = 0,
    last: int = 100,
) -> MinimalSource:
    return MinimalSource(
        file_path=file_path,
        first_character_index=first,
        last_character_index=last,
    )


def test_calculate_iou_uses_half_open_ranges() -> None:
    """Calculate the expected overlap for source character ranges."""
    assert calculate_iou(
        _source(first=0, last=100),
        _source(first=50, last=150),
    ) == pytest.approx(1 / 3)
    assert calculate_iou(
        _source(first=0, last=10),
        _source(first=10, last=20),
    ) == 0.0


def test_sources_match_requires_same_path_and_iou_threshold() -> None:
    """Require both the exact file path and sufficient range overlap."""
    expected = _source(first=0, last=100)
    assert sources_match(expected, _source(first=90, last=110))
    assert not sources_match(expected, _source(first=96, last=196))
    assert not sources_match(expected, _source("other.py", 90, 110))


def test_calculate_recall_does_not_double_count_expected_sources() -> None:
    """Count each expected source once even with duplicate retrievals."""
    expected = [_source(first=0, last=100), _source(first=200, last=300)]
    retrieved = [
        _source(first=90, last=110),
        _source(first=90, last=110),
    ]

    assert calculate_recall(expected, retrieved) == pytest.approx(0.5)


def test_evaluate_recall_handles_multiple_questions_and_k() -> None:
    """Aggregate valid sources and limit evaluation to the requested k."""
    first_source = _source(first=0, last=100)
    second_source = _source(first=200, last=300)
    dataset = RagDataset(
        rag_questions=[
            AnsweredQuestion(
                question_id="answered-1",
                question="First?",
                sources=[first_source, second_source],
                answer="First answer",
            ),
            UnansweredQuestion(question_id="unanswered-1", question="Next?"),
        ]
    )
    results = StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id="answered-1",
                question="First?",
                retrieved_sources=[
                    _source(first=90, last=110),
                    _source(first=190, last=310),
                ],
            ),
            MinimalSearchResults(
                question_id="unknown",
                question="Unknown?",
                retrieved_sources=[first_source],
            ),
        ],
        k=1,
    )

    assert evaluate_recall(dataset, results) == pytest.approx(0.5)


def test_evaluate_recall_averages_each_question_equally() -> None:
    """Average per-question recall instead of weighting source-rich rows."""
    dataset = RagDataset(
        rag_questions=[
            AnsweredQuestion(
                question_id="question-1",
                question="First?",
                sources=[
                    _source(first=0, last=100),
                    _source(first=200, last=300),
                ],
                answer="First answer",
            ),
            AnsweredQuestion(
                question_id="question-2",
                question="Second?",
                sources=[_source(first=400, last=500)],
                answer="Second answer",
            ),
        ]
    )
    results = StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id="question-1",
                question="First?",
                retrieved_sources=[_source(first=0, last=100)],
            ),
            MinimalSearchResults(
                question_id="question-2",
                question="Second?",
                retrieved_sources=[_source(first=400, last=500)],
            ),
        ],
        k=5,
    )

    assert evaluate_recall(dataset, results) == pytest.approx(0.75)


def test_evaluate_recall_returns_zero_for_missing_or_empty_inputs() -> None:
    """Treat missing result rows and datasets without references safely."""
    dataset = RagDataset(
        rag_questions=[
            AnsweredQuestion(
                question_id="question-1",
                question="Question?",
                sources=[_source()],
                answer="Answer",
            )
        ]
    )
    empty_results = StudentSearchResults(search_results=[], k=5)
    no_expected_sources = RagDataset(
        rag_questions=[
            AnsweredQuestion(
                question_id="question-2",
                question="Question?",
                sources=[],
                answer="Answer",
            )
        ]
    )

    assert evaluate_recall(dataset, empty_results) == 0.0
    assert evaluate_recall(no_expected_sources, empty_results) == 0.0


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_evaluator_rejects_invalid_iou_threshold(threshold: float) -> None:
    """Reject thresholds that cannot represent an IoU criterion."""
    with pytest.raises(ValueError):
        sources_match(_source(), _source(), threshold)


def test_evaluate_loads_json_models(tmp_path: Path) -> None:
    """Evaluate persisted dataset and search-result models."""
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "rag_questions": [
                    {
                        "question_id": "question-1",
                        "question": "Question?",
                        "sources": [
                            {
                                "file_path": "example.py",
                                "first_character_index": 0,
                                "last_character_index": 100,
                            }
                        ],
                        "answer": "Answer",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    results_path = tmp_path / "results.json"
    save_student_search_results(
        StudentSearchResults(
            search_results=[
                MinimalSearchResults(
                    question_id="question-1",
                    question="Question?",
                    retrieved_sources=[_source(first=90, last=110)],
                )
            ],
            k=5,
        ),
        results_path,
    )

    assert evaluate(results_path, dataset_path) == 1.0
