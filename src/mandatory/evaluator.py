"""Recall@k evaluation for retrieved source locations."""

from __future__ import annotations

import math
from pathlib import Path

from .io import load_dataset, load_student_search_results
from .models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


DEFAULT_IOU_THRESHOLD = 0.05

__all__ = [
    "DEFAULT_IOU_THRESHOLD",
    "calculate_iou",
    "calculate_recall",
    "evaluate",
    "evaluate_recall",
    "sources_match",
]


def _validate_iou_threshold(iou_threshold: float) -> float:
    """Validate and normalize an IoU threshold."""
    if isinstance(iou_threshold, bool) or not isinstance(
        iou_threshold,
        (int, float),
    ):
        raise ValueError("iou_threshold must be a finite number")

    threshold = float(iou_threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")
    return threshold


def _source_range(source: MinimalSource) -> tuple[int, int] | None:
    """Return a valid non-empty half-open source range, if available."""
    first = source.first_character_index
    last = source.last_character_index
    if type(first) is not int or type(last) is not int:
        return None
    if first < 0 or last <= first:
        return None
    return first, last


def calculate_iou(first: MinimalSource, second: MinimalSource) -> float:
    """Calculate intersection over union for two source ranges.

    Ranges use Python's half-open convention: ``[first, last)``.  Invalid or
    empty ranges have an IoU of zero.
    """
    first_range = _source_range(first)
    second_range = _source_range(second)
    if first_range is None or second_range is None:
        return 0.0

    first_start, first_end = first_range
    second_start, second_end = second_range
    intersection = max(
        0,
        min(first_end, second_end) - max(first_start, second_start),
    )
    union = max(first_end, second_end) - min(first_start, second_start)
    if intersection == 0 or union == 0:
        return 0.0
    return intersection / union


def sources_match(
    expected: MinimalSource,
    retrieved: MinimalSource,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> bool:
    """Return whether a retrieved source matches an expected source."""
    threshold = _validate_iou_threshold(iou_threshold)
    if expected.file_path != retrieved.file_path:
        return False

    iou = calculate_iou(expected, retrieved)
    return iou > 0.0 and iou >= threshold


def _count_matched_sources(
    expected_sources: list[MinimalSource],
    retrieved_sources: list[MinimalSource],
    iou_threshold: float,
) -> int:
    """Count expected sources covered by at least one retrieved source."""
    return sum(
        any(
            sources_match(expected, retrieved, iou_threshold)
            for retrieved in retrieved_sources
        )
        for expected in expected_sources
        if _source_range(expected) is not None
    )


def calculate_recall(
    expected_sources: list[MinimalSource],
    retrieved_sources: list[MinimalSource],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> float:
    """Calculate source-level recall for one question.

    Each expected source is counted at most once when any retrieved source
    has the same path and reaches the requested IoU threshold.  If there are
    no valid expected sources, the result is ``0.0``.
    """
    threshold = _validate_iou_threshold(iou_threshold)
    valid_expected_sources = [
        source
        for source in expected_sources
        if _source_range(source) is not None
    ]
    if not valid_expected_sources:
        return 0.0

    matched_sources = _count_matched_sources(
        valid_expected_sources,
        retrieved_sources,
        threshold,
    )
    return matched_sources / len(valid_expected_sources)


def _result_by_question_id(
    search_results: list[MinimalSearchResults],
) -> dict[str, MinimalSearchResults]:
    """Index the first result for each question ID."""
    results_by_question_id: dict[str, MinimalSearchResults] = {}
    for result in search_results:
        results_by_question_id.setdefault(result.question_id, result)
    return results_by_question_id


def evaluate_recall(
    dataset: RagDataset,
    search_results: StudentSearchResults,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> float:
    """Calculate aggregate Recall@k for an answered-question dataset.
    Unanswered questions and unknown question IDs are ignored.  Missing
    results count as zero recall for the corresponding answered question.
    Recall is calculated per question and then averaged, so every answered
    question contributes equally.  ``search_results.k`` limits the retrieved
    sources used for every question.
    """
    threshold = _validate_iou_threshold(iou_threshold)
    results_by_question_id = _result_by_question_id(
        search_results.search_results,
    )
    result_count = search_results.k if type(search_results.k) is int else 0
    result_count = max(result_count, 0)

    question_count = 0
    recall_sum = 0.0
    for question in dataset.rag_questions:
        if not isinstance(question, AnsweredQuestion):
            continue

        expected_sources = [
            source
            for source in question.sources
            if _source_range(source) is not None
        ]
        if not expected_sources:
            continue

        question_count += 1
        result = results_by_question_id.get(question.question_id)
        if result is None:
            continue

        retrieved_sources = result.retrieved_sources[:result_count]
        recall_sum += (
            _count_matched_sources(
                expected_sources,
                retrieved_sources,
                threshold,
            )
            / len(expected_sources)
        )

    if question_count == 0:
        return 0.0
    return recall_sum / question_count


def evaluate(
    student_search_results_path: Path,
    dataset_path: Path,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> float:
    """Load JSON files and calculate aggregate Recall@k."""
    dataset = load_dataset(dataset_path)
    search_results = load_student_search_results(
        student_search_results_path,
    )
    return evaluate_recall(dataset, search_results, iou_threshold)
