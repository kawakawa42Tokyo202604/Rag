"""Read and write JSON data exchanged by the RAG pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .models import (
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)


__all__ = [
    "load_dataset",
    "load_student_search_results",
    "load_student_search_results_and_answer",
    "save_student_search_results",
    "save_student_search_results_and_answer",
]


ModelT = TypeVar("ModelT", bound=BaseModel)


def _load_model(path: Path, model_type: type[ModelT]) -> ModelT:
    """Load and validate one Pydantic model from a UTF-8 JSON file."""
    try:
        with path.open("r", encoding="utf-8") as json_file:
            payload = json.load(json_file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON: {path}") from error
    except UnicodeError as error:
        raise ValueError(f"Unable to decode JSON as UTF-8: {path}") from error

    try:
        return model_type.model_validate(payload)
    except ValidationError as error:
        raise ValueError(
            f"Invalid {model_type.__name__} data: {path}"
        ) from error


def _save_model(model: BaseModel, path: Path) -> Path:
    """Save a Pydantic model as indented UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(
            model.model_dump(mode="json"),
            json_file,
            ensure_ascii=False,
            indent=2,
        )
        json_file.write("\n")
    return path


def load_dataset(dataset_path: Path) -> RagDataset:
    """Load and validate an answered or unanswered RAG dataset."""
    return _load_model(dataset_path, RagDataset)


def load_student_search_results(
    results_path: Path,
) -> StudentSearchResults:
    """Load and validate batched student search results."""
    return _load_model(results_path, StudentSearchResults)


def load_student_search_results_and_answer(
    results_path: Path,
) -> StudentSearchResultsAndAnswer:
    """Load and validate batched search results with generated answers."""
    return _load_model(results_path, StudentSearchResultsAndAnswer)


def save_student_search_results(
    results: StudentSearchResults,
    output_path: Path,
) -> Path:
    """Save validated batched student search results as JSON."""
    return _save_model(results, output_path)


def save_student_search_results_and_answer(
    results: StudentSearchResultsAndAnswer,
    output_path: Path,
) -> Path:
    """Save validated batched search results and answers as JSON."""
    return _save_model(results, output_path)
