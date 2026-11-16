"""Pydantic models exchanged between the RAG pipeline stages."""

from uuid import uuid4
from typing import Any

from pydantic import BaseModel, Field, model_validator


__all__ = [
    "AnsweredQuestion",
    "MinimalAnswer",
    "MinimalSearchResults",
    "MinimalSource",
    "RagDataset",
    "StudentSearchResults",
    "StudentSearchResultsAndAnswer",
    "UnansweredQuestion",
]


def _new_question_id() -> str:
    """Return a UUID string for a question without an explicit identifier."""
    return str(uuid4())


class MinimalSource(BaseModel):
    """Represent the location of a source snippet in the indexed corpus."""

    file_path: str
    first_character_index: int = Field(ge=0, strict=True)
    last_character_index: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def _validate_character_range(self) -> "MinimalSource":
        """Reject empty or reversed source ranges."""
        if self.last_character_index <= self.first_character_index:
            raise ValueError(
                "last_character_index must be greater than "
                "first_character_index"
            )
        return self


class UnansweredQuestion(BaseModel):
    """Represent a question that has not yet been answered."""

    question_id: str = Field(default_factory=_new_question_id)
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a question with its reference sources and answer."""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represent a collection of answered or unanswered RAG questions."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]

    @model_validator(mode="before")
    @classmethod
    def _validate_question_variants(cls, value: Any) -> Any:
        """Reject records that partially declare an answered question."""
        if not isinstance(value, dict):
            return value

        questions = value.get("rag_questions")
        if not isinstance(questions, list):
            return value

        for question in questions:
            if not isinstance(question, dict):
                continue
            has_sources = "sources" in question
            has_answer = "answer" in question
            if has_sources != has_answer:
                raise ValueError(
                    "Answered question must include both sources and answer"
                )
            if has_sources:
                AnsweredQuestion.model_validate(question)
        return value


class MinimalSearchResults(BaseModel):
    """Represent retrieved source locations for one question."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Represent retrieved sources and a generated answer for one question."""

    answer: str


class StudentSearchResults(BaseModel):
    """Represent batched retrieval output and its requested result count."""

    search_results: list[MinimalSearchResults]
    k: int = Field(ge=0, strict=True)


class StudentSearchResultsAndAnswer(BaseModel):
    """Represent batched retrieval and answer-generation output."""

    search_results: list[MinimalAnswer]
    k: int = Field(ge=0, strict=True)
