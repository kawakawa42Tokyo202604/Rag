"""A small dependency-free BM25 implementation for corpus retrieval."""

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Iterable


_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word and identifier tokens."""
    return _TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class BM25Index:
    """Store the corpus statistics needed to score BM25 queries."""

    document_term_frequencies: list[dict[str, int]]
    document_frequencies: dict[str, int]
    document_lengths: list[int]
    average_document_length: float
    total_documents: int
    k1: float = 1.2
    b: float = 1.0

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[str],
        k1: float = 1.2,
        b: float = 1.0,
    ) -> "BM25Index":
        """Build BM25 statistics from document texts."""
        if (
            isinstance(k1, bool)
            or not isinstance(k1, (int, float))
            or not math.isfinite(k1)
            or k1 < 0
        ):
            raise ValueError("k1 must be finite and non-negative")
        if (
            isinstance(b, bool)
            or not isinstance(b, (int, float))
            or not math.isfinite(b)
            or not 0 <= b <= 1
        ):
            raise ValueError("b must be finite and between 0 and 1")

        term_frequencies: list[dict[str, int]] = []
        document_frequencies: Counter[str] = Counter()
        document_lengths: list[int] = []

        for document in documents:
            tokens = tokenize(document)
            frequencies = Counter(tokens)
            term_frequencies.append(dict(frequencies))
            document_lengths.append(len(tokens))
            document_frequencies.update(frequencies.keys())

        total_documents = len(term_frequencies)
        average_document_length = (
            sum(document_lengths) / total_documents
            if total_documents
            else 0.0
        )
        return cls(
            document_term_frequencies=term_frequencies,
            document_frequencies=dict(document_frequencies),
            document_lengths=document_lengths,
            average_document_length=average_document_length,
            total_documents=total_documents,
            k1=k1,
            b=b,
        )

    def get_scores(self, query: str) -> list[float]:
        """Return one BM25 score for every indexed document."""
        scores = [0.0] * self.total_documents
        if not self.total_documents:
            return scores

        query_terms = set(tokenize(query))
        if not query_terms:
            return scores

        for term in query_terms:
            document_frequency = self.document_frequencies.get(term, 0)
            if document_frequency == 0:
                continue

            inverse_document_frequency = math.log(
                1.0
                + (
                    self.total_documents - document_frequency + 0.5
                )
                / (document_frequency + 0.5)
            )

            for document_index, frequencies in enumerate(
                self.document_term_frequencies
            ):
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue

                document_length = self.document_lengths[document_index]
                length_normalizer = (
                    self.k1
                    * (
                        1.0
                        - self.b
                        + self.b
                        * document_length
                        / self.average_document_length
                    )
                    if self.average_document_length
                    else self.k1
                )
                scores[document_index] += inverse_document_frequency * (
                    term_frequency
                    * (self.k1 + 1.0)
                    / (term_frequency + length_normalizer)
                )

        return scores
