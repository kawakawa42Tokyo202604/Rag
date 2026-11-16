"""Orchestration layer for the RAG command-line interface."""

import json

from pathlib import Path

from tqdm import tqdm

from .answerer import (
    AnswerGenerator,
    DEFAULT_MAX_CONTEXT_CHARACTERS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_NAME,
    NO_CONTEXT_ANSWER,
)
from .evaluator import evaluate as evaluate_results
from .indexer import MAX_CHUNK_SIZE, build_index
from .io import (
    load_dataset,
    load_student_search_results,
    save_student_search_results,
    save_student_search_results_and_answer,
)
from .models import (
    MinimalAnswer,
    MinimalSearchResults,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from .retriever import retrieve_sources


class RAGPipeline:
    """Receive CLI arguments and pass them to each pipeline stage."""

    def index(
        self,
        file_path: str = "data/raw/vllm-0.10.1",
        max_chunk_size: int = MAX_CHUNK_SIZE,
        output_directory: str = "data/processed",
    ) -> None:
        """Create an index from the path received from the CLI.

        Args:
            file_path: File or directory to read as the source corpus.
            max_chunk_size: Maximum number of characters in one chunk.
            output_directory: Directory in which to save the index.
        """
        source_path = Path(file_path)
        index_path = build_index(
            source_path,
            output_directory=Path(output_directory),
            max_chunk_size=max_chunk_size,
        )
        print(f"[index] file_path={source_path}")
        print(f"[index] index_path={index_path}")

    def search(
        self,
        query: str,
        k: int = 5,
        index_path: str = "data/processed",
    ) -> None:
        """Search the temporary index path received from the CLI.

        Args:
            query: Question or keywords to search for.
            k: Number of results to return.
            index_path: Directory containing the saved index.
        """
        sources = retrieve_sources(
            query=query,
            k=k,
            index_path=Path(index_path),
        )
        print(
            json.dumps(
                [source.model_dump() for source in sources],
                ensure_ascii=False,
            )
        )

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        iou_threshold: float = 0.05,
    ) -> None:
        """Evaluate saved retrieval results against an answered dataset."""
        recall = evaluate_results(
            student_search_results_path=Path(student_search_results_path),
            dataset_path=Path(dataset_path),
            iou_threshold=iou_threshold,
        )
        print(f"[evaluate] recall={recall:.4f}")

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 5,
        save_directory: str = "data/output/search_results",
        index_path: str = "data/processed",
    ) -> None:
        """Search every question in a dataset and save the results."""
        if type(k) is not int:
            raise ValueError("k must be an integer")
        if k < 0:
            raise ValueError("k must be non-negative")

        input_path = Path(dataset_path)
        dataset = load_dataset(input_path)
        search_results: list[MinimalSearchResults] = []
        for question in tqdm(
            dataset.rag_questions,
            desc="Searching questions",
        ):
            sources = retrieve_sources(
                query=question.question,
                k=k,
                index_path=Path(index_path),
            )
            search_results.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=sources,
                )
            )

        output_path = Path(save_directory) / input_path.name
        save_student_search_results(
            StudentSearchResults(search_results=search_results, k=k),
            output_path,
        )
        print(f"[search_dataset] output_path={output_path}")

    def answer(
        self,
        query: str,
        k: int = 5,
        index_path: str = "data/processed",
        model_name: str = DEFAULT_MODEL_NAME,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        max_context_characters: int = DEFAULT_MAX_CONTEXT_CHARACTERS,
    ) -> None:
        """Retrieve sources and generate one source-grounded answer."""
        answer_text = answer_question(
            query=query,
            k=k,
            index_path=Path(index_path),
            answer_generator=AnswerGenerator(
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                max_context_characters=max_context_characters,
            ),
        )
        print(answer_text)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
        model_name: str = DEFAULT_MODEL_NAME,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        max_context_characters: int = DEFAULT_MAX_CONTEXT_CHARACTERS,
    ) -> None:
        """Generate answers and save them.

        Stop without partial output when one answer fails.
        """
        input_path = Path(student_search_results_path)
        search_results = load_student_search_results(input_path)
        answer_generator = AnswerGenerator(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            max_context_characters=max_context_characters,
        )
        answers: list[MinimalAnswer] = []

        for result in tqdm(
            search_results.search_results,
            desc="Generating answers",
        ):
            try:
                answer_text = answer_generator.generate(
                    result.question,
                    result.retrieved_sources,
                )
            except RuntimeError as error:
                raise RuntimeError(
                    "Failed to generate answer for question "
                    f"{result.question_id}"
                ) from error
            answers.append(
                MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=answer_text,
                )
            )

        output_path = Path(save_directory) / input_path.name
        save_student_search_results_and_answer(
            StudentSearchResultsAndAnswer(
                search_results=answers,
                k=search_results.k,
            ),
            output_path,
        )
        print(f"[answer_dataset] output_path={output_path}")


def answer_question(
    query: str,
    k: int = 5,
    index_path: Path = Path("data/processed"),
    answer_generator: AnswerGenerator | None = None,
) -> str:
    """Return one answer using retrieved source locations as context."""
    if type(k) is not int:
        raise ValueError("k must be an integer")
    if k < 0:
        raise ValueError("k must be non-negative")

    sources = retrieve_sources(
        query=query,
        k=k,
        index_path=index_path,
    )
    generator = answer_generator or AnswerGenerator()
    if not sources:
        return NO_CONTEXT_ANSWER
    return generator.generate(query, sources)
