"""CLI orchestration for the optional bonus features."""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from ..mandatory.answerer import (
    AnswerGenerator,
    DEFAULT_MAX_CONTEXT_CHARACTERS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_NAME,
    NO_CONTEXT_ANSWER,
)
from ..mandatory.generator import RAGPipeline as MandatoryRAGPipeline
from ..mandatory.io import load_dataset
from ..mandatory.models import MinimalSearchResults, StudentSearchResults
from .api import create_server
from .indexer import MAX_CHUNK_SIZE, build_index
from .retriever import retrieve_sources


class RAGPipeline(MandatoryRAGPipeline):
    """Expose semantic, incremental, hybrid, cache, and API operations."""

    def index(
        self,
        file_path: str = "data/raw/vllm-0.10.1",
        max_chunk_size: int = MAX_CHUNK_SIZE,
        output_directory: str = "data/processed",
        semantic_model_name: str | None = None,
        incremental: bool = False,
    ) -> None:
        """Build an optional semantic and/or incremental index."""
        source_path = Path(file_path)
        index_path = build_index(
            source_path,
            output_directory=Path(output_directory),
            max_chunk_size=max_chunk_size,
            semantic_model_name=semantic_model_name,
            incremental=incremental,
        )
        print(f"[index] file_path={source_path}")
        print(f"[index] index_path={index_path}")

    def search(
        self,
        query: str,
        k: int = 5,
        index_path: str = "data/processed",
        retrieval_mode: str = "bm25",
        semantic_weight: float = 0.5,
    ) -> None:
        """Search with BM25, semantic, or hybrid retrieval."""
        sources = retrieve_sources(
            query=query,
            k=k,
            index_path=Path(index_path),
            retrieval_mode=retrieval_mode,
            semantic_weight=semantic_weight,
        )
        print(
            json.dumps(
                [source.model_dump() for source in sources],
                ensure_ascii=False,
            )
        )

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 5,
        save_directory: str = "data/output/search_results",
        index_path: str = "data/processed",
        retrieval_mode: str = "bm25",
        semantic_weight: float = 0.5,
    ) -> None:
        """Run optional retrieval over every question in a dataset."""
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
                retrieval_mode=retrieval_mode,
                semantic_weight=semantic_weight,
            )
            search_results.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=sources,
                )
            )

        output_path = Path(save_directory) / input_path.name
        from ..mandatory.io import save_student_search_results

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
        retrieval_mode: str = "bm25",
        semantic_weight: float = 0.5,
    ) -> None:
        """Generate an answer from optional retrieval results."""
        answer_text = answer_question(
            query=query,
            k=k,
            index_path=Path(index_path),
            answer_generator=AnswerGenerator(
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                max_context_characters=max_context_characters,
            ),
            retrieval_mode=retrieval_mode,
            semantic_weight=semantic_weight,
        )
        print(answer_text)

    def serve(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        index_path: str = "data/processed",
        model_name: str = DEFAULT_MODEL_NAME,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        max_context_characters: int = DEFAULT_MAX_CONTEXT_CHARACTERS,
        retrieval_mode: str = "bm25",
        semantic_weight: float = 0.5,
    ) -> None:
        """Start the optional local HTTP API."""
        server = create_server(
            host=host,
            port=port,
            index_path=Path(index_path),
            answer_generator=AnswerGenerator(
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                max_context_characters=max_context_characters,
            ),
            retrieval_mode=retrieval_mode,
            semantic_weight=semantic_weight,
        )
        print(f"[serve] listening=http://{host}:{port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("[serve] stopped")
        finally:
            server.server_close()


def answer_question(
    query: str,
    k: int = 5,
    index_path: Path = Path("data/processed"),
    answer_generator: AnswerGenerator | None = None,
    retrieval_mode: str = "bm25",
    semantic_weight: float = 0.5,
) -> str:
    """Return one answer using optional retrieval results as context."""
    if type(k) is not int:
        raise ValueError("k must be an integer")
    if k < 0:
        raise ValueError("k must be non-negative")

    sources = retrieve_sources(
        query=query,
        k=k,
        index_path=index_path,
        retrieval_mode=retrieval_mode,
        semantic_weight=semantic_weight,
    )
    generator = answer_generator or AnswerGenerator()
    if not sources:
        return NO_CONTEXT_ANSWER
    return generator.generate(query, sources)
