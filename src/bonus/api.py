"""Local HTTP API for querying and answering from the RAG index."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
import math
from pathlib import Path
from typing import Any, Mapping, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, ValidationError

from ..mandatory.answerer import AnswerGenerator, NO_CONTEXT_ANSWER
from ..mandatory.models import MinimalSource
from .retriever import retrieve_sources


API_MAX_BODY_BYTES = 1_000_000

__all__ = [
    "APIQuery",
    "AnswerResponse",
    "SearchResponse",
    "create_server",
]


class APIQuery(BaseModel):
    """Represent the common query payload accepted by API endpoints."""

    query: str
    k: int = Field(default=5, ge=0, strict=True)


class SearchResponse(BaseModel):
    """Represent source locations returned by the search endpoint."""

    sources: list[MinimalSource]


class AnswerResponse(BaseModel):
    """Represent a generated answer and its supporting source locations."""

    answer: str
    sources: list[MinimalSource]


class _BadRequestError(ValueError):
    """Represent malformed HTTP request data."""


class _RAGHTTPServer(HTTPServer):
    """Store the RAG services used by request handlers."""

    index_path: Path
    answer_generator: AnswerGenerator
    retrieval_mode: str
    semantic_weight: float


class RAGRequestHandler(BaseHTTPRequestHandler):
    """Handle health, search, and answer HTTP requests."""

    def do_GET(self) -> None:
        """Return the health response for the health endpoint."""
        path = urlsplit(self.path).path
        if path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:
        """Process a search or answer request and return JSON."""
        path = urlsplit(self.path).path
        if path not in ("/search", "/answer"):
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return

        try:
            request = APIQuery.model_validate(self._read_json())
            server = cast(_RAGHTTPServer, self.server)
            sources = retrieve_sources(
                query=request.query,
                k=request.k,
                index_path=server.index_path,
                retrieval_mode=server.retrieval_mode,
                semantic_weight=server.semantic_weight,
            )
            if path == "/search":
                response: BaseModel = SearchResponse(sources=sources)
            else:
                answer = (
                    NO_CONTEXT_ANSWER
                    if not sources
                    else server.answer_generator.generate(
                        request.query,
                        sources,
                    )
                )
                response = AnswerResponse(
                    answer=answer,
                    sources=sources,
                )
            self._send_json(
                HTTPStatus.OK,
                response.model_dump(mode="json"),
            )
        except (ValidationError, _BadRequestError) as error:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": str(error)},
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": str(error)},
            )

    def _read_json(self) -> Any:
        """Read and decode one bounded JSON request body."""
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            raise _BadRequestError("Content-Length header is required")
        try:
            body_length = int(content_length)
        except ValueError as error:
            raise _BadRequestError(
                "Content-Length must be an integer"
            ) from error

        if body_length < 0 or body_length > API_MAX_BODY_BYTES:
            raise _BadRequestError("Request body is too large")

        body = self.rfile.read(body_length)
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise _BadRequestError(
                "Request body must be valid UTF-8 JSON"
            ) from error

    def _send_json(
        self,
        status: HTTPStatus,
        payload: Mapping[str, object],
    ) -> None:
        """Send one JSON response with an explicit content length."""
        body = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    index_path: Path = Path("data/processed"),
    answer_generator: AnswerGenerator | None = None,
    retrieval_mode: str = "bm25",
    semantic_weight: float = 0.5,
) -> HTTPServer:
    """Create a local HTTP server backed by the RAG index."""
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("port must be an integer between 0 and 65535")
    if not isinstance(retrieval_mode, str) or retrieval_mode not in {
        "bm25",
        "semantic",
        "hybrid",
    }:
        raise ValueError(
            "retrieval_mode must be one of bm25, semantic, or hybrid"
        )
    if (
        isinstance(semantic_weight, bool)
        or not isinstance(semantic_weight, (int, float))
        or not math.isfinite(semantic_weight)
        or not 0 <= semantic_weight <= 1
    ):
        raise ValueError("semantic_weight must be finite and between 0 and 1")

    server = _RAGHTTPServer((host, port), RAGRequestHandler)
    server.index_path = index_path
    server.answer_generator = (
        answer_generator
        if answer_generator is not None
        else AnswerGenerator()
    )
    server.retrieval_mode = retrieval_mode
    server.semantic_weight = semantic_weight
    return server
