"""Tests for the optional local HTTP API."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from threading import Thread
from typing import Any, cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from src.bonus.api import create_server
from src.mandatory.answerer import AnswerGenerator
from src.mandatory.indexer import build_index
from src.mandatory.models import MinimalSource


class _FakeAnswerGenerator:
    """Small answer-generator double for API tests."""

    def generate(
        self,
        question: str,
        sources: list[MinimalSource],
    ) -> str:
        return f"answer:{question}:{len(sources)}"


@pytest.fixture
def api_url(tmp_path: Path) -> Iterator[str]:
    """Start a local API server backed by a temporary index."""
    source_path = tmp_path / "example.py"
    source_path.write_text("trust_remote_code = False\n", encoding="utf-8")
    index_path = build_index(
        source_path,
        output_directory=tmp_path / "processed",
    )
    server = create_server(
        host="127.0.0.1",
        port=0,
        index_path=index_path,
        answer_generator=cast(
            AnswerGenerator,
            _FakeAnswerGenerator(),
        ),
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server_host = server.server_address[0]
    if isinstance(server_host, bytes):
        server_host = server_host.decode("ascii")
    server_port = server.server_address[1]
    try:
        yield f"http://{server_host}:{server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _request(
    url: str,
    method: str = "GET",
    payload: dict[str, object] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Send one JSON API request and decode its response."""
    data = (
        None
        if payload is None
        else json.dumps(payload).encode("utf-8")
    )
    request = Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urlopen(request) as response:
        return response.status, json.loads(response.read())


def test_health_endpoint_returns_ok(api_url: str) -> None:
    """Expose a simple health check without requiring an index query."""
    status, payload = _request(f"{api_url}/health")

    assert status == 200
    assert payload == {"status": "ok"}


def test_search_endpoint_returns_sources(api_url: str) -> None:
    """Return ranked source locations from a JSON search request."""
    status, payload = _request(
        f"{api_url}/search",
        method="POST",
        payload={"query": "trust_remote_code", "k": 1},
    )

    assert status == 200
    assert len(payload["sources"]) == 1
    assert payload["sources"][0]["last_character_index"] > 0


def test_answer_endpoint_returns_answer_and_sources(api_url: str) -> None:
    """Generate an answer through the HTTP endpoint."""
    status, payload = _request(
        f"{api_url}/answer",
        method="POST",
        payload={"query": "trust_remote_code", "k": 1},
    )

    assert status == 200
    assert payload["answer"] == "answer:trust_remote_code:1"
    assert len(payload["sources"]) == 1


def test_api_rejects_invalid_request_payload(api_url: str) -> None:
    """Return HTTP 400 instead of crashing for malformed request data."""
    request = Request(
        f"{api_url}/search",
        data=b'{"k": 1}',
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with pytest.raises(HTTPError) as error:
        urlopen(request)

    assert error.value.code == 400
    assert "query" in json.loads(error.value.read())["error"]


def test_create_server_rejects_invalid_port() -> None:
    """Reject ports outside the TCP range before opening a socket."""
    with pytest.raises(ValueError, match="port"):
        create_server(port=65536)
