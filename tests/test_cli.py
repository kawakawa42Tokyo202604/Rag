"""Tests for CLI-facing compatibility exports."""

import sys
from pathlib import Path

import pytest

from src.mandatory import cli
from src.mandatory.models import MinimalSource


def test_cli_reexports_canonical_models() -> None:
    """Keep the legacy CLI module pointed at the shared model definitions."""
    assert cli.MinimalSource is MinimalSource


def test_cli_reports_missing_index_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Turn a missing index into a concise CLI error."""
    from src.__main__ import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "src",
            "search",
            "query",
            "--k=1",
            f"--index_path={tmp_path / 'missing'}",
        ],
    )

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 1
    assert "Traceback" not in capsys.readouterr().err


def test_cli_reports_non_string_query_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Handle positional values that Python Fire parses as non-strings."""
    from src.__main__ import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "src",
            "search",
            "123",
            "--k=1",
            f"--index_path={tmp_path / 'missing'}",
        ],
    )

    with pytest.raises(SystemExit) as error:
        main()

    stderr = capsys.readouterr().err
    assert error.value.code == 1
    assert "query must be a string" in stderr
    assert "Traceback" not in stderr
