"""Tests for the project development environment."""

import importlib.util
import sys


RUNTIME_MODULES = (
    "fire",
    "pydantic",
    "sklearn",
    "torch",
    "tqdm",
    "transformers",
)


def test_supported_python_version() -> None:
    """Verify that the interpreter satisfies the project requirement."""
    assert sys.version_info >= (3, 10)


def test_src_package_is_importable() -> None:
    """Verify that the CLI package can be discovered from the project root."""
    assert importlib.util.find_spec("src") is not None


def test_runtime_dependencies_are_available() -> None:
    """Verify that every declared runtime dependency can be discovered."""
    missing_modules = [
        module_name
        for module_name in RUNTIME_MODULES
        if importlib.util.find_spec(module_name) is None
    ]
    assert missing_modules == []
