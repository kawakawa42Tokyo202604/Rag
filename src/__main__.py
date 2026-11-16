"""Command-line entry point for the RAG pipeline."""

import sys

from fire import Fire

from .mandatory.generator import RAGPipeline


def main() -> None:
    """Expose RAGPipeline methods as Fire CLI commands."""
    try:
        Fire(RAGPipeline())
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
