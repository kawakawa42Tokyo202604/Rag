"""Command-line entry point for the optional bonus pipeline."""

import sys

from fire import Fire

from .generator import RAGPipeline


def main() -> None:
    """Expose bonus pipeline methods as Fire CLI commands."""
    try:
        Fire(RAGPipeline())
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
