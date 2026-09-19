"""Expose the required python -m researcher command."""

from src.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
