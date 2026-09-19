"""Support running the CLI through the existing src package."""

from src.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
