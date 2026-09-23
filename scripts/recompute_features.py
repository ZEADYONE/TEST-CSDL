"""Recompute current-version features and dataset normalization for supplied roots."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.index_dataset import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
