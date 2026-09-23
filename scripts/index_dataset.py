"""Command-line dataset indexer."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import SessionLocal, create_tables  # noqa: E402
from app.services.indexing import index_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract handcrafted bird features into PostgreSQL")
    parser.add_argument(
        "--dataset",
        type=Path,
        nargs="+",
        required=True,
        help="One or more dataset roots; immediate parent folder is used as species",
    )
    parser.add_argument(
        "--already-preprocessed",
        action="store_true",
        help="Use alpha/constant background mask instead of running YOLO-Seg",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    create_tables()
    with SessionLocal() as session:
        summary = index_dataset(
            session,
            args.dataset,
            already_preprocessed=args.already_preprocessed,
        )
    print(
        f"Done: discovered={summary.discovered}, indexed={summary.indexed}, failed={summary.failed}"
    )
    return 0 if summary.failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
