"""Create PostgreSQL tables for the current feature version."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import create_database_if_missing, create_tables  # noqa: E402


if __name__ == "__main__":
    created = create_database_if_missing()
    print("Created database bird_cbir" if created else "Database already exists")
    create_tables()
    print("Created/verified tables: images, image_features")
