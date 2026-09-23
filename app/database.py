"""SQLAlchemy engine and session lifecycle."""

from __future__ import annotations

from collections.abc import Generator

import psycopg
from psycopg import sql
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    """Declarative model base."""


settings = get_settings()
engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that owns one database session."""

    with SessionLocal() as session:
        yield session


def create_tables() -> None:
    """Create the current schema. Import models first so metadata is populated."""

    from app.models import ImageFeature, ImageRecord  # noqa: F401

    Base.metadata.create_all(bind=engine)


def create_database_if_missing() -> bool:
    """Create the configured PostgreSQL database through the maintenance DB.

    Returns ``True`` when a new database was created and ``False`` when it
    already existed. The configured PostgreSQL user needs ``CREATEDB`` rights.
    """

    url = make_url(settings.database_url)
    database_name = url.database
    if not database_name:
        raise ValueError("DATABASE_URL does not contain a database name")
    if database_name == "postgres":
        return False

    connection_options: dict[str, object] = {
        "dbname": "postgres",
        "user": url.username,
        "password": url.password,
        "host": url.host or "localhost",
        "port": url.port or 5432,
        "connect_timeout": 5,
        "autocommit": True,
    }
    with psycopg.connect(**connection_options) as connection:
        exists = connection.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (database_name,)
        ).fetchone()
        if exists:
            return False
        connection.execute(
            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name))
        )
    return True
