"""Image metadata model."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class ImageRecord(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    file_name: Mapped[str] = mapped_column(String(512), nullable=False)
    file_path: Mapped[str] = mapped_column(String(2048), nullable=False, unique=True, index=True)
    processed_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    species: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    feature: Mapped["ImageFeature"] = relationship(
        back_populates="image", cascade="all, delete-orphan", uselist=False
    )
