"""Feature-vector model using native PostgreSQL REAL arrays."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import ARRAY, REAL
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class ImageFeature(Base):
    __tablename__ = "image_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), nullable=False, unique=True, index=True
    )
    color_vector: Mapped[list[float]] = mapped_column(ARRAY(REAL), nullable=False)
    texture_vector: Mapped[list[float]] = mapped_column(ARRAY(REAL), nullable=False)
    shape_vector: Mapped[list[float]] = mapped_column(ARRAY(REAL), nullable=False)
    feature_version: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    image: Mapped["ImageRecord"] = relationship(back_populates="feature")
