"""JSON and image endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import distinct, func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import ImageFeature, ImageRecord
from app.schemas.search import SearchResponse
from app.services.color_features import COLOR_DIM
from app.services.feature_extractor import SHAPE_DIM, TEXTURE_DIM
from app.services.search_service import search_similar_images


router = APIRouter(prefix="/api", tags=["api"])


def _read_upload(image: UploadFile) -> bytes:
    settings = get_settings()
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image uploads are accepted")
    data = image.file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image exceeds {settings.max_upload_mb} MB")
    return data


@router.post("/search", response_model=SearchResponse)
def search_api(
    image: UploadFile = File(...),
    top_k: int = Form(10),
    flip_horizontal: bool = Form(False),
    session: Session = Depends(get_db),
) -> SearchResponse:
    if top_k not in {5, 10, 20}:
        raise HTTPException(status_code=422, detail="top_k must be 5, 10, or 20")
    try:
        return search_similar_images(
            session,
            _read_upload(image),
            top_k=top_k,
            flip_horizontal=flip_horizontal,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (FileNotFoundError, LookupError, SQLAlchemyError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/images/{image_id}")
def image_file(image_id: int, session: Session = Depends(get_db)) -> FileResponse:
    record = session.get(ImageRecord, image_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Image not found")
    path = Path(record.processed_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Processed image file is missing")
    return FileResponse(path, media_type="image/png")


@router.get("/health")
def health(session: Session = Depends(get_db)) -> dict[str, object]:
    settings = get_settings()
    try:
        session.execute(text("SELECT 1"))
        database = "connected"
    except Exception:
        database = "unavailable"
    return {
        "status": "ok" if database == "connected" else "degraded",
        "database": database,
        "feature_version": settings.feature_version,
        "search_artifacts_ready": (
            (settings.scaler_dir / "glcm_scaler.joblib").is_file()
            and settings.distance_stats_path.is_file()
        ),
    }


@router.get("/stats")
def stats(session: Session = Depends(get_db)) -> dict[str, object]:
    settings = get_settings()
    total = session.scalar(select(func.count()).select_from(ImageRecord)) or 0
    species_count = session.scalar(
        select(func.count(distinct(ImageRecord.species))).where(ImageRecord.species.is_not(None))
    ) or 0
    return {
        "total_images": total,
        "species_count": species_count,
        "feature_version": settings.feature_version,
        "dimensions": {"color": COLOR_DIM, "texture": TEXTURE_DIM, "shape": SHAPE_DIM},
    }
