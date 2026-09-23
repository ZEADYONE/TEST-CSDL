"""Server-rendered web UI."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import distinct, func, select
from sqlalchemy.orm import Session

from app.config import PROJECT_ROOT
from app.database import get_db
from app.models import ImageRecord
from app.routes.api import _read_upload
from app.services.search_service import search_similar_images


router = APIRouter(tags=["web"])
templates = Jinja2Templates(directory=PROJECT_ROOT / "app" / "templates")


def _stats(session: Session) -> dict[str, int]:
    return {
        "total_images": session.scalar(select(func.count()).select_from(ImageRecord)) or 0,
        "species_count": session.scalar(
            select(func.count(distinct(ImageRecord.species))).where(ImageRecord.species.is_not(None))
        )
        or 0,
    }


@router.get("/", response_class=HTMLResponse)
def index(request: Request, session: Session = Depends(get_db)) -> HTMLResponse:
    try:
        stats = _stats(session)
        error = None
    except Exception:
        stats = {"total_images": 0, "species_count": 0}
        error = "Chưa kết nối được PostgreSQL. Hãy kiểm tra DATABASE_URL và khởi tạo database."
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"stats": stats, "error": error},
    )


@router.post("/search", response_class=HTMLResponse)
def search_page(
    request: Request,
    image: UploadFile = File(...),
    top_k: int = Form(10),
    flip_horizontal: bool = Form(False),
    session: Session = Depends(get_db),
) -> HTMLResponse:
    try:
        if top_k not in {5, 10, 20}:
            raise ValueError("Top K không hợp lệ")
        response = search_similar_images(
            session,
            _read_upload(image),
            top_k=top_k,
            flip_horizontal=flip_horizontal,
        )
        return templates.TemplateResponse(
            request=request,
            name="results.html",
            context={"search": response},
        )
    except Exception as exc:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"stats": {"total_images": 0, "species_count": 0}, "error": str(exc)},
            status_code=422,
        )
