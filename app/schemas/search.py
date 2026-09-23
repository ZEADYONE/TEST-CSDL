"""Search API schemas."""

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    image_id: int
    file_name: str
    species: str | None
    image_url: str
    similarity: float = Field(ge=0.0, le=1.0)
    color_similarity: float = Field(ge=0.0, le=1.0)
    texture_similarity: float = Field(ge=0.0, le=1.0)
    shape_similarity: float = Field(ge=0.0, le=1.0)


class SearchDebug(BaseModel):
    mask_quality: float
    effective_weights: dict[str, float]


class SearchResponse(BaseModel):
    query_original_url: str
    query_processed_url: str
    query_mask_url: str
    mask_quality: float
    results: list[SearchResult]
    debug: SearchDebug | None = None
