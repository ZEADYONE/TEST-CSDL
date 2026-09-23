"""Image API schemas."""

from pydantic import BaseModel, ConfigDict


class ImageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    file_name: str
    species: str | None
    width: int
    height: int
    image_url: str
