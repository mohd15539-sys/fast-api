"""Pydantic models for the image analysis API request and response."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class Geometry(BaseModel):
    bounding_box: BoundingBox
    baseline: List[float] = Field(
        ..., description="[x1, y1, x2, y2] baseline coordinates"
    )
    rotation: float = 0.0
    alignment: str = "left"


class FontAlternative(BaseModel):
    name: str
    confidence: float


class FontMetrics(BaseModel):
    ascender_px: float
    descender_px: float
    cap_height_px: float
    x_height_px: float
    units_per_em: int = 1000
    scale_factor: float = 1.0


class FontInfo(BaseModel):
    primary: str
    confidence: float
    alternatives: List[FontAlternative] = []
    category: Optional[str] = None
    metrics: FontMetrics


class Rendering(BaseModel):
    font_size_px: float
    line_height_px: float
    letter_spacing_px: float
    word_spacing_px: float
    fill_color: str = "#000000"
    antialiasing: str = "grayscale"
    hinting: str = "none"


class CharacterInfo(BaseModel):
    char: str
    box: List[float] = Field(
        ..., description="[x1, y1, x2, y2] bounding box"
    )
    advance_width: float
    baseline_offset: float = 0.0


class TextBlock(BaseModel):
    id: str
    text: str
    language: str = "en"
    confidence: float = 0.0
    reading_order: int = 0
    geometry: Geometry
    font: FontInfo
    rendering: Rendering
    characters: List[CharacterInfo] = []


class ImageMetadata(BaseModel):
    width: int
    height: int
    dpi: int = 72
    color_mode: str = "RGB"


class FontSources(BaseModel):
    strategy: str = "fallback"
    notes: str = "Embed font when possible to ensure rendering parity"


class Reconstruction(BaseModel):
    guarantee: str = "near-pixel-perfect"
    supported_renderers: List[str] = ["canvas", "svg", "pdf", "html"]


class AnalysisResponse(BaseModel):
    image_metadata: ImageMetadata
    blocks: List[TextBlock] = []
    font_sources: FontSources = FontSources()
    reconstruction: Reconstruction = Reconstruction()
    warnings: List[str] = []
