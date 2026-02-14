"""Typography and geometry extraction from OCR results and image data."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from PIL import Image

from app.models import (
    BoundingBox,
    CharacterInfo,
    FontMetrics,
    Geometry,
    Rendering,
)
from app.pipeline.ocr import OCRBlock


def _dominant_color(image: Image.Image, box: List[float]) -> str:
    """Return the dominant (most common) colour in the region as a hex string."""
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2 = min(x2, image.width)
    y2 = min(y2, image.height)
    if x2 <= x1 or y2 <= y1:
        return "#000000"

    crop = image.crop((x1, y1, x2, y2)).convert("RGB")
    arr = np.array(crop).reshape(-1, 3)
    # simple approach: find darkest colour cluster (text is usually dark)
    dark_mask = arr.sum(axis=1) < 384  # rough threshold
    if dark_mask.any():
        mean_col = arr[dark_mask].mean(axis=0).astype(int)
    else:
        mean_col = arr.mean(axis=0).astype(int)

    return "#{:02x}{:02x}{:02x}".format(*mean_col)


def extract_geometry(block: OCRBlock, img_width: int, img_height: int) -> Geometry:
    x1, y1, x2, y2 = block.box
    width = x2 - x1
    height = y2 - y1
    baseline_y = y1 + height * 0.85

    return Geometry(
        bounding_box=BoundingBox(x=x1, y=y1, width=width, height=height),
        baseline=[x1, baseline_y, x2, baseline_y],
        rotation=0.0,
        alignment="left",
    )


def estimate_font_metrics(font_size_px: float) -> FontMetrics:
    """Estimate standard font metrics from the font size."""
    return FontMetrics(
        ascender_px=round(font_size_px * 0.8, 2),
        descender_px=round(-font_size_px * 0.2, 2),
        cap_height_px=round(font_size_px * 0.7, 2),
        x_height_px=round(font_size_px * 0.48, 2),
        units_per_em=1000,
        scale_factor=round(font_size_px / 1000.0 * 1000 / font_size_px, 2)
        if font_size_px > 0
        else 1.0,
    )


def extract_rendering(
    block: OCRBlock, image: Image.Image
) -> Tuple[Rendering, float]:
    """Compute rendering attributes; returns (Rendering, font_size_px)."""
    x1, y1, x2, y2 = block.box
    height = y2 - y1
    font_size_px = round(height * 0.75, 2) if height > 0 else 12.0
    line_height_px = round(height, 2)

    text = block.text
    n_chars = max(len(text), 1)
    width = x2 - x1

    letter_spacing = round((width / n_chars) - font_size_px * 0.6, 2)
    if letter_spacing < 0:
        letter_spacing = 0.0

    words = text.split()
    n_spaces = max(len(words) - 1, 1)
    total_char_width = n_chars * font_size_px * 0.6
    word_spacing = round((width - total_char_width) / n_spaces, 2)
    if word_spacing < 0:
        word_spacing = round(font_size_px * 0.25, 2)

    fill_color = _dominant_color(image, block.box)

    rendering = Rendering(
        font_size_px=font_size_px,
        line_height_px=line_height_px,
        letter_spacing_px=letter_spacing,
        word_spacing_px=word_spacing,
        fill_color=fill_color,
        antialiasing="grayscale",
        hinting="none",
    )
    return rendering, font_size_px


def extract_characters(
    block: OCRBlock, geometry: Geometry, font_size_px: float
) -> List[CharacterInfo]:
    """Generate per-character bounding boxes spread across the block."""
    text = block.text
    if not text:
        return []

    bb = geometry.bounding_box
    x_start = bb.x
    y_start = bb.y
    total_width = bb.width
    height = bb.height

    advance = total_width / max(len(text), 1)

    chars: List[CharacterInfo] = []
    for i, ch in enumerate(text):
        cx1 = round(x_start + i * advance, 2)
        cy1 = round(y_start, 2)
        cx2 = round(cx1 + advance, 2)
        cy2 = round(y_start + height, 2)
        chars.append(
            CharacterInfo(
                char=ch,
                box=[cx1, cy1, cx2, cy2],
                advance_width=round(advance, 2),
                baseline_offset=0.0,
            )
        )
    return chars
