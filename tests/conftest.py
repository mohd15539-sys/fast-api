"""Shared test fixtures."""

from __future__ import annotations

import io
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def sample_png_bytes() -> bytes:
    """Create a small in-memory PNG with text-like content."""
    img = Image.new("RGB", (200, 60), color=(255, 255, 255))
    # Draw a simple dark rectangle to simulate text
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    draw.text((10, 15), "Hello", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture()
def sample_jpg_bytes() -> bytes:
    img = Image.new("RGB", (100, 40), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()
