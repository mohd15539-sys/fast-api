"""Tests for POST /analyze/image endpoint."""

from __future__ import annotations

import io
from unittest.mock import patch

from PIL import Image

from app.pipeline.font_id import FontCandidate, FontResult
from app.pipeline.ocr import OCRBlock, OCRWord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_ocr_blocks():
    """Return a deterministic list of OCRBlock objects."""
    return [
        OCRBlock(
            text="Hello World",
            words=[
                OCRWord(text="Hello", box=[10, 10, 80, 40], confidence=0.95),
                OCRWord(text="World", box=[90, 10, 160, 40], confidence=0.93),
            ],
            box=[10, 10, 160, 40],
            confidence=0.94,
            language="en",
            reading_order=0,
        ),
    ]


def _mock_font_result(*_args, **_kwargs):
    return FontResult(
        primary="Helvetica",
        confidence=0.91,
        alternatives=[FontCandidate(name="Arial", confidence=0.62)],
        category="sans",
        uncertain=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRootEndpoint:
    def test_root_returns_ok(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestAnalyzeImageEndpoint:
    @patch("app.main.identify_font", side_effect=_mock_font_result)
    @patch("app.main.run_ocr", return_value=_mock_ocr_blocks())
    def test_successful_analysis(self, mock_ocr, mock_font, client, sample_png_bytes):
        resp = client.post(
            "/analyze/image",
            files={"image": ("test.png", sample_png_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()

        # Top-level keys
        assert "image_metadata" in data
        assert "blocks" in data
        assert "font_sources" in data
        assert "reconstruction" in data
        assert "warnings" in data

        # Image metadata
        meta = data["image_metadata"]
        assert meta["width"] == 200
        assert meta["height"] == 60
        assert meta["color_mode"] == "RGB"
        assert isinstance(meta["dpi"], int)

    @patch("app.main.identify_font", side_effect=_mock_font_result)
    @patch("app.main.run_ocr", return_value=_mock_ocr_blocks())
    def test_block_structure(self, mock_ocr, mock_font, client, sample_png_bytes):
        resp = client.post(
            "/analyze/image",
            files={"image": ("test.png", sample_png_bytes, "image/png")},
        )
        data = resp.json()
        blocks = data["blocks"]
        assert len(blocks) == 1

        block = blocks[0]
        assert block["id"] == "block_001"
        assert block["text"] == "Hello World"
        assert block["language"] == "en"
        assert "geometry" in block
        assert "font" in block
        assert "rendering" in block
        assert "characters" in block

    @patch("app.main.identify_font", side_effect=_mock_font_result)
    @patch("app.main.run_ocr", return_value=_mock_ocr_blocks())
    def test_font_info_populated(self, mock_ocr, mock_font, client, sample_png_bytes):
        resp = client.post(
            "/analyze/image",
            files={"image": ("test.png", sample_png_bytes, "image/png")},
        )
        font = resp.json()["blocks"][0]["font"]
        assert font["primary"] == "Helvetica"
        assert font["confidence"] == 0.91
        assert len(font["alternatives"]) == 1
        assert font["alternatives"][0]["name"] == "Arial"
        assert "metrics" in font

    @patch("app.main.identify_font", side_effect=_mock_font_result)
    @patch("app.main.run_ocr", return_value=_mock_ocr_blocks())
    def test_geometry_fields(self, mock_ocr, mock_font, client, sample_png_bytes):
        resp = client.post(
            "/analyze/image",
            files={"image": ("test.png", sample_png_bytes, "image/png")},
        )
        geom = resp.json()["blocks"][0]["geometry"]
        assert "bounding_box" in geom
        bb = geom["bounding_box"]
        assert all(k in bb for k in ("x", "y", "width", "height"))
        assert "baseline" in geom
        assert len(geom["baseline"]) == 4

    @patch("app.main.identify_font", side_effect=_mock_font_result)
    @patch("app.main.run_ocr", return_value=_mock_ocr_blocks())
    def test_characters_present(self, mock_ocr, mock_font, client, sample_png_bytes):
        resp = client.post(
            "/analyze/image",
            files={"image": ("test.png", sample_png_bytes, "image/png")},
        )
        chars = resp.json()["blocks"][0]["characters"]
        assert len(chars) == len("Hello World")
        first = chars[0]
        assert first["char"] == "H"
        assert "box" in first
        assert "advance_width" in first

    @patch("app.main.identify_font", side_effect=_mock_font_result)
    @patch("app.main.run_ocr", return_value=_mock_ocr_blocks())
    def test_rendering_fields(self, mock_ocr, mock_font, client, sample_png_bytes):
        resp = client.post(
            "/analyze/image",
            files={"image": ("test.png", sample_png_bytes, "image/png")},
        )
        rendering = resp.json()["blocks"][0]["rendering"]
        assert "font_size_px" in rendering
        assert "line_height_px" in rendering
        assert "letter_spacing_px" in rendering
        assert "word_spacing_px" in rendering
        assert "fill_color" in rendering
        assert rendering["fill_color"].startswith("#")

    def test_unsupported_format_rejected(self, client):
        resp = client.post(
            "/analyze/image",
            files={"image": ("test.gif", b"GIF89a", "image/gif")},
        )
        assert resp.status_code == 400

    @patch("app.main.run_ocr", side_effect=RuntimeError("service down"))
    def test_ocr_failure_returns_503(self, mock_ocr, client, sample_png_bytes):
        resp = client.post(
            "/analyze/image",
            files={"image": ("test.png", sample_png_bytes, "image/png")},
        )
        assert resp.status_code == 503

    @patch("app.main.identify_font", side_effect=_mock_font_result)
    @patch("app.main.run_ocr", return_value=[])
    def test_no_text_detected(self, mock_ocr, mock_font, client, sample_png_bytes):
        resp = client.post(
            "/analyze/image",
            files={"image": ("test.png", sample_png_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["blocks"] == []
        assert any("no text" in w.lower() for w in data["warnings"])

    @patch("app.main.identify_font", side_effect=_mock_font_result)
    @patch("app.main.run_ocr", return_value=_mock_ocr_blocks())
    def test_optional_params(self, mock_ocr, mock_font, client, sample_png_bytes):
        resp = client.post(
            "/analyze/image",
            files={"image": ("test.png", sample_png_bytes, "image/png")},
            data={
                "dpi": "300",
                "language_hint": "en,fr",
                "output_units": "px",
                "preserve_whitespace": "true",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["image_metadata"]["dpi"] == 300

    @patch("app.main.identify_font", side_effect=_mock_font_result)
    @patch("app.main.run_ocr", return_value=_mock_ocr_blocks())
    def test_reconstruction_field(self, mock_ocr, mock_font, client, sample_png_bytes):
        resp = client.post(
            "/analyze/image",
            files={"image": ("test.png", sample_png_bytes, "image/png")},
        )
        data = resp.json()
        assert data["reconstruction"]["guarantee"] == "near-pixel-perfect"
        assert "canvas" in data["reconstruction"]["supported_renderers"]

    @patch("app.main.identify_font", side_effect=_mock_font_result)
    @patch("app.main.run_ocr", return_value=_mock_ocr_blocks())
    def test_font_sources_field(self, mock_ocr, mock_font, client, sample_png_bytes):
        resp = client.post(
            "/analyze/image",
            files={"image": ("test.png", sample_png_bytes, "image/png")},
        )
        fs = resp.json()["font_sources"]
        assert fs["strategy"] in ("embedded", "referenced", "fallback")

    @patch("app.main.identify_font", side_effect=_mock_font_result)
    @patch("app.main.run_ocr", return_value=_mock_ocr_blocks())
    def test_jpg_accepted(self, mock_ocr, mock_font, client, sample_jpg_bytes):
        resp = client.post(
            "/analyze/image",
            files={"image": ("test.jpg", sample_jpg_bytes, "image/jpeg")},
        )
        assert resp.status_code == 200
