"""FastAPI application – image analysis endpoint."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from app.models import (
    AnalysisResponse,
    FontAlternative,
    FontInfo,
    FontSources,
    ImageMetadata,
    Reconstruction,
    TextBlock,
)
from app.pipeline.font_id import identify_font
from app.pipeline.ocr import run_ocr
from app.pipeline.typography import (
    estimate_font_metrics,
    extract_characters,
    extract_geometry,
    extract_rendering,
)

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}

app = FastAPI(
    title="Image Analysis API",
    description="Analyzes images and returns JSON for near-pixel-perfect reconstruction.",
    version="1.0.0",
)


@app.get("/")
async def root():
    return {"status": "ok", "message": "Image Analysis API is running."}


@app.post("/analyze/image", response_model=AnalysisResponse)
async def analyze_image(
    image: UploadFile = File(...),
    dpi: Optional[int] = Form(None),
    language_hint: Optional[str] = Form(None),
    output_units: Optional[str] = Form("px"),
    preserve_whitespace: Optional[bool] = Form(True),
):
    """Analyze an input image and return structured JSON for reconstruction.

    Pipeline:
    1. OCR text detection & recognition
    2. Font identification on OCR-detected regions
    3. Typography & geometry extraction
    """
    analysis_warnings: list[str] = []

    # --- Validate file extension ---
    filename = image.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # --- Save upload to temp file ---
    contents = await image.read()
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(contents)
    tmp.close()
    tmp_path = tmp.name

    try:
        img = Image.open(tmp_path)
        img_width, img_height = img.size
        color_mode = img.mode  # RGB, RGBA, L, etc.
        if color_mode == "L":
            color_mode = "GRAY"

        detected_dpi = dpi
        if detected_dpi is None:
            info = img.info
            if "dpi" in info:
                detected_dpi = int(info["dpi"][0])
            else:
                detected_dpi = 72

        image_meta = ImageMetadata(
            width=img_width,
            height=img_height,
            dpi=detected_dpi,
            color_mode=color_mode,
        )

        # --- Step 1: OCR ---
        try:
            ocr_blocks = run_ocr(tmp_path, language_hint=language_hint)
        except RuntimeError:
            raise HTTPException(status_code=503, detail="OCR service unavailable")

        if not ocr_blocks:
            analysis_warnings.append("OCR returned no text blocks")
            return AnalysisResponse(
                image_metadata=image_meta,
                blocks=[],
                warnings=analysis_warnings,
            )

        # --- Steps 2 & 3: Font ID + Typography ---
        blocks: list[TextBlock] = []
        for idx, ocr_block in enumerate(ocr_blocks):
            block_id = f"block_{idx + 1:03d}"

            # Geometry
            geometry = extract_geometry(ocr_block, img_width, img_height)

            # Font identification on the cropped region
            font_result = identify_font(img, ocr_block.box)

            # Typography / rendering
            rendering, font_size_px = extract_rendering(ocr_block, img)

            # Font metrics
            metrics = estimate_font_metrics(font_size_px)

            font_info = FontInfo(
                primary=font_result.primary,
                confidence=font_result.confidence,
                alternatives=[
                    FontAlternative(name=a.name, confidence=a.confidence)
                    for a in font_result.alternatives
                ],
                category=font_result.category,
                metrics=metrics,
            )

            if font_result.uncertain:
                analysis_warnings.append(
                    f"Font identification uncertain for {block_id}"
                )

            # Characters
            characters = extract_characters(ocr_block, geometry, font_size_px)

            if not preserve_whitespace:
                text = " ".join(ocr_block.text.split())
            else:
                text = ocr_block.text

            blocks.append(
                TextBlock(
                    id=block_id,
                    text=text,
                    language=ocr_block.language,
                    confidence=ocr_block.confidence,
                    reading_order=ocr_block.reading_order,
                    geometry=geometry,
                    font=font_info,
                    rendering=rendering,
                    characters=characters,
                )
            )

        return AnalysisResponse(
            image_metadata=image_meta,
            blocks=blocks,
            font_sources=FontSources(
                strategy="fallback",
                notes="Embed font when possible to ensure rendering parity",
            ),
            reconstruction=Reconstruction(),
            warnings=analysis_warnings,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during analysis")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        Path(tmp_path).unlink(missing_ok=True)
