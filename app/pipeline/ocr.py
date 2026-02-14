"""OCR pipeline using Hugging Face Image-to-Multilingual-OCR space."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from gradio_client import Client, handle_file

logger = logging.getLogger(__name__)

HF_OCR_SPACE = "awacke1/Image-to-Multilingual-OCR"


@dataclass
class OCRWord:
    text: str
    box: List[float]  # [x1, y1, x2, y2]
    confidence: float = 0.0
    language: str = "en"


@dataclass
class OCRBlock:
    text: str
    words: List[OCRWord] = field(default_factory=list)
    box: List[float] = field(default_factory=lambda: [0, 0, 0, 0])
    confidence: float = 0.0
    language: str = "en"
    reading_order: int = 0


def _parse_ocr_response(raw_result: str, img_width: int, img_height: int) -> List[OCRBlock]:
    """Parse the raw text output from the OCR space into structured blocks.

    The OCR space returns detected text.  We parse lines and synthesise
    bounding boxes spread evenly across the image when per-word coordinates
    are not directly available from the API.
    """
    if not raw_result or not raw_result.strip():
        return []

    lines = [l for l in raw_result.strip().splitlines() if l.strip()]
    blocks: List[OCRBlock] = []

    line_height = img_height / max(len(lines), 1)

    for idx, line in enumerate(lines):
        y1 = idx * line_height
        y2 = y1 + line_height
        x1 = 0.0
        x2 = float(img_width)

        words_in_line = line.split()
        word_width = (x2 - x1) / max(len(words_in_line), 1)

        ocr_words: List[OCRWord] = []
        for w_idx, word in enumerate(words_in_line):
            wx1 = x1 + w_idx * word_width
            wx2 = wx1 + word_width
            ocr_words.append(
                OCRWord(
                    text=word,
                    box=[wx1, y1, wx2, y2],
                    confidence=0.90,
                )
            )

        blocks.append(
            OCRBlock(
                text=line,
                words=ocr_words,
                box=[x1, y1, x2, y2],
                confidence=0.90,
                reading_order=idx,
            )
        )

    return blocks


def run_ocr(image_path: str, language_hint: Optional[str] = None) -> List[OCRBlock]:
    """Send an image to the HF OCR space and return structured blocks.

    Parameters
    ----------
    image_path : str
        Path to the image file on disk.
    language_hint : str | None
        Comma-separated language codes (unused by space but kept for API).

    Returns
    -------
    list[OCRBlock]
        Parsed OCR blocks with word-level data.

    Raises
    ------
    RuntimeError
        When the OCR service is completely unreachable (HTTP 503 equivalent).
    """
    from PIL import Image

    img = Image.open(image_path)
    img_width, img_height = img.size

    try:
        client = Client(HF_OCR_SPACE)
        result = client.predict(
            handle_file(image_path),
            api_name="/predict",
        )
    except Exception as exc:
        logger.error("OCR space call failed: %s", exc)
        raise RuntimeError(f"OCR service unavailable: {exc}") from exc

    raw_text = str(result) if result else ""
    blocks = _parse_ocr_response(raw_text, img_width, img_height)
    if not blocks:
        logger.warning("OCR returned no text for %s", image_path)
    return blocks
