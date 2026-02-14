"""Font identification using Hugging Face font-identifier model."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

HF_FONT_MODEL = "gaborcselle/font-identifier"

FONT_CATEGORIES = {
    "arial": "sans",
    "helvetica": "sans",
    "verdana": "sans",
    "tahoma": "sans",
    "calibri": "sans",
    "roboto": "sans",
    "open sans": "sans",
    "times": "serif",
    "times new roman": "serif",
    "georgia": "serif",
    "garamond": "serif",
    "palatino": "serif",
    "courier": "mono",
    "courier new": "mono",
    "consolas": "mono",
    "monaco": "mono",
    "comic sans": "display",
    "impact": "display",
    "papyrus": "handwritten",
}


@dataclass
class FontCandidate:
    name: str
    confidence: float


@dataclass
class FontResult:
    primary: str = "unknown"
    confidence: float = 0.0
    alternatives: List[FontCandidate] = field(default_factory=list)
    category: Optional[str] = None
    uncertain: bool = False


def _categorise(font_name: str) -> Optional[str]:
    lower = font_name.lower()
    for key, cat in FONT_CATEGORIES.items():
        if key in lower:
            return cat
    return None


def identify_font(image: Image.Image, box: List[float]) -> FontResult:
    """Crop the image to *box* and identify the font via the HF model.

    Parameters
    ----------
    image : PIL.Image.Image
        Full original image.
    box : list[float]
        [x1, y1, x2, y2] bounding box of the text region.

    Returns
    -------
    FontResult
        Identified font with confidence and alternatives.
    """
    x1, y1, x2, y2 = box
    crop = image.crop((int(x1), int(y1), int(x2), int(y2)))

    if crop.width < 2 or crop.height < 2:
        return FontResult(primary="unknown", confidence=0.0, uncertain=True)

    try:
        from gradio_client import Client, handle_file

        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        buf.seek(0)

        import tempfile, os

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.write(buf.getvalue())
        tmp.close()

        try:
            client = Client(HF_FONT_MODEL)
            result = client.predict(handle_file(tmp.name), api_name="/predict")
        finally:
            os.unlink(tmp.name)

        if isinstance(result, dict) and "label" in result:
            label = result["label"]
            conf = float(result.get("confidences", [{}])[0].get("confidence", 0.0))
            alternatives = []
            for alt in result.get("confidences", [])[1:4]:
                alternatives.append(
                    FontCandidate(
                        name=alt.get("label", "unknown"),
                        confidence=float(alt.get("confidence", 0.0)),
                    )
                )
            return FontResult(
                primary=label,
                confidence=conf,
                alternatives=alternatives,
                category=_categorise(label),
                uncertain=conf < 0.5,
            )

        if isinstance(result, str):
            return FontResult(
                primary=result.strip(),
                confidence=0.5,
                category=_categorise(result.strip()),
                uncertain=True,
            )

        return FontResult(primary="unknown", confidence=0.0, uncertain=True)

    except Exception as exc:
        logger.warning("Font identification failed: %s", exc)
        return FontResult(primary="unknown", confidence=0.0, uncertain=True)
