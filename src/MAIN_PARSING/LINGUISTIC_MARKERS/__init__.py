"""Reproducible linguistic-marker extraction and paired analysis."""

from .config import PIPELINE_VERSION
from .extractor import extract_markers
from .registry import MARKERS, MarkerSpec
from .surface import TextSurface, prepare_text_surface

__all__ = [
    "MARKERS",
    "PIPELINE_VERSION",
    "MarkerSpec",
    "TextSurface",
    "extract_markers",
    "prepare_text_surface",
]
