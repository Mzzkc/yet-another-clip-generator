"""
YACG — Yet Another Clip Generator. Extract viral-potential clips from long-form videos.

Detects scene boundaries, analyzes audio/visual/semantic signals, scores viral potential,
and extracts formatted clips with auto-generated Instagram captions.
"""

__version__ = "0.1.0"

from yacg.models import ClipData, PipelineConfig
from yacg.pipeline import ViralClipPipeline

__all__ = [
    "ClipData",
    "PipelineConfig",
    "ViralClipPipeline",
    "__version__",
]
