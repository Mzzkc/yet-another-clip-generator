"""
Viral Clip Extractor — automatically extract viral-potential clips from long-form videos.

Detects scene boundaries, analyzes audio/visual/semantic signals, scores viral potential,
and extracts formatted clips with auto-generated Instagram captions.
"""

__version__ = "0.1.0"

from viral_clip_extractor.models import ClipData, PipelineConfig
from viral_clip_extractor.pipeline import ViralClipPipeline

__all__ = [
    "ClipData",
    "PipelineConfig",
    "ViralClipPipeline",
    "__version__",
]
