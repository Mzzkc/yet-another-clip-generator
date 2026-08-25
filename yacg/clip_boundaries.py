"""Shared source-time boundaries for extracted clips and their subtitles."""

from dataclasses import dataclass


_ASMR_MAX_CONTEXT_PADDING = 0.3


@dataclass(frozen=True)
class ClipBoundaries:
    """The effective source-time interval represented by an extracted clip."""

    start_time: float
    end_time: float


def effective_clip_boundaries(
    start_time: float,
    end_time: float,
    source_duration: float,
    *,
    context_padding: float,
    content_type: str,
) -> ClipBoundaries:
    """Return the padded interval after applying content-aware safety limits.

    ASMR keeps at most 0.3 seconds of context so pause-snapped boundaries are
    not obscured by extra spoken material.  Other content uses its configured
    padding.  Both bounds are clamped to the source interval here so every
    consumer shares the exact same clock origin.
    """
    padding = context_padding
    if "asmr" in (content_type or "").lower():
        padding = min(padding, _ASMR_MAX_CONTEXT_PADDING)

    duration = max(0.0, source_duration)
    return ClipBoundaries(
        start_time=min(max(0.0, start_time - padding), duration),
        end_time=min(max(0.0, end_time + padding), duration),
    )
