"""
Clip extraction module.

Extracts scored video segments as standalone clip files using FFmpeg,
with vertical (9:16) reformatting for Instagram Reels / TikTok.
Supports single-clip and batch extraction with configurable context padding.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

from yacg.models import ClipData, PipelineConfig, SceneSegment
from yacg.utils.video_utils import translate_ffmpeg_error

logger = logging.getLogger(__name__)


class ClipExtractor:
    """Extract and format video clips for social media.

    Wraps FFmpeg to cut segments from a source video, re-encode to
    H.264/AAC, and apply a smart vertical crop via :class:`SmartCropper`.

    Args:
        context_padding: Seconds to add before/after the segment
            boundaries (clamped to video duration).
        config: Pipeline configuration (overrides *context_padding*
            if supplied).
    """

    # Maximum context padding (seconds) for ASMR/whispered-cadence content.
    # The pause-snap selected boundaries are what compose was meant to play;
    # extra padding adds un-snapped audio that lands mid-clause/mid-word and
    # defeats the snap.  ASMR speakers have continuous slow speech where 2s
    # of pre-roll catches several extra phrases.  Cap ensures the snap is
    # respected by default while still allowing some breathing room.
    _ASMR_MAX_PADDING = 0.3

    def __init__(
        self,
        context_padding: float = 2.0,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        raw_padding = (
            context_padding if config is None else config.context_padding
        )
        # Content-type-aware cap: ASMR caps at 0.3s regardless of configured
        # value.  This intentionally overrides higher user-specified values —
        # ASMR cadence makes >0.3s padding always wrong.  Set 0.0 explicitly
        # to disable padding entirely (e.g. for tier-0 isolation tests).
        content_type = (
            self.config.content_profile.content_type.lower()
            if self.config.content_profile and self.config.content_profile.content_type
            else ""
        )
        if "asmr" in content_type and raw_padding > self._ASMR_MAX_PADDING:
            logger.info(
                "Capping context_padding %.2fs → %.2fs for ASMR content "
                "(set explicitly to override)",
                raw_padding, self._ASMR_MAX_PADDING,
            )
            self.context_padding = self._ASMR_MAX_PADDING
        else:
            self.context_padding = raw_padding
        self._cropper = None  # cached SmartCropper instance

    def extract_clip(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        output_path: str,
    ) -> bool:
        """Extract a single clip with context padding and vertical crop.

        Pads segment boundaries by :attr:`context_padding` seconds (clamped
        to ``[0, video_duration]``), re-encodes with libx264/CRF 23/AAC 128k,
        applies the SmartCropper filter for 9:16 vertical cropping, and
        validates that the output exists and is at least 10 KB.

        Args:
            video_path: Source video file path.
            start_time: Segment start in seconds.
            end_time: Segment end in seconds.
            output_path: Destination file path.

        Returns:
            ``True`` if the clip was created and validated, ``False`` otherwise.
        """
        # Determine video duration for clamping
        video_duration = self._get_video_duration(video_path)

        # Apply context padding, clamped to video bounds
        padded_start = max(0.0, start_time - self.context_padding)
        padded_end = min(video_duration, end_time + self.context_padding)
        duration = padded_end - padded_start

        if duration <= 0:
            logger.error(
                "Invalid clip duration %.2f (start=%.2f, end=%.2f, padding=%.1f)",
                duration, start_time, end_time, self.context_padding,
            )
            return False

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Build FFmpeg command
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(padded_start),
            "-i", str(video_path),
            "-t", str(duration),
        ]

        # Apply vertical crop filter (_get_vertical_filter raises on failure)
        vf_filter = self._get_vertical_filter(video_path, start_time, end_time)
        cmd.extend(["-vf", vf_filter])

        cmd.extend([
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(output_path),
        ])

        # Execute FFmpeg with single retry on failure
        max_ffmpeg_attempts = 2
        for ffmpeg_attempt in range(1, max_ffmpeg_attempts + 1):
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300,
                )
            except FileNotFoundError:
                logger.error("FFmpeg not found — install FFmpeg to extract clips")
                return False
            except subprocess.TimeoutExpired:
                logger.error("FFmpeg timed out while extracting clip")
                if ffmpeg_attempt < max_ffmpeg_attempts:
                    logger.info("Retrying FFmpeg extraction (attempt %d/%d)...", ffmpeg_attempt + 1, max_ffmpeg_attempts)
                    import time
                    time.sleep(2)
                    continue
                return False

            if result.returncode == 0:
                break

            human_msg = translate_ffmpeg_error(result.stderr)
            if ffmpeg_attempt < max_ffmpeg_attempts:
                logger.warning(
                    "FFmpeg attempt %d failed: %s — retrying in 2s",
                    ffmpeg_attempt, human_msg,
                )
                import time
                time.sleep(2)
            else:
                logger.error("Clip extraction failed: %s", human_msg)
                return False

        # Validate output
        out = Path(output_path)
        if not out.exists():
            logger.error("Output file was not created: %s", output_path)
            return False

        file_size = out.stat().st_size
        if file_size < 10240:  # 10 KB minimum
            logger.error(
                "Output file too small (%d bytes), likely corrupt: %s",
                file_size, output_path,
            )
            out.unlink(missing_ok=True)
            return False

        logger.info(
            "Extracted clip %.1f–%.1fs → %s (%d KB)",
            padded_start, padded_end, output_path, file_size // 1024,
        )
        return True

    def batch_extract(
        self,
        video_path: str,
        segments: list[SceneSegment],
        output_dir: str,
        scores: Optional[list[float]] = None,
    ) -> list[str]:
        """Extract multiple clips from scored segments.

        Files are named ``clip_NN_scoreXX.mp4`` where NN is the 1-based
        index and XX is the integer virality score.

        Args:
            video_path: Source video file path.
            segments: Scene segments to extract.
            output_dir: Directory for output clips.
            scores: Optional virality scores (parallel to *segments*).

        Returns:
            List of successfully created output file paths.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        extracted: list[str] = []

        for i, seg in enumerate(segments):
            score_val = int(scores[i]) if scores and i < len(scores) else 0
            filename = f"clip_{i + 1:02d}_score{score_val}.mp4"
            output_path = os.path.join(output_dir, filename)

            success = self.extract_clip(
                video_path, seg.start_time, seg.end_time, output_path,
            )
            if success:
                extracted.append(output_path)
            else:
                logger.warning(
                    "Failed to extract clip %d (%.1f–%.1fs)",
                    i + 1, seg.start_time, seg.end_time,
                )

        logger.info(
            "Batch extracted %d/%d clips to %s",
            len(extracted), len(segments), output_dir,
        )
        return extracted

    # Backward-compatible method matching the original stub signature
    def extract_batch(self, video_path: str, clips: list[ClipData]) -> list[str]:
        """Extract multiple clips from a list of :class:`ClipData`.

        Args:
            video_path: Source video file path.
            clips: List of ClipData with scene boundaries and scores.

        Returns:
            List of output file paths.
        """
        segments = [c.scene for c in clips]
        scores = [c.virality.total_score for c in clips]
        output_dir = self.config.output_dir
        return self.batch_extract(video_path, segments, output_dir, scores)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_video_duration(self, video_path: str) -> float:
        """Query video duration via ffprobe, falling back to a large but finite value.

        Returns 86400.0 (24 hours) as a safe upper bound when duration cannot be
        determined, rather than ``float("inf")`` which could produce absurdly
        long clips from corrupt metadata.
        """
        _FALLBACK_DURATION = 86400.0  # 24 hours
        try:
            from yacg.utils.video_utils import extract_metadata
            meta = extract_metadata(video_path)
            duration = float(meta.get("duration", 0))
            if duration > 0:
                return duration
            logger.warning(
                "Video duration is zero or missing for %s — using fallback %.0fs",
                video_path, _FALLBACK_DURATION,
            )
            return _FALLBACK_DURATION
        except Exception as exc:
            logger.warning(
                "Could not determine video duration for %s: %s — using fallback %.0fs",
                video_path, exc, _FALLBACK_DURATION,
            )
            return _FALLBACK_DURATION

    def _get_vertical_filter(
        self, video_path: str, start_time: float = 0, end_time: float | None = None,
    ) -> str:
        """Build a SmartCropper filter. Raises on failure."""
        try:
            if self._cropper is None:
                from yacg.extractors.smart_cropper import SmartCropper
                self._cropper = SmartCropper(config=self.config)
            vf = self._cropper.get_ffmpeg_filter(
                video_path, start_time=start_time, end_time=end_time,
            )
            if not vf:
                raise RuntimeError(
                    f"SmartCropper returned empty filter for {video_path}"
                )
            return vf
        except ImportError as exc:
            raise RuntimeError(
                "SmartCropper unavailable — cv2 is required for vertical cropping"
            ) from exc
