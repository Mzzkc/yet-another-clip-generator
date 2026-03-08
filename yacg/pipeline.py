"""
Main pipeline orchestrator for Yet Another Clip Generator (YACG).

Coordinates transcript-first segmentation, multi-modal analysis, virality
scoring, clip extraction, subtitle burning, and caption generation into an
end-to-end workflow. Supports both local video files and YouTube URLs.
"""

import csv
import logging
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from yacg.models import (
    AudioFeatures,
    CaptionData,
    ClipData,
    PipelineConfig,
    ProcessingResult,
    SceneSegment,
    SegmentBoundary,
    SemanticFeatures,
    SubtitleStyle,
    ViralityScore,
    VisualFeatures,
    WordTimestamp,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Component Protocols — interfaces for dependency injection
# ------------------------------------------------------------------

@runtime_checkable
class TranscriptSegmenterProtocol(Protocol):
    """Interface for transcript segmentation."""

    def full_transcribe(self, video_path: str) -> list[WordTimestamp]: ...
    def segment_by_content(
        self, words: list[WordTimestamp], title: str,
        target_count: int = 20,
    ) -> list[SegmentBoundary]: ...
    def refine_boundaries(
        self, segments: list[SegmentBoundary], words: list[WordTimestamp]
    ) -> list[SceneSegment]: ...


@runtime_checkable
class SubtitleBurnerProtocol(Protocol):
    """Interface for subtitle burning."""

    def get_video_dimensions(self, video_path: str) -> tuple[int, int]: ...
    def process_clip(
        self, video_path: str, words: list[WordTimestamp],
        width: int, height: int,
        style: Optional[SubtitleStyle] = None,
    ) -> str: ...


@runtime_checkable
class SemanticAnalyzerProtocol(Protocol):
    """Interface for semantic analysis."""

    def analyze_segment(
        self, video_path: str, start: float, end: float,
        title: str = "",
    ) -> SemanticFeatures: ...


@runtime_checkable
class CaptionGeneratorProtocol(Protocol):
    """Interface for caption generation."""

    def analyze_video(
        self, video_path: str, title: str, transcript_text: str = "",
    ) -> CaptionData: ...


@runtime_checkable
class AudioAnalyzerProtocol(Protocol):
    """Interface for audio analysis."""

    def analyze_segment(
        self, video_path: str, start_time: float, end_time: float,
        words: Optional[list[WordTimestamp]] = None,
    ) -> AudioFeatures: ...


@runtime_checkable
class VisualAnalyzerProtocol(Protocol):
    """Interface for visual analysis."""

    def analyze_segment(
        self, video_path: str, start_time: float, end_time: float,
    ) -> VisualFeatures: ...


@runtime_checkable
class ViralityScorerProtocol(Protocol):
    """Interface for virality scoring."""

    def calculate_score(
        self, audio: AudioFeatures, visual: VisualFeatures,
        semantic: Optional[SemanticFeatures], duration: float,
    ) -> ViralityScore: ...


@runtime_checkable
class ClipExtractorProtocol(Protocol):
    """Interface for clip extraction."""

    def extract_clip(
        self, video_path: str, start_time: float, end_time: float,
        output_path: str,
    ) -> bool: ...


class ViralClipPipeline:
    """End-to-end pipeline for extracting viral clips from video.

    Orchestrates transcript-first segmentation, audio/visual/semantic
    analysis, virality scoring, clip extraction, subtitle burning, and
    caption generation.

    Args:
        config: Pipeline configuration. Uses defaults if not provided.
        transcript_segmenter: Optional custom segmenter implementation.
        subtitle_burner: Optional custom subtitle burner implementation.
        semantic_analyzer: Optional custom semantic analyzer implementation.
        caption_generator: Optional custom caption generator implementation.
        audio_analyzer: Optional custom audio analyzer implementation.
        visual_analyzer: Optional custom visual analyzer implementation.
        virality_scorer: Optional custom virality scorer implementation.
        clip_extractor: Optional custom clip extractor implementation.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        transcript_segmenter: Optional[TranscriptSegmenterProtocol] = None,
        subtitle_burner: Optional[SubtitleBurnerProtocol] = None,
        semantic_analyzer: Optional[SemanticAnalyzerProtocol] = None,
        caption_generator: Optional[CaptionGeneratorProtocol] = None,
        audio_analyzer: Optional[AudioAnalyzerProtocol] = None,
        visual_analyzer: Optional[VisualAnalyzerProtocol] = None,
        virality_scorer: Optional[ViralityScorerProtocol] = None,
        clip_extractor: Optional[ClipExtractorProtocol] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        # Optional callback: (step_name, current, total) for GUI/web embedding
        self._progress_callback = progress_callback

        # Lazy-initialized components (injectable via constructor)
        self._audio_analyzer = audio_analyzer
        self._visual_analyzer = visual_analyzer
        self._semantic_analyzer = semantic_analyzer
        self._virality_scorer = virality_scorer
        self._clip_extractor = clip_extractor
        self._caption_analyzer = caption_generator
        self._transcript_segmenter = transcript_segmenter
        self._subtitle_burner = subtitle_burner

    # ------------------------------------------------------------------
    # Component accessors (lazy init)
    # ------------------------------------------------------------------

    def _get_audio_analyzer(self):
        if self._audio_analyzer is None:
            from yacg.core.audio_analyzer import AudioAnalyzer
            # Pass content-type-aware keywords: ASMR keywords for ASMR
            # content, general engagement keywords for other content types.
            if self.config.content_profile.content_type == "asmr":
                self._audio_analyzer = AudioAnalyzer()
            else:
                general_keywords = [
                    "amazing", "incredible", "wow", "unbelievable",
                    "shocking", "secret", "hack", "trick", "tip",
                ]
                self._audio_analyzer = AudioAnalyzer(asmr_keywords=general_keywords)
        return self._audio_analyzer

    def _get_visual_analyzer(self):
        if self._visual_analyzer is None:
            from yacg.core.visual_analyzer import VisualAnalyzer
            self._visual_analyzer = VisualAnalyzer(config=self.config)
        return self._visual_analyzer

    def _get_semantic_analyzer(self):
        if self._semantic_analyzer is None:
            from yacg.core.semantic_analyzer import SemanticAnalyzer
            cp = self.config.content_profile
            self._semantic_analyzer = SemanticAnalyzer(
                model=self.config.model_name,
                ollama_host=self.config.ollama_host,
                content_type=cp.content_type,
                num_frames=self.config.num_frames,
                channel_description=cp.channel_description,
                target_audience=cp.target_audience,
                tone=cp.tone,
                custom_instructions=cp.custom_instructions,
            )
        return self._semantic_analyzer

    def _get_virality_scorer(self):
        if self._virality_scorer is None:
            from yacg.core.virality_scorer import ViralityScorer
            self._virality_scorer = ViralityScorer(config=self.config)
        return self._virality_scorer

    def _get_clip_extractor(self):
        if self._clip_extractor is None:
            from yacg.extractors.clip_extractor import ClipExtractor
            self._clip_extractor = ClipExtractor(config=self.config)
        return self._clip_extractor

    def _get_caption_analyzer(self):
        """Get caption analyzer (OllamaVideoAnalyzer)."""
        if self._caption_analyzer is None:
            from yacg.caption_generator import OllamaVideoAnalyzer
            cp = self.config.content_profile
            self._caption_analyzer = OllamaVideoAnalyzer(
                model=self.config.model_name,
                ollama_host=self.config.ollama_host,
                content_type=cp.content_type,
                channel_description=cp.channel_description,
                target_audience=cp.target_audience,
                tone=cp.tone,
                platform=cp.platform,
                caption_length=cp.caption_length,
                hashtag_count=cp.hashtag_count,
                custom_instructions=cp.custom_instructions,
            )
        return self._caption_analyzer

    def _get_transcript_segmenter(self):
        """Get transcript segmenter (TranscriptSegmenter)."""
        if self._transcript_segmenter is None:
            from yacg.transcript_segmenter import TranscriptSegmenter
            seg_model = self.config.segmentation_model or self.config.model_name
            cp = self.config.content_profile
            self._transcript_segmenter = TranscriptSegmenter(
                whisper_model=self.config.whisper_model,
                ollama_host=self.config.ollama_host,
                model_name=seg_model,
                whisper_device=self.config.whisper_device,
                whisper_compute_type=self.config.whisper_compute_type,
                pause_threshold=self.config.pause_threshold,
                min_segment_duration=self.config.min_segment_duration,
                max_segment_duration=self.config.max_segment_duration,
                vad_filter=self.config.vad_filter,
                content_type=cp.content_type,
                channel_description=cp.channel_description,
                target_audience=cp.target_audience,
                custom_instructions=cp.custom_instructions,
            )
        return self._transcript_segmenter

    def _get_subtitle_burner(self):
        """Get subtitle burner (SubtitleBurner)."""
        if self._subtitle_burner is None:
            from yacg.subtitle_burner import SubtitleBurner
            self._subtitle_burner = SubtitleBurner()
        return self._subtitle_burner

    def _emit_progress(self, step_name: str, current: int, total: int) -> None:
        """Notify the progress callback, if one was registered."""
        if self._progress_callback is not None:
            try:
                self._progress_callback(step_name, current, total)
            except Exception:
                logger.debug("Progress callback error (ignored)", exc_info=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path: str,
        title: str = "",
        top_n: int = 20,
        min_score: float = 70.0,
    ) -> ProcessingResult:
        """Process a local video file through the full pipeline.

        Orchestrates seven steps: transcribe, segment, analyze, rank,
        extract, burn subtitles, generate captions.

        Args:
            video_path: Path to the video file.
            title: Video title for context in semantic analysis.
            top_n: Maximum number of clips to extract.
            min_score: Minimum virality score threshold.

        Returns:
            A ProcessingResult with extracted clips and metadata.
        """
        start_time_wall = time.time()
        errors: list[str] = []

        # Validate video
        if not os.path.exists(video_path):
            return self._error_result(video_path, title, 0.0, [f"Video not found: {video_path}"])
        if not os.path.isfile(video_path):
            return self._error_result(video_path, title, 0.0, [f"Not a file: {video_path}"])

        logger.info("Processing video: %s", video_path)

        # Disk space pre-check: warn if < 1 GB free in output directory
        try:
            output_parent = Path(self.config.output_dir)
            output_parent.mkdir(parents=True, exist_ok=True)
            disk_usage = shutil.disk_usage(str(output_parent))
            free_gb = disk_usage.free / (1024 ** 3)
            if free_gb < 1.0:
                logger.warning(
                    "Low disk space: %.1f GB free in %s — pipeline may fail "
                    "during extraction", free_gb, self.config.output_dir,
                )
        except OSError:
            pass  # non-critical — proceed anyway

        # Ensure codec compatibility — track original path for cleanup
        original_video_path = video_path
        try:
            from yacg.utils.video_utils import ensure_compatible_video
            video_path = ensure_compatible_video(video_path)
        except Exception as exc:
            error_msg = f"Codec compatibility check failed: {exc}"
            logger.error(error_msg)
            errors.append(error_msg)
        transcoded_file = video_path if video_path != original_video_path else None

        # Extract title from metadata if not provided
        if not title:
            try:
                from yacg.utils.video_utils import extract_metadata
                meta = extract_metadata(video_path)
                title = meta.get("filename", Path(video_path).stem)
            except Exception as exc:
                logger.warning("Could not extract metadata: %s", exc)
                title = Path(video_path).stem

        # Steps 1-2: Transcribe and segment
        step_start = time.time()
        ts_result = self._transcribe_and_segment(
            video_path, title, start_time_wall, errors, target_count=top_n,
        )
        if isinstance(ts_result, ProcessingResult):
            return ts_result
        all_words, segments = ts_result
        total_segments = len(segments)
        logger.info("Steps 1-2 completed in %.1fs", time.time() - step_start)

        # Step 3: Analyze segments
        step_start = time.time()
        clip_data_list = self._analyze_segments(video_path, segments, title, all_words, errors)
        logger.info("Step 3 completed in %.1fs", time.time() - step_start)

        # Step 4: Rank and select
        selected = self._rank_and_select(clip_data_list, min_score, top_n)

        # Dry-run: log ranked clip list and stop
        if self.config.dry_run:
            processing_time = time.time() - start_time_wall
            logger.info("=" * 60)
            logger.info("DRY RUN — %d clips would be extracted:", len(selected))
            logger.info("=" * 60)
            for i, clip in enumerate(selected):
                logger.info(
                    "  %2d. [%7.2fs – %7.2fs] score=%.1f  hook=%.1f  emotion=%.1f",
                    i + 1, clip.scene.start_time, clip.scene.end_time,
                    clip.virality.total_score, clip.semantic.hook_potential,
                    clip.semantic.emotional_intensity,
                )
            logger.info("Analysis completed in %.1fs", processing_time)
            return ProcessingResult(
                video_path=video_path, video_title=title,
                clips=selected, total_scenes=total_segments,
                processing_time_seconds=round(processing_time, 2),
                errors=errors,
            )

        # Steps 5-7: Extract, subtitle, caption (in staging directory)
        output_dir = self.config.output_dir
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            error_msg = f"Cannot create output directory '{output_dir}': {exc}"
            logger.error(error_msg)
            return self._error_result(
                video_path, title, time.time() - start_time_wall,
                errors + [error_msg], total_segments,
            )

        staging_dir = os.path.join(output_dir, ".yacg_tmp")
        try:
            Path(staging_dir).mkdir(parents=True, exist_ok=True)
            step_start = time.time()
            self._extract_clips(video_path, selected, staging_dir, title, errors)
            logger.info("Step 5 completed in %.1fs", time.time() - step_start)
            step_start = time.time()
            self._burn_subtitles(selected, all_words, errors)
            logger.info("Step 6 completed in %.1fs", time.time() - step_start)
            self._generate_thumbnails(selected, errors)
            step_start = time.time()
            self._generate_captions(selected, title, errors)
            logger.info("Step 7 completed in %.1fs", time.time() - step_start)

            # Move completed clips and thumbnails from staging to final output
            for clip in selected:
                if clip.output_path and os.path.exists(clip.output_path):
                    # Move thumbnail first (before clip.output_path changes)
                    staging_thumb = os.path.splitext(clip.output_path)[0] + "_thumb.jpg"
                    if os.path.exists(staging_thumb):
                        final_thumb = os.path.join(output_dir, os.path.basename(staging_thumb))
                        shutil.move(staging_thumb, final_thumb)
                        clip.thumbnail_path = final_thumb
                    final_path = os.path.join(output_dir, os.path.basename(clip.output_path))
                    shutil.move(clip.output_path, final_path)
                    clip.output_path = final_path
        except Exception as exc:
            error_msg = f"Pipeline extraction stage failed: {exc}"
            logger.error(error_msg)
            errors.append(error_msg)
        finally:
            if os.path.isdir(staging_dir):
                shutil.rmtree(staging_dir, ignore_errors=True)
            # Clean up transcoded file created by ensure_compatible_video()
            if transcoded_file and os.path.exists(transcoded_file):
                try:
                    os.unlink(transcoded_file)
                    logger.debug("Cleaned up transcoded file: %s", transcoded_file)
                except OSError as exc:
                    logger.warning("Failed to clean up transcoded file %s: %s", transcoded_file, exc)

        return self._write_report(
            video_path, title, selected, total_segments, start_time_wall, errors,
        )

    # ------------------------------------------------------------------
    # Pipeline step methods (decomposed from process_video)
    # ------------------------------------------------------------------

    def _error_result(
        self, video_path: str, title: str, elapsed: float,
        errors: list[str], total_scenes: int = 0,
    ) -> ProcessingResult:
        """Build an error ProcessingResult."""
        return ProcessingResult(
            video_path=video_path, video_title=title,
            clips=[], total_scenes=total_scenes,
            processing_time_seconds=elapsed, errors=errors,
        )

    def _transcribe_and_segment(
        self, video_path: str, title: str,
        start_time_wall: float, errors: list[str],
        target_count: int = 20,
    ) -> "ProcessingResult | tuple[list[WordTimestamp], list[SceneSegment]]":
        """Steps 1-2: Transcribe video and identify segments."""
        # Log video duration so users know what to expect
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "format=duration", "-of", "csv=p=0", video_path],
                capture_output=True, text=True, timeout=10,
            )
            if probe.returncode == 0 and probe.stdout.strip():
                dur = float(probe.stdout.strip())
                mins, secs = int(dur // 60), int(dur % 60)
                if dur > 3600:
                    logger.warning(
                        "Video is %d:%02d long — transcription and analysis "
                        "will take a very long time. Consider trimming the "
                        "input or using a shorter segment.",
                        mins, secs,
                    )
                logger.info(
                    "Step 1/7: Transcribing video (%d:%02d long)...", mins, secs,
                )
            else:
                logger.info("Step 1/7: Transcribing video...")
        except Exception:
            logger.info("Step 1/7: Transcribing video...")

        self._emit_progress("transcribe", 1, 7)
        try:
            segmenter = self._get_transcript_segmenter()
            all_words = segmenter.full_transcribe(video_path)
        except Exception as exc:
            return self._error_result(
                video_path, title, time.time() - start_time_wall,
                [f"Transcription failed: {exc}"],
            )

        logger.info("Step 2/7: Identifying viral segments...")
        self._emit_progress("segment", 2, 7)
        try:
            boundaries = segmenter.segment_by_content(
                all_words, title, target_count=target_count,
            )
        except Exception as exc:
            return self._error_result(
                video_path, title, time.time() - start_time_wall,
                [f"LLM segmentation failed: {exc}"],
            )

        segments = segmenter.refine_boundaries(boundaries, all_words)
        logger.info("Refined to %d segments", len(segments))

        if not segments:
            return self._error_result(
                video_path, title, time.time() - start_time_wall,
                ["No viable segments after refinement"],
            )
        return all_words, segments

    def _analyze_segments(
        self, video_path: str, segments: list[SceneSegment],
        title: str, all_words: list[WordTimestamp], errors: list[str],
    ) -> list[ClipData]:
        """Step 3: Analyze and score each segment."""
        logger.info("Step 3/7: Analyzing %d segments...", len(segments))
        self._emit_progress("analyze", 3, 7)
        clip_data_list: list[ClipData] = []
        total = len(segments)
        for i, scene in enumerate(segments):
            logger.info("Processing segment %d/%d (%.0f%%)", i + 1, total, ((i + 1) / total) * 100)
            self._emit_progress("analyze_segment", i + 1, total)
            try:
                clip_data = self._analyze_segment(
                    video_path, scene, title, all_words=all_words,
                )
                clip_data_list.append(clip_data)
            except Exception as exc:
                error_msg = f"Segment {i} analysis failed: {exc}"
                logger.error(error_msg)
                errors.append(error_msg)
        return clip_data_list

    # TODO: Add manual clip boundary adjustment step — allow users to review
    # LLM-suggested segments and adjust start/end times before extraction.
    # This would require a preview/review UI or CLI interactive mode.

    def _rank_and_select(
        self, clip_data_list: list[ClipData], min_score: float, top_n: int,
    ) -> list[ClipData]:
        """Step 4: Rank by score, filter, and select top N."""
        logger.info("Step 4/7: Ranking and filtering clips...")
        clip_data_list.sort(key=lambda c: c.virality.total_score, reverse=True)
        filtered = [c for c in clip_data_list if c.virality.total_score >= min_score]
        selected = filtered[:top_n]
        logger.info(
            "Selected %d clips (from %d scored, %d above threshold %.1f)",
            len(selected), len(clip_data_list), len(filtered), min_score,
        )
        if not filtered and clip_data_list:
            top_score = clip_data_list[0].virality.total_score
            logger.warning(
                "No clips passed --min-score %.1f (highest score: %.1f). "
                "VLM scores are not calibrated to an absolute scale — "
                "try lowering --min-score or use --top-n without a threshold.",
                min_score, top_score,
            )
        return selected

    def _extract_clips(
        self, video_path: str, selected: list[ClipData],
        staging_dir: str, title: str, errors: list[str],
    ) -> None:
        """Step 5: Extract clip files to staging directory (parallel)."""
        logger.info("Step 5/7: Extracting %d clip files...", len(selected))
        self._emit_progress("extract", 5, 7)
        extractor = self._get_clip_extractor()
        title_slug = self._slugify(title) if title else ""

        extraction_tasks: list[tuple[int, ClipData, str]] = []
        for i, clip in enumerate(selected):
            score_int = int(clip.virality.total_score)
            if title_slug:
                filename = f"{title_slug}_clip_{i + 1:02d}_score{score_int}.mp4"
            else:
                filename = f"clip_{i + 1:02d}_score{score_int}.mp4"
            staging_path = os.path.join(staging_dir, filename)
            extraction_tasks.append((i, clip, staging_path))

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _extract_one(task: tuple[int, ClipData, str]) -> tuple[int, bool, str]:
            idx, clip_data, out_path = task
            try:
                ok = extractor.extract_clip(
                    video_path, clip_data.scene.start_time,
                    clip_data.scene.end_time, out_path,
                )
                return (idx, ok, out_path)
            except Exception as exc:
                logger.error("Clip %d extraction failed: %s", idx + 1, exc)
                return (idx, False, out_path)

        max_workers = min(4, len(extraction_tasks)) or 1
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_extract_one, t): t for t in extraction_tasks}
            for future in as_completed(futures, timeout=600):
                idx, ok, out_path = future.result()
                if ok:
                    selected[idx].output_path = out_path
                else:
                    errors.append(f"Failed to extract clip {idx + 1}")

    def _burn_subtitles(
        self, selected: list[ClipData],
        all_words: list[WordTimestamp], errors: list[str],
    ) -> None:
        """Step 6: Burn subtitles into clips (parallel). Mandatory — failure deletes clip."""
        logger.info("Step 6/7: Burning subtitles...")
        self._emit_progress("subtitles", 6, 7)
        burner = self._get_subtitle_burner()

        eligible = [
            (i, clip) for i, clip in enumerate(selected)
            if clip.output_path and os.path.exists(clip.output_path)
        ]
        if not eligible:
            return

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _burn_one(
            task: tuple[int, ClipData],
        ) -> tuple[int, Optional[str], Optional[list[WordTimestamp]]]:
            idx, clip = task
            try:
                padded_start = max(0.0, clip.scene.start_time - self.config.context_padding)
                padded_end = clip.scene.end_time + self.config.context_padding
                if all_words:
                    padded_end = min(padded_end, all_words[-1].end)

                _BOUNDARY_TOLERANCE = 0.05
                clip_words = [
                    WordTimestamp(
                        word=w.word,
                        start=max(0.0, w.start - padded_start),
                        end=w.end - padded_start,
                        probability=w.probability,
                    )
                    for w in all_words
                    if (w.start >= padded_start - _BOUNDARY_TOLERANCE
                        and w.end <= padded_end + _BOUNDARY_TOLERANCE)
                ]

                if not clip_words:
                    raise RuntimeError(
                        f"No speech in clip {idx + 1} time range — "
                        f"cannot burn subtitles"
                    )

                width, height = burner.get_video_dimensions(clip.output_path)
                burner.process_clip(
                    clip.output_path, clip_words, width, height,
                    style=self.config.subtitle_style,
                )
                return (idx, None, clip_words)
            except Exception as exc:
                return (idx, f"Subtitle burning failed for clip {idx + 1}: {exc}", None)

        max_workers = min(4, len(eligible)) or 1
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_burn_one, t): t for t in eligible}
            for future in as_completed(futures, timeout=600):
                idx, error_msg, clip_words = future.result()
                if error_msg:
                    clip = selected[idx]
                    if clip.output_path and os.path.exists(clip.output_path):
                        os.unlink(clip.output_path)
                        clip.output_path = None
                    logger.error(error_msg)
                    errors.append(error_msg)
                else:
                    selected[idx].words = clip_words

    def _generate_thumbnails(self, selected: list[ClipData], errors: list[str]) -> None:
        """Generate thumbnails for all extracted clips. Failure is an error."""
        for clip in selected:
            if clip.output_path and os.path.exists(clip.output_path):
                thumb = self._generate_thumbnail(clip.output_path)
                if thumb:
                    clip.thumbnail_path = thumb
                    logger.debug("Thumbnail: %s", thumb)
                else:
                    error_msg = (
                        f"Thumbnail generation failed for "
                        f"{os.path.basename(clip.output_path)}"
                    )
                    logger.error(error_msg)
                    errors.append(error_msg)

    def _generate_captions(
        self, selected: list[ClipData], title: str, errors: list[str],
    ) -> None:
        """Step 7: Generate captions for each clip."""
        logger.info("Step 7/7: Generating captions...")
        self._emit_progress("captions", 7, 7)
        caption_analyzer = self._get_caption_analyzer()
        caption_total = sum(1 for c in selected if c.output_path and os.path.exists(c.output_path))
        for i, clip in enumerate(selected):
            if clip.output_path and os.path.exists(clip.output_path):
                logger.info("Generating caption for clip %d/%d...", i + 1, caption_total)
                try:
                    # Build transcript text from clip.words (populated in Step 6)
                    transcript_text = " ".join(w.word for w in clip.words) if clip.words else ""
                    clip.caption = caption_analyzer.analyze_video(
                        clip.output_path, title, transcript_text=transcript_text,
                    )
                except Exception as exc:
                    error_msg = f"Caption generation failed for clip {i + 1}: {exc}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    # Caption is mandatory — delete clip and thumbnail on
                    # failure, consistent with subtitle failure behavior.
                    if clip.output_path and os.path.exists(clip.output_path):
                        os.unlink(clip.output_path)
                        clip.output_path = None
                    if clip.thumbnail_path and os.path.exists(clip.thumbnail_path):
                        os.unlink(clip.thumbnail_path)
                        clip.thumbnail_path = None

    def _write_report(
        self, video_path: str, title: str, selected: list[ClipData],
        total_segments: int, start_time_wall: float, errors: list[str],
    ) -> ProcessingResult:
        """Build final result and write CSV report."""
        output_dir = self.config.output_dir
        processing_time = time.time() - start_time_wall
        result = ProcessingResult(
            video_path=video_path, video_title=title,
            clips=selected, total_scenes=total_segments,
            processing_time_seconds=round(processing_time, 2),
            errors=errors,
        )

        csv_path = os.path.join(output_dir, "clips_report.csv")
        self._generate_csv(result, csv_path)

        extracted_count = sum(1 for c in selected if c.output_path)
        logger.info(
            "Pipeline complete: %d clips extracted in %.1fs (%d errors)",
            extracted_count, processing_time, len(errors),
        )
        if extracted_count:
            logger.info("=" * 60)
            logger.info("Extracted clips:")
            logger.info("=" * 60)
            for i, clip in enumerate(selected):
                if clip.output_path:
                    hook_text = ""
                    if clip.caption and isinstance(clip.caption, CaptionData):
                        hook_text = f'  hook="{clip.caption.hook[:50]}"'
                    logger.info(
                        "  %2d. %-40s  score=%.1f  [%.1fs-%.1fs]%s",
                        i + 1,
                        os.path.basename(clip.output_path),
                        clip.virality.total_score,
                        clip.scene.start_time,
                        clip.scene.end_time,
                        hook_text,
                    )
            logger.info("=" * 60)
        logger.info("Output: %s", output_dir)
        logger.info("CSV report: %s", csv_path)

        return result

    def process_youtube(
        self,
        url: str,
        top_n: int = 20,
        min_score: float = 70.0,
    ) -> ProcessingResult:
        """Process a YouTube video through the full pipeline.

        Downloads the video, extracts title from metadata, processes it,
        and optionally cleans up the downloaded file.

        Args:
            url: YouTube video URL.
            top_n: Maximum number of clips to extract.
            min_score: Minimum virality score threshold.

        Returns:
            A ProcessingResult with extracted clips and metadata.
        """
        from yacg.youtube_downloader import YouTubeDownloader

        download_dir = os.path.join(self.config.output_dir, "downloads")
        downloader = YouTubeDownloader(output_dir=download_dir)

        logger.info("Downloading YouTube video: %s", url)

        try:
            dl_result = downloader.download(url)
        except Exception as exc:
            error_msg = f"YouTube download failed: {exc}"
            logger.error(error_msg)
            return ProcessingResult(
                video_path=url,
                video_title="",
                clips=[],
                total_scenes=0,
                processing_time_seconds=0.0,
                errors=[error_msg],
            )

        video_path = dl_result["video_path"]
        title = dl_result.get("title", "")
        logger.info("Downloaded: %s -> %s", title, video_path)

        try:
            result = self.process_video(
                video_path=video_path,
                title=title,
                top_n=top_n,
                min_score=min_score,
            )
        except Exception:
            # Clean up downloaded file on pipeline failure
            if os.path.exists(video_path):
                try:
                    os.unlink(video_path)
                    logger.info("Cleaned up downloaded file: %s", video_path)
                except OSError as cleanup_exc:
                    logger.warning("Failed to clean up download %s: %s", video_path, cleanup_exc)
            raise

        return result

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _analyze_segment(
        self,
        video_path: str,
        scene: SceneSegment,
        title: str,
        all_words: Optional[list[WordTimestamp]] = None,
    ) -> ClipData:
        """Analyze a single scene segment for virality signals.

        Runs audio, visual, and semantic analysis, then computes a
        composite virality score. All analysis is mandatory — failures
        propagate as exceptions.

        Args:
            video_path: Path to the source video.
            scene: The scene boundaries.
            title: Video title for semantic context.
            all_words: Full-video word timestamps for trigger word detection
                (avoids redundant Whisper re-transcription in audio analyzer).

        Returns:
            A ClipData instance with all analysis results.
        """
        start = scene.start_time
        end = scene.end_time

        # Audio analysis — passes existing words to avoid redundant Whisper run
        audio = self._get_audio_analyzer().analyze_segment(
            video_path, start, end, words=all_words,
        )

        # Visual analysis — raises on failure
        visual = self._get_visual_analyzer().analyze_segment(video_path, start, end)

        # Semantic analysis — always runs, raises on failure
        semantic = self._get_semantic_analyzer().analyze_segment(
            video_path, start, end, title=title,
        )

        # Score virality
        virality = self._get_virality_scorer().calculate_score(
            audio=audio,
            visual=visual,
            semantic=semantic,
            duration=scene.duration,
        )

        return ClipData(
            scene=scene,
            audio=audio,
            visual=visual,
            semantic=semantic,
            virality=virality,
        )

    def _generate_csv(self, result: ProcessingResult, output_path: str) -> None:
        """Generate a CSV report of extracted clips.

        Args:
            result: The processing result containing clip data.
            output_path: Path for the CSV file.
        """
        fieldnames = [
            "Clip_Filename", "Start_Time", "End_Time", "Duration",
            "Virality_Score", "Hook", "Description", "Hashtags",
            "Full_Caption", "Category", "Audio_Peak",
            "Motion_Score", "Face_Presence",
            "Zero_Crossing_Rate", "Composition",
            "ASMR_Quality", "Processing_Timestamp",
        ]

        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            # Single timestamp for all rows — consistent across the report
            run_timestamp = datetime.now().isoformat()
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for clip in result.clips:
                    # Skip clips deleted by caption/subtitle failure
                    if not clip.output_path:
                        continue
                    cap = clip.caption
                    if cap is not None:
                        hashtags_str = " ".join(cap.hashtags)
                        hook = cap.hook
                        description = cap.description
                        full_caption = cap.full_caption
                        category = cap.category
                    else:
                        hashtags_str = ""
                        hook = ""
                        description = ""
                        full_caption = ""
                        category = ""

                    writer.writerow({
                        "Clip_Filename": (
                            os.path.basename(clip.output_path)
                            if clip.output_path else ""
                        ),
                        "Start_Time": f"{clip.scene.start_time:.2f}",
                        "End_Time": f"{clip.scene.end_time:.2f}",
                        "Duration": f"{clip.scene.duration:.2f}",
                        "Virality_Score": f"{clip.virality.total_score:.2f}",
                        "Hook": hook,
                        "Description": description,
                        "Hashtags": hashtags_str,
                        "Full_Caption": full_caption,
                        "Category": category,
                        "Audio_Peak": f"{clip.audio.audio_peak_score:.4f}",
                        "Motion_Score": f"{clip.visual.motion_score:.4f}",
                        "Face_Presence": f"{clip.visual.face_presence:.4f}",
                        "Zero_Crossing_Rate": f"{clip.audio.zcr_score:.4f}",
                        "Composition": f"{clip.visual.composition_score:.4f}",
                        "ASMR_Quality": f"{clip.semantic.asmr_quality:.2f}",
                        "Processing_Timestamp": run_timestamp,
                    })

            logger.info("CSV report written to %s", output_path)
        except Exception as exc:
            error_msg = f"Failed to write CSV report: {exc}"
            logger.error(error_msg)
            result.errors.append(error_msg)

    @staticmethod
    def _slugify(text: str, max_len: int = 40) -> str:
        """Convert text to a filesystem-safe slug.

        Args:
            text: Text to slugify.
            max_len: Maximum slug length.

        Returns:
            Lowercase, hyphen-separated slug.
        """
        slug = re.sub(r"[^\w\s-]", "", text.lower())
        slug = re.sub(r"[\s_]+", "-", slug).strip("-")
        return slug[:max_len].rstrip("-")

    @staticmethod
    def _generate_thumbnail(clip_path: str) -> Optional[str]:
        """Extract a thumbnail from the midpoint of a clip.

        Uses FFmpeg to extract a single frame and scales to 720px wide.

        Args:
            clip_path: Path to the clip video file.

        Returns:
            Path to the thumbnail JPEG, or None on failure.
        """
        try:
            # Get clip duration via ffprobe
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "format=duration", "-of", "csv=p=0", clip_path],
                capture_output=True, text=True, timeout=10,
            )
            duration = float(probe.stdout.strip()) if probe.returncode == 0 else 0
            mid = duration / 2.0 if duration > 0 else 0.0

            thumb_path = os.path.splitext(clip_path)[0] + "_thumb.jpg"
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", clip_path,
                 "-ss", str(mid), "-vframes", "1",
                 "-vf", "scale=720:-1",
                 thumb_path],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and os.path.exists(thumb_path):
                return thumb_path
            return None
        except Exception as exc:
            logger.warning("Thumbnail generation failed for %s: %s", clip_path, exc)
            return None
