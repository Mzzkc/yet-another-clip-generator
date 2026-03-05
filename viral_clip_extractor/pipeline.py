"""
Main pipeline orchestrator for the Viral Clip Extractor.

Coordinates scene detection, multi-modal analysis, virality scoring,
clip extraction, and optional caption generation into an end-to-end
workflow. Supports both local video files and YouTube URLs.
"""

import csv
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from viral_clip_extractor.models import (
    AudioFeatures,
    ClipData,
    PipelineConfig,
    ProcessingResult,
    SceneSegment,
    SemanticFeatures,
    ViralityScore,
    VisualFeatures,
)

logger = logging.getLogger(__name__)


class ViralClipPipeline:
    """End-to-end pipeline for extracting viral clips from video.

    Orchestrates scene detection, audio/visual/semantic analysis,
    virality scoring, clip extraction, and optional caption generation.

    Args:
        config: Pipeline configuration. Uses defaults if not provided.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()

        # Lazy-initialized components
        self._scene_detector = None
        self._audio_analyzer = None
        self._visual_analyzer = None
        self._semantic_analyzer = None
        self._virality_scorer = None
        self._clip_extractor = None
        self._caption_analyzer = None

    # ------------------------------------------------------------------
    # Component accessors (lazy init)
    # ------------------------------------------------------------------

    def _get_scene_detector(self):
        if self._scene_detector is None:
            from viral_clip_extractor.core.scene_detector import SceneDetector
            self._scene_detector = SceneDetector(config=self.config)
        return self._scene_detector

    def _get_audio_analyzer(self):
        if self._audio_analyzer is None:
            from viral_clip_extractor.core.audio_analyzer import AudioAnalyzer
            self._audio_analyzer = AudioAnalyzer()
        return self._audio_analyzer

    def _get_visual_analyzer(self):
        if self._visual_analyzer is None:
            from viral_clip_extractor.core.visual_analyzer import VisualAnalyzer
            self._visual_analyzer = VisualAnalyzer(config=self.config)
        return self._visual_analyzer

    def _get_semantic_analyzer(self):
        if self._semantic_analyzer is None:
            from viral_clip_extractor.core.semantic_analyzer import SemanticAnalyzer
            self._semantic_analyzer = SemanticAnalyzer(
                model=self.config.model_name,
                ollama_host=self.config.ollama_host,
            )
        return self._semantic_analyzer

    def _get_virality_scorer(self):
        if self._virality_scorer is None:
            from viral_clip_extractor.core.virality_scorer import ViralityScorer
            self._virality_scorer = ViralityScorer(config=self.config)
        return self._virality_scorer

    def _get_clip_extractor(self):
        if self._clip_extractor is None:
            from viral_clip_extractor.extractors.clip_extractor import ClipExtractor
            self._clip_extractor = ClipExtractor(config=self.config)
        return self._clip_extractor

    def _get_caption_analyzer(self):
        """Get caption analyzer (OllamaVideoAnalyzer from caption_generator)."""
        if self._caption_analyzer is None:
            try:
                from caption_generator import OllamaVideoAnalyzer
                self._caption_analyzer = OllamaVideoAnalyzer(
                    model=self.config.model_name,
                    ollama_host=self.config.ollama_host,
                )
            except ImportError:
                logger.warning("caption_generator not available — captions disabled")
                self._caption_analyzer = None
        return self._caption_analyzer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path: str,
        title: str = "",
        top_n: int = 10,
        min_score: float = 70.0,
    ) -> ProcessingResult:
        """Process a local video file through the full pipeline.

        Flow:
        1. Validate video and extract metadata
        2. Detect scenes
        3. Analyze each scene (audio, visual, semantic)
        4. Score virality
        5. Rank, filter, and select top clips
        6. Extract clip files
        7. Generate captions (if enabled)
        8. Return ProcessingResult

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

        # 1. Validate video
        if not os.path.exists(video_path):
            return ProcessingResult(
                video_path=video_path,
                video_title=title,
                clips=[],
                total_scenes=0,
                processing_time_seconds=0.0,
                errors=[f"Video not found: {video_path}"],
            )

        logger.info("Processing video: %s", video_path)

        # Extract metadata for title fallback
        if not title:
            try:
                from viral_clip_extractor.utils.video_utils import extract_metadata
                meta = extract_metadata(video_path)
                title = meta.get("filename", Path(video_path).stem)
            except Exception as exc:
                logger.warning("Could not extract metadata: %s", exc)
                title = Path(video_path).stem

        # 2. Detect scenes
        logger.info("Step 1/5: Detecting scenes...")
        try:
            scenes = self._get_scene_detector().detect_scenes(video_path)
        except Exception as exc:
            error_msg = f"Scene detection failed: {exc}"
            logger.error(error_msg)
            return ProcessingResult(
                video_path=video_path,
                video_title=title,
                clips=[],
                total_scenes=0,
                processing_time_seconds=time.time() - start_time_wall,
                errors=[error_msg],
            )

        total_scenes = len(scenes)
        logger.info("Detected %d scenes", total_scenes)

        if total_scenes == 0:
            return ProcessingResult(
                video_path=video_path,
                video_title=title,
                clips=[],
                total_scenes=0,
                processing_time_seconds=time.time() - start_time_wall,
                errors=["No scenes detected in video"],
            )

        # 3-4. Analyze and score each scene
        logger.info("Step 2/5: Analyzing %d segments...", total_scenes)
        clip_data_list: list[ClipData] = []

        for i, scene in enumerate(scenes):
            pct = ((i + 1) / total_scenes) * 100
            print(f"Processing segment {i + 1}/{total_scenes} ({pct:.0f}%)")

            try:
                clip_data = self._analyze_segment(video_path, scene, title)
                clip_data_list.append(clip_data)
            except Exception as exc:
                error_msg = f"Segment {i} analysis failed: {exc}"
                logger.error(error_msg)
                errors.append(error_msg)

        # 5. Rank by score, filter, take top_n
        logger.info("Step 3/5: Ranking and filtering clips...")
        clip_data_list.sort(key=lambda c: c.virality.total_score, reverse=True)
        filtered = [c for c in clip_data_list if c.virality.total_score >= min_score]
        selected = filtered[:top_n]

        logger.info(
            "Selected %d clips (from %d scored, %d above threshold %.1f)",
            len(selected), len(clip_data_list), len(filtered), min_score,
        )

        # 6. Extract clip files
        logger.info("Step 4/5: Extracting %d clip files...", len(selected))
        output_dir = self.config.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        extractor = self._get_clip_extractor()
        for i, clip in enumerate(selected):
            score_int = int(clip.virality.total_score)
            filename = f"clip_{i + 1:02d}_score{score_int}.mp4"
            output_path = os.path.join(output_dir, filename)

            try:
                success = extractor.extract_clip(
                    video_path,
                    clip.scene.start_time,
                    clip.scene.end_time,
                    output_path,
                )
                if success:
                    clip.output_path = output_path
                else:
                    errors.append(f"Failed to extract clip {i + 1}")
            except Exception as exc:
                error_msg = f"Clip {i + 1} extraction failed: {exc}"
                logger.error(error_msg)
                errors.append(error_msg)

        # 7. Generate captions (if enabled)
        if self.config.enable_captions:
            logger.info("Step 5/5: Generating captions...")
            analyzer = self._get_caption_analyzer()
            if analyzer is not None:
                for i, clip in enumerate(selected):
                    if clip.output_path and os.path.exists(clip.output_path):
                        try:
                            caption_data = analyzer.analyze_video(
                                clip.output_path, title
                            )
                            if caption_data:
                                clip.caption = {
                                    "hook": caption_data.hook,
                                    "description": caption_data.description,
                                    "hashtags": caption_data.hashtags,
                                    "category": caption_data.category,
                                    "virality_score": caption_data.virality_score,
                                    "full_caption": caption_data.full_caption,
                                }
                        except Exception as exc:
                            logger.warning(
                                "Caption generation failed for clip %d: %s", i + 1, exc
                            )
            else:
                logger.info("Caption analyzer not available, skipping captions")
        else:
            logger.info("Step 5/5: Captions disabled, skipping")

        # 8. Build result
        processing_time = time.time() - start_time_wall
        result = ProcessingResult(
            video_path=video_path,
            video_title=title,
            clips=selected,
            total_scenes=total_scenes,
            processing_time_seconds=round(processing_time, 2),
            errors=errors,
        )

        # Generate CSV
        csv_path = os.path.join(output_dir, "clips_report.csv")
        self._generate_csv(result, csv_path)

        logger.info(
            "Pipeline complete: %d clips extracted in %.1fs (%d errors)",
            len(selected), processing_time, len(errors),
        )
        print(f"\nDone! {len(selected)} clips extracted in {processing_time:.1f}s")
        print(f"Output: {output_dir}")
        print(f"CSV report: {csv_path}")

        return result

    def process_youtube(
        self,
        url: str,
        top_n: int = 10,
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
        from viral_clip_extractor.youtube_downloader import YouTubeDownloader

        download_dir = os.path.join(self.config.output_dir, "downloads")
        downloader = YouTubeDownloader(output_dir=download_dir)

        logger.info("Downloading YouTube video: %s", url)
        print("Downloading YouTube video...")

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
        print(f"Downloaded: {title}")

        result = self.process_video(
            video_path=video_path,
            title=title,
            top_n=top_n,
            min_score=min_score,
        )

        return result

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _analyze_segment(
        self, video_path: str, scene: SceneSegment, title: str
    ) -> ClipData:
        """Analyze a single scene segment for virality signals.

        Runs audio, visual, and (optionally) semantic analysis, then
        computes a composite virality score.

        Args:
            video_path: Path to the source video.
            scene: The scene boundaries.
            title: Video title for semantic context.

        Returns:
            A ClipData instance with all analysis results.
        """
        start = scene.start_time
        end = scene.end_time

        # Audio analysis
        try:
            audio = self._get_audio_analyzer().analyze_segment(video_path, start, end)
        except Exception as exc:
            logger.warning(
                "Audio analysis failed for scene %d: %s", scene.scene_index, exc
            )
            audio = AudioFeatures(
                audio_peak_score=0.0, high_freq_score=0.0,
                dynamic_range=0.0, zcr_score=0.0,
            )

        # Visual analysis
        try:
            visual = self._get_visual_analyzer().analyze_segment(video_path, start, end)
        except Exception as exc:
            logger.warning(
                "Visual analysis failed for scene %d: %s", scene.scene_index, exc
            )
            visual = VisualFeatures(
                motion_score=0.0, face_presence=0.0,
                visual_interest=0.0, composition_score=0.0,
            )

        # Semantic analysis (optional)
        semantic: Optional[SemanticFeatures] = None
        if self.config.enable_semantic:
            try:
                semantic = self._get_semantic_analyzer().analyze_segment(
                    video_path, start, end, title=title,
                )
            except Exception as exc:
                logger.warning(
                    "Semantic analysis failed for scene %d: %s",
                    scene.scene_index, exc,
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
            "clip_filename", "start_time", "end_time", "duration",
            "virality_score", "hook", "description", "hashtags",
            "full_caption", "category", "audio_peak", "motion_score",
            "face_presence", "asmr_quality", "processing_timestamp",
        ]

        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for clip in result.clips:
                    caption = clip.caption or {}
                    hashtags_list = caption.get("hashtags", [])
                    if isinstance(hashtags_list, list):
                        hashtags_str = " ".join(hashtags_list)
                    else:
                        hashtags_str = str(hashtags_list)

                    asmr_quality = 0.0
                    if clip.semantic is not None:
                        asmr_quality = clip.semantic.asmr_quality

                    writer.writerow({
                        "clip_filename": (
                            os.path.basename(clip.output_path)
                            if clip.output_path else ""
                        ),
                        "start_time": f"{clip.scene.start_time:.2f}",
                        "end_time": f"{clip.scene.end_time:.2f}",
                        "duration": f"{clip.scene.duration:.2f}",
                        "virality_score": f"{clip.virality.total_score:.2f}",
                        "hook": caption.get("hook", ""),
                        "description": caption.get("description", ""),
                        "hashtags": hashtags_str,
                        "full_caption": caption.get("full_caption", ""),
                        "category": caption.get("category", ""),
                        "audio_peak": f"{clip.audio.audio_peak_score:.4f}",
                        "motion_score": f"{clip.visual.motion_score:.4f}",
                        "face_presence": f"{clip.visual.face_presence:.4f}",
                        "asmr_quality": f"{asmr_quality:.2f}",
                        "processing_timestamp": datetime.now().isoformat(),
                    })

            logger.info("CSV report written to %s", output_path)
        except Exception as exc:
            logger.error("Failed to write CSV report: %s", exc)
