"""
Scene detection module using PySceneDetect.

Detects natural scene boundaries in video files using adaptive detection,
optimized for ASMR content with subtle transitions. Merges short scenes
and splits overly long ones to keep clips within platform duration bounds.
"""

import logging
from pathlib import Path
from typing import Optional

from yacg.models import PipelineConfig, SceneSegment

logger = logging.getLogger(__name__)

try:
    from scenedetect import AdaptiveDetector, detect, open_video

    _HAS_SCENEDETECT = True
except ImportError:
    _HAS_SCENEDETECT = False


class SceneDetector:
    """Detect scene boundaries using PySceneDetect's AdaptiveDetector.

    Optimized for ASMR content with lower thresholds (2.0-4.0) to catch
    subtle transitions. Post-processes results to merge very short scenes
    and split overly long ones.

    Args:
        config: Pipeline configuration with scene detection parameters.
        threshold: AdaptiveDetector threshold (used when *config* is None).
        min_scene_len: Minimum scene duration in seconds (used when *config* is None).
        max_scene_len: Maximum scene duration in seconds (used when *config* is None).
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        threshold: float = 3.0,
        min_scene_len: float = 7.0,
        max_scene_len: float = 60.0,
    ) -> None:
        if config is not None:
            self.config = config
            self.threshold = config.scene_threshold
            self.min_scene_len = config.min_scene_len
            self.max_scene_len = config.max_scene_len
        else:
            self.config = PipelineConfig()
            self.threshold = threshold
            self.min_scene_len = min_scene_len
            self.max_scene_len = max_scene_len

    def detect_scenes(self, video_path: str) -> list[SceneSegment]:
        """Detect scene boundaries in a video file.

        Uses PySceneDetect's AdaptiveDetector with ASMR-optimized thresholds.
        Detected scenes are post-processed to merge short scenes and split
        overly long ones.

        Args:
            video_path: Path to the video file.

        Returns:
            A list of SceneSegment instances representing detected scenes.

        Raises:
            ImportError: If PySceneDetect is not installed.
            FileNotFoundError: If the video file does not exist.
            RuntimeError: If scene detection fails (e.g., corrupt file).
        """
        if not _HAS_SCENEDETECT:
            raise ImportError(
                "PySceneDetect is required for scene detection. "
                "Install it with: pip install scenedetect[opencv]"
            )

        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info(
            "Detecting scenes in %s (threshold=%.1f, min=%.1fs, max=%.1fs)",
            path.name,
            self.threshold,
            self.min_scene_len,
            self.max_scene_len,
        )

        try:
            scene_list = detect(
                str(path),
                AdaptiveDetector(adaptive_threshold=self.threshold),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Scene detection failed for {video_path}: {exc}"
            ) from exc

        if not scene_list:
            # No boundaries found — treat entire video as one scene.
            logger.warning(
                "No scene boundaries detected in %s — treating as single scene",
                path.name,
            )
            duration = self._get_video_duration(str(path))
            if duration is None or duration < 0.1:
                logger.warning(
                    "Video %s is too short or unreadable (duration=%.2fs)",
                    path.name,
                    duration or 0.0,
                )
                return []
            return [SceneSegment(start_time=0.0, end_time=duration, scene_index=0)]

        # Convert to SceneSegment objects
        segments = [
            SceneSegment(
                start_time=start.get_seconds(),
                end_time=end.get_seconds(),
                scene_index=i,
            )
            for i, (start, end) in enumerate(scene_list)
        ]

        logger.info("Detected %d raw scenes in %s", len(segments), path.name)

        # Post-process: merge short scenes, then split long ones
        segments = self.merge_short_scenes(segments, self.min_scene_len)
        segments = self.split_long_scenes(segments, self.max_scene_len)

        # Re-index after merge/split
        for i, seg in enumerate(segments):
            seg.scene_index = i

        logger.info(
            "Final scene count: %d (after merge/split) in %s",
            len(segments),
            path.name,
        )
        for seg in segments:
            logger.debug(
                "  Scene %d: %.1f-%.1fs (%.1fs)",
                seg.scene_index,
                seg.start_time,
                seg.end_time,
                seg.duration,
            )

        return segments

    def merge_short_scenes(
        self, scenes: list[SceneSegment], min_duration: float
    ) -> list[SceneSegment]:
        """Merge scenes shorter than *min_duration* with adjacent scenes.

        Short scenes are merged into their preceding neighbor. If the first
        scene is short and the next one is long enough, they are combined.

        Args:
            scenes: List of detected scenes.
            min_duration: Minimum scene length in seconds.

        Returns:
            A filtered list of scenes with short scenes merged.
        """
        if len(scenes) <= 1:
            return list(scenes)

        merged: list[SceneSegment] = []

        for scene in scenes:
            if merged and scene.duration < min_duration:
                # Merge short scene into previous neighbor
                prev = merged[-1]
                merged[-1] = SceneSegment(
                    start_time=prev.start_time,
                    end_time=scene.end_time,
                    scene_index=prev.scene_index,
                )
                logger.debug(
                    "Merged short scene (%.1fs) into previous", scene.duration
                )
            elif not merged:
                # First scene — always add (handle shortness on next iteration)
                merged.append(scene)
            else:
                # Current scene is long enough
                if merged[-1].duration < min_duration:
                    # Previous scene is still short — merge it forward
                    prev = merged[-1]
                    merged[-1] = SceneSegment(
                        start_time=prev.start_time,
                        end_time=scene.end_time,
                        scene_index=prev.scene_index,
                    )
                    logger.debug(
                        "Merged previous short scene (%.1fs) into current",
                        prev.duration,
                    )
                else:
                    merged.append(scene)

        count = len(scenes) - len(merged)
        if count > 0:
            logger.info(
                "Merged %d short scenes (threshold: %.1fs)", count, min_duration
            )

        return merged

    def split_long_scenes(
        self, scenes: list[SceneSegment], max_duration: float
    ) -> list[SceneSegment]:
        """Split scenes longer than *max_duration* at midpoints.

        Long scenes are recursively halved until every sub-scene fits within
        the maximum duration.

        Args:
            scenes: List of detected scenes.
            max_duration: Maximum scene length in seconds.

        Returns:
            A list of scenes with long scenes divided.
        """
        result: list[SceneSegment] = []

        for scene in scenes:
            if scene.duration <= max_duration:
                result.append(scene)
            else:
                mid = (scene.start_time + scene.end_time) / 2.0
                first_half = SceneSegment(
                    start_time=scene.start_time,
                    end_time=mid,
                    scene_index=scene.scene_index,
                )
                second_half = SceneSegment(
                    start_time=mid,
                    end_time=scene.end_time,
                    scene_index=scene.scene_index,
                )
                logger.debug(
                    "Split long scene (%.1fs) at midpoint %.1fs",
                    scene.duration,
                    mid,
                )
                # Recurse in case halves are still too long
                result.extend(self.split_long_scenes([first_half], max_duration))
                result.extend(
                    self.split_long_scenes([second_half], max_duration)
                )

        return result

    def _get_video_duration(self, video_path: str) -> Optional[float]:
        """Get video duration in seconds using scenedetect's open_video.

        Args:
            video_path: Path to the video file.

        Returns:
            Duration in seconds, or None if it cannot be determined.
        """
        try:
            video = open_video(video_path)
            duration = video.duration.get_seconds()
            return duration
        except Exception as exc:
            logger.warning(
                "Could not determine video duration for %s: %s",
                video_path,
                exc,
            )
            return None
