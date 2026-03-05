"""
Visual analysis module using OpenCV.

Analyzes video segments for motion intensity (optical flow), face presence
(Haar cascades), visual interest (color variance), and composition quality
(rule-of-thirds scoring).
"""

import logging
from typing import Optional

import numpy as np

from viral_clip_extractor.models import PipelineConfig, VisualFeatures

logger = logging.getLogger(__name__)

# Lazy-loaded OpenCV reference — set on first use
_cv2 = None


def _get_cv2():  # type: ignore[no-untyped-def]
    """Lazy-import cv2 so the module can be imported even without OpenCV."""
    global _cv2
    if _cv2 is None:
        try:
            import cv2

            _cv2 = cv2
        except ImportError:
            raise ImportError(
                "OpenCV (cv2) is required for visual analysis. "
                "Install it with: pip install opencv-python-headless"
            )
    return _cv2


class VisualAnalyzer:
    """Analyze visual composition and motion in video segments.

    Uses OpenCV for optical flow, Haar cascade face detection,
    HSV color variance, and rule-of-thirds composition scoring.

    Args:
        config: Pipeline configuration.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()

    def analyze_segment(
        self, video_path: str, start_time: float, end_time: float
    ) -> VisualFeatures:
        """Analyze visual features for a time range within a video.

        Samples one frame per second and computes:
        - motion_score: mean optical flow magnitude between consecutive frames
        - face_presence: fraction of sampled frames containing faces
        - visual_interest: color variance in HSV space
        - composition_score: rule-of-thirds scoring

        Args:
            video_path: Path to video file.
            start_time: Segment start in seconds.
            end_time: Segment end in seconds.

        Returns:
            A VisualFeatures instance with computed scores (all 0.0–1.0).
        """
        frames = self._sample_frames(video_path, start_time, end_time, fps=1)

        if not frames:
            logger.warning(
                "No frames sampled from %s (%.1f–%.1f); returning zero features",
                video_path,
                start_time,
                end_time,
            )
            return VisualFeatures(
                motion_score=0.0,
                face_presence=0.0,
                visual_interest=0.0,
                composition_score=0.0,
            )

        motion = self._compute_motion(frames)
        face_pres = self._compute_face_presence(frames)
        interest = self._compute_visual_interest(frames)
        composition = self._compute_composition(frames)

        logger.debug(
            "Visual features for %s [%.1f–%.1f]: motion=%.3f face=%.3f "
            "interest=%.3f composition=%.3f",
            video_path,
            start_time,
            end_time,
            motion,
            face_pres,
            interest,
            composition,
        )

        return VisualFeatures(
            motion_score=motion,
            face_presence=face_pres,
            visual_interest=interest,
            composition_score=composition,
        )

    def _sample_frames(
        self, video_path: str, start: float, end: float, fps: int = 1
    ) -> list[np.ndarray]:
        """Sample frames from a video segment at a fixed rate.

        Args:
            video_path: Path to video file.
            start: Start time in seconds.
            end: End time in seconds.
            fps: Frames to sample per second (default 1).

        Returns:
            List of BGR numpy arrays (HxWx3).
        """
        cv2 = _get_cv2()
        frames: list[np.ndarray] = []

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Could not open video for frame sampling: %s", video_path)
            return frames

        try:
            duration = end - start
            if duration <= 0:
                logger.warning("Invalid segment duration: %.1f–%.1f", start, end)
                return frames

            # Determine sample timestamps
            interval = 1.0 / fps
            timestamps = []
            t = start
            while t < end:
                timestamps.append(t)
                t += interval

            for ts in timestamps:
                cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames.append(frame)
                else:
                    logger.debug("Failed to read frame at %.2fs in %s", ts, video_path)
        finally:
            cap.release()

        logger.debug("Sampled %d frames from %s", len(frames), video_path)
        return frames

    def _compute_motion(self, frames: list[np.ndarray]) -> float:
        """Compute motion score using Farneback optical flow.

        Converts consecutive frame pairs to grayscale, computes dense
        optical flow, and returns the normalised mean magnitude.

        Returns:
            A float in [0, 1] (clamped) representing motion intensity.
        """
        cv2 = _get_cv2()

        if len(frames) < 2:
            return 0.0

        magnitudes: list[float] = []
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        for frame in frames[1:]:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    curr_gray,
                    None,  # type: ignore[arg-type]
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                magnitudes.append(float(np.mean(mag)))
            except Exception:
                logger.debug("Optical flow computation failed for a frame pair")
            prev_gray = curr_gray

        if not magnitudes:
            return 0.0

        mean_mag = float(np.mean(magnitudes))
        # Normalise: typical magnitude range 0–20 pixels → 0–1
        normalised = min(mean_mag / 20.0, 1.0)
        return normalised

    def _compute_face_presence(self, frames: list[np.ndarray]) -> float:
        """Compute the fraction of frames containing at least one face.

        Uses the Haar cascade frontal-face detector bundled with OpenCV.

        Returns:
            A float in [0, 1] — ratio of frames with detected faces.
        """
        cv2 = _get_cv2()

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
        face_cascade = cv2.CascadeClassifier(cascade_path)

        if face_cascade.empty():
            logger.warning("Failed to load Haar cascade — face detection disabled")
            return 0.0

        face_count = 0
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            if len(faces) > 0:
                face_count += 1

        return face_count / len(frames) if frames else 0.0

    def _compute_visual_interest(self, frames: list[np.ndarray]) -> float:
        """Compute visual interest from HSV color variance.

        Converts each frame to HSV, computes the standard deviation of
        the hue and saturation channels, then averages across frames.

        Returns:
            A float in [0, 1] representing colour richness / diversity.
        """
        cv2 = _get_cv2()

        if not frames:
            return 0.0

        variances: list[float] = []
        for frame in frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h_std = float(np.std(hsv[:, :, 0]))  # Hue
            s_std = float(np.std(hsv[:, :, 1]))  # Saturation
            # Combine hue and saturation variance (V is brightness, less useful)
            combined = (h_std / 180.0 + s_std / 255.0) / 2.0
            variances.append(combined)

        mean_var = float(np.mean(variances))
        # Normalise: typical combined std range 0–0.5 → 0–1
        return min(mean_var / 0.5, 1.0)

    def _compute_composition(self, frames: list[np.ndarray]) -> float:
        """Score composition by rule-of-thirds alignment.

        Detects faces (or, if none, bright/salient regions) and measures
        how close they are to the four rule-of-thirds intersection points.

        Returns:
            A float in [0, 1] where 1 means subjects sit perfectly on thirds.
        """
        cv2 = _get_cv2()

        if not frames:
            return 0.0

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
        face_cascade = cv2.CascadeClassifier(cascade_path)

        scores: list[float] = []
        for frame in frames:
            h, w = frame.shape[:2]
            # Rule-of-thirds intersection points (normalised)
            thirds_pts = [
                (w / 3, h / 3),
                (2 * w / 3, h / 3),
                (w / 3, 2 * h / 3),
                (2 * w / 3, 2 * h / 3),
            ]

            # Try face centres as subjects
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) > 0:
                # Use centre of first (largest) face
                fx, fy, fw, fh = faces[0]
                cx, cy = fx + fw / 2.0, fy + fh / 2.0
            else:
                # Fallback: centre of brightest region
                blurred = cv2.GaussianBlur(gray, (21, 21), 0)
                _, _, _, max_loc = cv2.minMaxLoc(blurred)
                cx, cy = float(max_loc[0]), float(max_loc[1])

            # Distance to nearest thirds point, normalised by image diagonal
            diag = np.sqrt(w ** 2 + h ** 2)
            min_dist = min(
                np.sqrt((cx - px) ** 2 + (cy - py) ** 2) for px, py in thirds_pts
            )
            # Convert distance to score: 0 distance → 1.0, far → 0.0
            score = max(1.0 - (min_dist / (diag * 0.25)), 0.0)
            scores.append(score)

        return float(np.mean(scores)) if scores else 0.0

    def detect_faces(self, video_path: str, time_seconds: float) -> int:
        """Count faces in a single frame at the given timestamp.

        Args:
            video_path: Path to video file.
            time_seconds: Timestamp in seconds.

        Returns:
            Number of faces detected.
        """
        cv2 = _get_cv2()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Could not open video: %s", video_path)
            return 0

        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000.0)
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning(
                    "Failed to read frame at %.2fs in %s", time_seconds, video_path
                )
                return 0
        finally:
            cap.release()

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
        face_cascade = cv2.CascadeClassifier(cascade_path)

        if face_cascade.empty():
            logger.warning("Failed to load Haar cascade")
            return 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return len(faces)
