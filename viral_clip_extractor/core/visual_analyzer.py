"""
Visual analysis module using OpenCV.

Analyzes video segments for motion intensity (optical flow), face presence
(DNN SSD detector with Haar cascade fallback), visual interest (color
variance), and composition quality (rule-of-thirds scoring).
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from viral_clip_extractor.models import PipelineConfig, VisualFeatures
from viral_clip_extractor.utils.video_utils import get_cv2 as _shared_get_cv2

logger = logging.getLogger(__name__)

# Module-level cv2 reference (allows test patching via ``va_mod._cv2 = mock``)
_cv2 = None


def _get_cv2():  # type: ignore[no-untyped-def]
    """Return the module-local cv2 reference, falling back to the shared loader."""
    global _cv2
    if _cv2 is None:
        _cv2 = _shared_get_cv2()
    return _cv2


# Cached face detection models — loaded once, reused across all calls.
# Avoids O(N*M) model loads for N frames/clip across M clips.
_cached_dnn_net = None
_cached_dnn_checked = False
_cached_haar_cascade = None


def _validate_face_detection(bbox, frame_w, frame_h):
    """Validate a face detection bounding box for plausibility.

    Rejects detections that are too small, too large, or have extreme
    aspect ratios.  For small-area detections (< 3% of frame), faces
    with reasonable dimensions and non-square proportions are rescued
    — real faces on high-resolution frames legitimately occupy a small
    area fraction.

    Args:
        bbox: Tuple/list of (x, y, w, h) for the detected face.
        frame_w: Width of the source frame in pixels.
        frame_h: Height of the source frame in pixels.

    Returns:
        True if the face detection is plausible, False if not.
    """
    _, _, w, h = bbox
    face_area = w * h
    frame_area = frame_w * frame_h
    if frame_area <= 0:
        logger.debug("Rejecting face: invalid frame dimensions %dx%d", frame_w, frame_h)
        return False
    area_fraction = face_area / frame_area

    if area_fraction > 0.70:
        logger.debug(
            "Rejecting face: area %.1f%% exceeds 70%% of frame",
            area_fraction * 100,
        )
        return False

    if w > 0 and h > 0:
        if w / h >= 2.0 or h / w >= 2.0:
            logger.debug(
                "Rejecting face: aspect ratio w/h=%.2f, h/w=%.2f >= 2.0",
                w / h, h / w,
            )
            return False

    if area_fraction < 0.03:
        min_dim = min(w, h)
        min_frame_dim = min(frame_w, frame_h)
        # Rescue faces with reasonable dimensions and non-square proportions:
        # real faces are always slightly rectangular, never perfectly square.
        if min_dim >= 0.10 * min_frame_dim and max(w, h) / min(w, h) > 1.05:
            logger.debug(
                "Accepting face despite small area (%.1f%%): "
                "dimensions %dx%d are plausible",
                area_fraction * 100, w, h,
            )
            return True
        logger.debug(
            "Rejecting face: area %.1f%% below 3%% threshold "
            "(dimensions %dx%d)",
            area_fraction * 100, w, h,
        )
        return False

    return True


def validate_spatial_consistency(face_centers, frame_w):
    """Filter face center positions for spatial consistency.

    If the spread of face center X coordinates exceeds 30% of the
    frame width, the detections are too scattered to be reliable and
    all are rejected.

    Args:
        face_centers: List of face center X coordinates.
        frame_w: Frame width in pixels.

    Returns:
        The original list if consistent, empty list if too spread out.
    """
    if len(face_centers) <= 1:
        return face_centers

    spread = max(face_centers) - min(face_centers)
    if spread > 0.30 * frame_w:
        logger.debug(
            "Rejecting %d face centers: spread %d exceeds "
            "30%% of frame width %d",
            len(face_centers), spread, frame_w,
        )
        return []

    return face_centers


def _get_dnn_net(cv2):
    """Load and cache the DNN SSD face detection network.

    Returns the cached ``cv2.dnn.Net`` if model files are found, or
    ``None`` if no model files exist.
    """
    global _cached_dnn_net, _cached_dnn_checked
    if _cached_dnn_checked:
        return _cached_dnn_net

    _cached_dnn_checked = True
    try:
        model_dirs = [
            str(Path(__file__).resolve().parent.parent / "models"),
            str(Path.home() / ".vce" / "models"),
        ]
        for d in model_dirs:
            p = Path(d) / "deploy.prototxt"
            m = Path(d) / "res10_300x300_ssd_iter_140000.caffemodel"
            if p.exists() and m.exists():
                _cached_dnn_net = cv2.dnn.readNetFromCaffe(str(p), str(m))
                return _cached_dnn_net
    except Exception as e:
        logger.warning("DNN model loading failed: %s", e)
    logger.debug("DNN face model not found — using Haar cascade only")
    return None


def _get_haar_cascade(cv2):
    """Load and cache the Haar cascade face classifier."""
    global _cached_haar_cascade
    if _cached_haar_cascade is not None:
        return _cached_haar_cascade

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
    _cached_haar_cascade = cv2.CascadeClassifier(cascade_path)
    return _cached_haar_cascade


def _detect_faces_in_gray(cv2, gray_frame: np.ndarray) -> np.ndarray:
    """Detect faces in a grayscale frame using DNN SSD when available,
    falling back to Haar cascade.

    The DNN SSD detector handles side profiles, partial occlusion, and
    diverse skin tones far better than the legacy Haar cascade.  However,
    it requires external model files that are not bundled with OpenCV.
    Both detectors are loaded once and cached for all subsequent calls.

    Args:
        cv2: The OpenCV module.
        gray_frame: Single-channel grayscale image.

    Returns:
        An ndarray of (x, y, w, h) bounding boxes — may be empty.
    """
    # --- Try DNN SSD first (cached) ------------------------------------------
    net = _get_dnn_net(cv2)
    if net is not None:
        try:
            color = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            h, w = gray_frame.shape[:2]
            blob = cv2.dnn.blobFromImage(
                color, 1.0, (300, 300), (104.0, 177.0, 123.0),
            )
            net.setInput(blob)
            detections = net.forward()
            boxes = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    boxes.append([x1, y1, x2 - x1, y2 - y1])
            if boxes:
                logger.debug("DNN detector found %d face(s)", len(boxes))
                return np.array(boxes)
            logger.debug("DNN detector found 0 faces")
            return np.empty((0, 4), dtype=int)
        except Exception as e:
            logger.warning("DNN detection failed, falling back to Haar: %s", e)

    # --- Haar cascade fallback (cached) --------------------------------------
    face_cascade = _get_haar_cascade(cv2)
    if face_cascade.empty():
        logger.warning("Failed to load Haar cascade — face detection disabled")
        return np.empty((0, 4), dtype=int)
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
    )
    if len(faces) == 0:
        logger.debug("Haar cascade detector found 0 faces")
        return np.empty((0, 4), dtype=int)
    logger.debug("Haar cascade detector found %d face(s)", len(faces))
    return faces


class VisualAnalyzer:
    """Analyze visual composition and motion in video segments.

    Uses OpenCV for optical flow, DNN SSD / Haar cascade face detection,
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
                try:
                    cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
                    ret, frame = cap.read()
                except Exception as e:
                    logger.warning("Failed to read frame at %.2fs in %s: %s", ts, video_path, e)
                    continue
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
                # Farneback optical flow parameters tuned for 1 FPS sampling:
                # - winsize=31 (larger window compensates for larger inter-frame
                #   displacement at 1 FPS vs typical 24-30 FPS)
                # - levels=5 (more pyramid levels handle the bigger motion
                #   vectors between 1-second-apart frames)
                # - iterations=5 (more iterations improve convergence for
                #   large displacements)
                # - poly_n=7 / poly_sigma=1.5 (larger polynomial neighbourhood
                #   smooths noise from frame-to-frame jitter at low FPS)
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    curr_gray,
                    None,  # type: ignore[arg-type]
                    pyr_scale=0.5,
                    levels=5,
                    winsize=31,
                    iterations=5,
                    poly_n=7,
                    poly_sigma=1.5,
                    flags=0,
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                magnitudes.append(float(np.mean(mag)))
            except Exception as e:
                logger.debug("Optical flow computation failed for a frame pair: %s", e)
            prev_gray = curr_gray

        if not magnitudes:
            return 0.0

        mean_mag = float(np.mean(magnitudes))
        # Normalise: typical magnitude range 0–20 pixels → 0–1
        normalised = min(mean_mag / 20.0, 1.0)
        return normalised

    def _compute_face_presence(self, frames: list[np.ndarray]) -> float:
        """Compute the fraction of frames containing at least one face.

        Uses the DNN-based face detector (``cv2.dnn``) when model files
        are available, falling back to the bundled Haar cascade otherwise.

        Returns:
            A float in [0, 1] — ratio of frames with detected faces.
        """
        cv2 = _get_cv2()

        face_count = 0
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = _detect_faces_in_gray(cv2, gray)
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

        Detects faces via DNN/Haar (or, if none, bright/salient regions)
        and measures proximity to the four rule-of-thirds intersection points.

        Returns:
            A float in [0, 1] where 1 means subjects sit perfectly on thirds.
        """
        cv2 = _get_cv2()

        if not frames:
            return 0.0

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
            faces = _detect_faces_in_gray(cv2, gray)

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
            try:
                cap.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000.0)
                ret, frame = cap.read()
            except Exception as e:
                logger.warning(
                    "Failed to read frame at %.2fs in %s: %s",
                    time_seconds, video_path, e,
                )
                return 0
            if not ret or frame is None:
                logger.warning(
                    "Failed to read frame at %.2fs in %s", time_seconds, video_path
                )
                return 0
        finally:
            cap.release()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _detect_faces_in_gray(cv2, gray)
        return len(faces)
