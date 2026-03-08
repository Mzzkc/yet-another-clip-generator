"""
Smart cropping module for vertical video reformatting.

Performs intelligent cropping to 9:16 (or custom) aspect ratio with
face-aware positioning, keeping subjects centered in the output frame.
Handles already-vertical and square videos gracefully.
"""

import logging
from typing import Optional

from yacg.core.visual_analyzer import (
    _detect_faces_in_gray,
    _validate_face_detection,
    validate_spatial_consistency,
)
from yacg.models import PipelineConfig
from yacg.utils.video_utils import get_cv2 as _shared_get_cv2

logger = logging.getLogger(__name__)

# Module-level cv2 reference (allows test patching via ``sc_mod._cv2 = mock``)
_cv2 = None


def _get_cv2():  # type: ignore[no-untyped-def]
    """Return the module-local cv2 reference, falling back to the shared loader."""
    global _cv2
    if _cv2 is None:
        _cv2 = _shared_get_cv2()
    return _cv2


class SmartCropper:
    """Intelligently crop horizontal video to vertical format.

    Uses face detection to keep subjects in frame when cropping from
    landscape to portrait orientation. Falls back to center-crop when
    no faces are detected.

    Args:
        config: Pipeline configuration.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()

    def get_crop_params(
        self,
        video_path: str,
        start_time: float = 0,
        target_ratio: float = 9 / 16,
        end_time: float | None = None,
    ) -> dict[str, int]:
        """Calculate crop parameters for vertical reformatting.

        Samples multiple frames across the segment for face detection,
        using the median face center X to resist outliers. Falls back
        to center crop when no faces are detected.

        Args:
            video_path: Source video file path.
            start_time: Segment start timestamp (seconds).
            target_ratio: Desired width/height ratio (default 9/16 = 0.5625).
            end_time: Segment end timestamp (seconds) for multi-frame sampling.

        Returns:
            Dict with keys: crop_x, crop_y, crop_w, crop_h (all ints, pixels).
        """
        cv2 = _get_cv2()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Could not open video: %s", video_path)
            return {"crop_x": 0, "crop_y": 0, "crop_w": 0, "crop_h": 0}

        try:
            src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Build sample timestamps: 3 frames at 25%, 50%, 75% of segment
            if end_time is not None and end_time > start_time:
                duration = end_time - start_time
                sample_times = [
                    start_time + duration * 0.25,
                    start_time + duration * 0.50,
                    start_time + duration * 0.75,
                ]
            else:
                sample_times = [start_time]

            # Read frames and attempt face detection on each
            face_centers: list[int] = []
            total_raw_detections = 0
            total_validated = 0
            for t in sample_times:
                try:
                    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                    ret, frame = cap.read()
                except Exception as e:
                    logger.warning("Failed to read frame at %.2fs: %s", t, e)
                    continue
                if ret and frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = _detect_faces_in_gray(cv2, gray)
                    raw = len(faces)
                    total_raw_detections += raw
                    if raw > 0:
                        frame_h, frame_w = frame.shape[:2]
                        valid = [
                            f for f in faces
                            if _validate_face_detection(tuple(f), frame_w, frame_h)
                        ]
                        total_validated += len(valid)
                        if valid:
                            largest = max(valid, key=lambda f: f[2] * f[3])
                            fx, _, fw, _ = largest
                            center_x = int(fx + fw / 2)
                            face_centers.append(center_x)
                            logger.debug(
                                "Frame at %.2fs: accepted face at x=%d "
                                "(%d/%d passed validation)",
                                t, center_x, len(valid), raw,
                            )
                        else:
                            logger.debug(
                                "Frame at %.2fs: all %d detection(s) "
                                "rejected by validation",
                                t, raw,
                            )
        finally:
            cap.release()

        if src_w == 0 or src_h == 0:
            logger.error("Could not determine video dimensions for %s", video_path)
            return {"crop_x": 0, "crop_y": 0, "crop_w": 0, "crop_h": 0}

        current_ratio = src_w / src_h

        # Already vertical or square — no horizontal cropping needed
        if current_ratio <= target_ratio:
            logger.info(
                "Video %s is already vertical/square (%.2f <= %.2f), no crop needed",
                video_path,
                current_ratio,
                target_ratio,
            )
            return {"crop_x": 0, "crop_y": 0, "crop_w": src_w, "crop_h": src_h}

        # Target dimensions: keep full height, compute target width
        crop_h = src_h
        crop_w = int(src_h * target_ratio)
        # Ensure crop width doesn't exceed source
        crop_w = min(crop_w, src_w)

        face_aware = False
        # Apply spatial consistency check before using face centers
        if face_centers:
            face_centers = validate_spatial_consistency(face_centers, src_w)
        if face_centers:
            # Use median of detected face centers (robust to outliers)
            sorted_centers = sorted(face_centers)
            median_idx = len(sorted_centers) // 2
            face_center_x = sorted_centers[median_idx]
            crop_x = int(face_center_x - crop_w / 2)
            crop_x = max(0, min(crop_x, src_w - crop_w))
            face_aware = True
        else:
            # No faces detected in any sample — center crop
            crop_x = (src_w - crop_w) // 2

        crop_y = 0  # Always start from top for vertical crops

        logger.info(
            "Crop params for %s: x=%d y=%d w=%d h=%d "
            "(face_aware=%s, detected=%d, validated=%d, samples=%d/%d)",
            video_path,
            crop_x,
            crop_y,
            crop_w,
            crop_h,
            face_aware,
            total_raw_detections,
            total_validated,
            len(face_centers),
            len(sample_times),
        )

        return {
            "crop_x": crop_x,
            "crop_y": crop_y,
            "crop_w": crop_w,
            "crop_h": crop_h,
        }

    def get_ffmpeg_filter(
        self,
        video_path: str,
        start_time: float = 0,
        target_ratio: float = 9 / 16,
        end_time: float | None = None,
    ) -> str:
        """Generate an FFmpeg ``-vf`` crop filter string.

        Args:
            video_path: Source video file path.
            start_time: Timestamp for face detection sampling.
            target_ratio: Desired width/height ratio (default 9/16).
            end_time: End timestamp for multi-frame sampling range.

        Returns:
            FFmpeg filter string, e.g. ``crop=608:1080:236:0``.
        """
        params = self.get_crop_params(video_path, start_time, target_ratio, end_time)

        crop_w = params["crop_w"]
        crop_h = params["crop_h"]
        crop_x = params["crop_x"]
        crop_y = params["crop_y"]

        if crop_w == 0 or crop_h == 0:
            logger.warning("Invalid crop dimensions — returning empty filter")
            return ""

        return f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}"

    def get_crop_filter(
        self, video_path: str, width: int, height: int
    ) -> str:
        """Generate an FFmpeg crop filter string for vertical output.

        Convenience wrapper matching the original stub signature.

        Args:
            video_path: Source video (used for face detection sampling).
            width: Source video width in pixels.
            height: Source video height in pixels.

        Returns:
            An FFmpeg -vf filter string (e.g. ``crop=608:1080:236:0``).
        """
        return self.get_ffmpeg_filter(video_path, start_time=0, target_ratio=9 / 16)

    def detect_subject_center(
        self, video_path: str, time_seconds: float
    ) -> tuple[int, int]:
        """Detect the primary subject center point in a frame.

        Args:
            video_path: Source video path.
            time_seconds: Timestamp to sample.

        Returns:
            (x, y) center coordinates of the primary subject.
        """
        cv2 = _get_cv2()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Could not open video: %s", video_path)
            return (0, 0)

        try:
            src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            try:
                cap.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000.0)
                ret, frame = cap.read()
            except Exception as e:
                logger.warning(
                    "Failed to read frame at %.2fs in %s: %s",
                    time_seconds, video_path, e,
                )
                ret, frame = False, None
        finally:
            cap.release()

        if src_w == 0 or src_h == 0:
            logger.error("Corrupt video headers (0x0) for %s", video_path)
            return (0, 0)

        if not ret or frame is None:
            logger.warning("Could not read frame at %.2fs", time_seconds)
            return (src_w // 2, src_h // 2)

        # Try face detection (DNN SSD with Haar fallback)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _detect_faces_in_gray(cv2, gray)
        if len(faces) > 0:
            frame_h, frame_w = frame.shape[:2]
            valid_faces = [
                f for f in faces
                if _validate_face_detection(tuple(f), frame_w, frame_h)
            ]
            if valid_faces:
                largest = max(valid_faces, key=lambda f: f[2] * f[3])
                fx, fy, fw, fh = largest
                logger.debug(
                    "Subject center from validated face at (%d, %d), "
                    "validated %d/%d detections",
                    int(fx + fw / 2), int(fy + fh / 2),
                    len(valid_faces), len(faces),
                )
                return (int(fx + fw / 2), int(fy + fh / 2))
            logger.debug(
                "All %d face detection(s) rejected by validation in "
                "detect_subject_center, falling back to brightness",
                len(faces),
            )

        # Fallback: centre of brightest region (reuse existing gray)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        _, _, _, max_loc = cv2.minMaxLoc(blurred)
        return (int(max_loc[0]), int(max_loc[1]))

    def _brightness_center_x_from_frame(self, frame) -> Optional[int]:
        """Find horizontal center of the brightest region in a pre-read frame.

        Args:
            frame: BGR numpy array, or None if frame read failed.

        Returns:
            X coordinate of brightness center, or None on failure.
        """
        if frame is None:
            return None

        cv2 = _get_cv2()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        _, _, _, max_loc = cv2.minMaxLoc(blurred)
        return int(max_loc[0])

    def _brightness_center_x(
        self, video_path: str, time_seconds: float, src_w: int
    ) -> Optional[int]:
        """Find horizontal center of the brightest region in a frame.

        Opens the video to read the frame. Prefer
        ``_brightness_center_x_from_frame`` when a frame is already available.

        Args:
            video_path: Path to video file.
            time_seconds: Timestamp in seconds to sample.
            src_w: Source video width (for center fallback).

        Returns:
            X coordinate of brightness center, or None on failure.
        """
        cv2 = _get_cv2()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000.0)
            ret, frame = cap.read()
        finally:
            cap.release()

        if not ret or frame is None:
            return None

        return self._brightness_center_x_from_frame(frame)

    def _detect_face_center_x_from_frame(self, frame) -> Optional[int]:
        """Detect the horizontal centre of the primary face in a pre-read frame.

        Uses DNN SSD detector when model files are available, falling
        back to the Haar cascade bundled with OpenCV.

        Args:
            frame: BGR numpy array, or None if frame read failed.

        Returns:
            X coordinate of the face centre, or None if no face found.
        """
        if frame is None:
            return None

        cv2 = _get_cv2()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _detect_faces_in_gray(cv2, gray)

        if len(faces) == 0:
            return None

        # Validate each face detection against frame dimensions
        frame_h, frame_w = frame.shape[:2]
        valid_faces = [
            f for f in faces
            if _validate_face_detection(tuple(f), frame_w, frame_h)
        ]

        if len(valid_faces) == 0:
            logger.debug(
                "All %d face detection(s) rejected by validation",
                len(faces),
            )
            return None

        # Use the largest valid face (highest area)
        largest = max(valid_faces, key=lambda f: f[2] * f[3])
        fx, _, fw, _ = largest
        center_x = int(fx + fw / 2)

        logger.debug(
            "Face center at x=%d (validated %d/%d faces)",
            center_x, len(valid_faces), len(faces),
        )
        return center_x

    def _detect_face_center_x(
        self,
        video_path: str,
        time_seconds: float,
        src_w: int,
        src_h: int,
    ) -> Optional[int]:
        """Detect the horizontal centre of the primary face in a frame.

        Opens the video to read the frame. Prefer
        ``_detect_face_center_x_from_frame`` when a frame is already available.

        Args:
            video_path: Path to video file.
            time_seconds: Timestamp in seconds to sample.
            src_w: Source video width.
            src_h: Source video height.

        Returns:
            X coordinate of the face centre, or None if no face found.
        """
        cv2 = _get_cv2()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000.0)
            ret, frame = cap.read()
        finally:
            cap.release()

        if not ret or frame is None:
            return None

        return self._detect_face_center_x_from_frame(frame)
