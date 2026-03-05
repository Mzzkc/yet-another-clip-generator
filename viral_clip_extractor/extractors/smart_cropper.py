"""
Smart cropping module for vertical video reformatting.

Performs intelligent cropping to 9:16 (or custom) aspect ratio with
face-aware positioning, keeping subjects centered in the output frame.
Handles already-vertical and square videos gracefully.
"""

import logging
from typing import Optional

from viral_clip_extractor.models import PipelineConfig

logger = logging.getLogger(__name__)

# Lazy-loaded OpenCV reference
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
                "OpenCV (cv2) is required for smart cropping. "
                "Install it with: pip install opencv-python-headless"
            )
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
    ) -> dict[str, int]:
        """Calculate crop parameters for vertical reformatting.

        Detects faces in a representative frame and centres the crop
        window on the primary face. Falls back to centre crop if no
        faces are found.

        Args:
            video_path: Source video file path.
            start_time: Timestamp (seconds) to sample for face detection.
            target_ratio: Desired width/height ratio (default 9/16 = 0.5625).

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

        # Try face-aware centering
        face_center_x = self._detect_face_center_x(video_path, start_time, src_w, src_h)

        if face_center_x is not None:
            # Centre crop on face, clamped to frame bounds
            crop_x = int(face_center_x - crop_w / 2)
            crop_x = max(0, min(crop_x, src_w - crop_w))
        else:
            # Centre crop horizontally
            crop_x = (src_w - crop_w) // 2

        crop_y = 0  # Always start from top for vertical crops

        logger.info(
            "Crop params for %s: x=%d y=%d w=%d h=%d (face_aware=%s)",
            video_path,
            crop_x,
            crop_y,
            crop_w,
            crop_h,
            face_center_x is not None,
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
    ) -> str:
        """Generate an FFmpeg ``-vf`` crop filter string.

        Args:
            video_path: Source video file path.
            start_time: Timestamp for face detection sampling.
            target_ratio: Desired width/height ratio (default 9/16).

        Returns:
            FFmpeg filter string, e.g. ``crop=608:1080:236:0``.
        """
        params = self.get_crop_params(video_path, start_time, target_ratio)

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

            cap.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000.0)
            ret, frame = cap.read()
        finally:
            cap.release()

        if not ret or frame is None:
            logger.warning("Could not read frame at %.2fs", time_seconds)
            return (src_w // 2, src_h // 2)

        # Try face detection
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
        face_cascade = cv2.CascadeClassifier(cascade_path)

        if not face_cascade.empty():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            if len(faces) > 0:
                fx, fy, fw, fh = faces[0]
                return (int(fx + fw / 2), int(fy + fh / 2))

        # Fallback: centre of brightest region
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        _, _, _, max_loc = cv2.minMaxLoc(blurred)
        return (int(max_loc[0]), int(max_loc[1]))

    def _detect_face_center_x(
        self,
        video_path: str,
        time_seconds: float,
        src_w: int,
        src_h: int,
    ) -> Optional[int]:
        """Detect the horizontal centre of the primary face in a frame.

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

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
        face_cascade = cv2.CascadeClassifier(cascade_path)

        if face_cascade.empty():
            logger.warning("Haar cascade not available for face-aware cropping")
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            logger.debug("No faces detected at %.2fs in %s", time_seconds, video_path)
            return None

        # Use the largest face (highest area)
        largest = max(faces, key=lambda f: f[2] * f[3])
        fx, fy, fw, fh = largest
        center_x = int(fx + fw / 2)

        logger.debug("Face center at x=%d (frame %.2fs)", center_x, time_seconds)
        return center_x
