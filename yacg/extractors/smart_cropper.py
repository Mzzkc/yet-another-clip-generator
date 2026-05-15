"""
Smart cropping module for vertical video reformatting.

Performs intelligent cropping to 9:16 (or custom) aspect ratio with
face-aware positioning, keeping subjects centered in the output frame.
Handles already-vertical and square videos gracefully.

Two localization strategies:

  * **Face detection** (default): OpenCV Haar cascade or DNN SSD face
    detector finds faces in sampled frames; median face center is used
    as the crop horizontal anchor.  Works well for human subjects in
    photographic content.  Misfires on non-human characters (furry/
    anthro/cartoon avatars, abstract subjects) where face features are
    absent or stylized — the detector either returns no faces (falling
    back to center crop) or false positives in textures and props
    (pulling the crop window off the actual subject).

  * **VLM grounding** (opt-in via ``PipelineConfig.vlm_crop=True``):
    Sends a single sampled frame to the VLM (configured via
    ``model_name``) and asks for the horizontal subject position as a
    fraction (0.0=left edge, 1.0=right edge).  Works on any content the
    VLM can recognize as a subject — humans, anime, furry, abstract.
    Falls back to face detection (and then center crop) on any failure.
"""

import base64
import json
import logging
import re
from typing import Optional

import requests

from yacg.core.visual_analyzer import (
    _detect_faces_in_gray,
    _validate_face_detection,
    validate_spatial_consistency,
)
from yacg.models import PipelineConfig
from yacg.utils.video_utils import get_cv2 as _shared_get_cv2

logger = logging.getLogger(__name__)

# Single-frame VLM grounding call must be quick — if the model takes longer
# than this the crop falls back to face detection / center crop.  Set
# generously to accommodate first-call model load on Ollama.
_VLM_CROP_REQUEST_TIMEOUT = 60

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

        # VLM grounding takes precedence over face detection when enabled.
        # We try it first because it works on content where face detection
        # misfires (furry/anthro/cartoon/abstract subjects).  On failure we
        # fall through to face detection, then center crop.
        crop_x: Optional[int] = None
        anchor_strategy = "center"
        if self.config.vlm_crop:
            mid_time = sample_times[len(sample_times) // 2]
            vlm_x_pct = self._get_vlm_subject_x_pct(video_path, mid_time)
            if vlm_x_pct is not None:
                vlm_center_x = int(vlm_x_pct * src_w)
                crop_x = int(vlm_center_x - crop_w / 2)
                crop_x = max(0, min(crop_x, src_w - crop_w))
                anchor_strategy = "vlm"
                logger.info(
                    "VLM subject localization for %s: x_pct=%.3f → "
                    "center_x=%d → crop_x=%d",
                    video_path, vlm_x_pct, vlm_center_x, crop_x,
                )

        face_aware = False
        # Apply spatial consistency check before using face centers
        if crop_x is None and face_centers:
            face_centers = validate_spatial_consistency(face_centers, src_w)
        if crop_x is None and face_centers:
            # Use median of detected face centers (robust to outliers)
            sorted_centers = sorted(face_centers)
            median_idx = len(sorted_centers) // 2
            face_center_x = sorted_centers[median_idx]
            crop_x = int(face_center_x - crop_w / 2)
            crop_x = max(0, min(crop_x, src_w - crop_w))
            face_aware = True
            anchor_strategy = "face"
        if crop_x is None:
            # No VLM result and no faces detected — center crop
            crop_x = (src_w - crop_w) // 2

        crop_y = 0  # Always start from top for vertical crops

        logger.info(
            "Crop params for %s: x=%d y=%d w=%d h=%d "
            "(anchor=%s, face_aware=%s, detected=%d, validated=%d, samples=%d/%d)",
            video_path,
            crop_x,
            crop_y,
            crop_w,
            crop_h,
            anchor_strategy,
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

    def _get_vlm_subject_x_pct(
        self, video_path: str, time_seconds: float
    ) -> Optional[float]:
        """Ask the configured VLM where the main subject is horizontally.

        Reads one frame from the video at ``time_seconds``, encodes it as
        base64 JPEG, sends it to Ollama with a tight grounding prompt, and
        parses a single x-percentage (0.0=left, 1.0=right) from the
        response.

        Returns None on any failure (cannot read frame, Ollama error,
        unparseable response, value out of range).  Caller is responsible
        for falling back to face detection or center crop.

        Args:
            video_path: Path to source video.
            time_seconds: Timestamp to sample.

        Returns:
            Subject horizontal position as a float in [0.0, 1.0], or None.
        """
        cv2 = _get_cv2()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("VLM crop: could not open %s for frame read", video_path)
            return None

        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000.0)
            ret, frame = cap.read()
        except Exception as exc:
            logger.warning("VLM crop: frame read failed at %.2fs: %s", time_seconds, exc)
            return None
        finally:
            cap.release()

        if not ret or frame is None:
            logger.warning("VLM crop: no frame at %.2fs in %s", time_seconds, video_path)
            return None

        # Encode as JPEG (quality 85 — quality matters less than fidelity to
        # composition; the VLM doesn't need pixel-perfect detail to localize).
        ok, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            logger.warning("VLM crop: jpeg encode failed for frame at %.2fs", time_seconds)
            return None
        b64 = base64.b64encode(jpeg_buf.tobytes()).decode("ascii")

        # Reasoning prompt over a tight prompt is load-bearing here.  Smaller
        # VLMs (qwen3-vl:8b, qwen2.5-vl:7b) given a tight "output only a
        # number" prompt default to 0.5 across every frame because they
        # play it safe under output constraint.  Asking them to describe
        # the subject + reason about its position FIRST produces honest
        # coordinates even on stylized content (furry/anthro/cartoon).
        # The final line is parsed by the regex below.
        prompt = (
            "Look at this video frame. Identify the main subject — the "
            "character, person, or focal object.\n\n"
            "Step 1: Briefly say what the main subject is.\n"
            "Step 2: Where does it sit horizontally — left of center, "
            "near center, or right of center? Be specific about how far "
            "off-center.\n"
            "Step 3: On the FINAL line, output ONLY a single decimal "
            "number 0.0-1.0 for the subject's horizontal position. "
            "0.0=left edge, 0.5=exact center, 1.0=right edge."
        )
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
            "options": {
                "temperature": 0.1,  # near-deterministic; coordinate output
                "num_predict": 256,   # room for description + reasoning + number
            },
        }

        url = f"{self.config.ollama_host}/api/generate"
        try:
            resp = requests.post(url, json=payload, timeout=_VLM_CROP_REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            logger.warning("VLM crop: request failed for %s: %s", video_path, exc)
            return None

        if resp.status_code != 200:
            logger.warning(
                "VLM crop: Ollama returned status %d for %s",
                resp.status_code, video_path,
            )
            return None

        try:
            text = resp.json().get("response", "").strip()
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("VLM crop: bad JSON in Ollama response: %s", exc)
            return None

        # The reasoning prompt produces multi-line output ending with the
        # coordinate on the final line.  Parse from the LAST line, then
        # the last numeric in the full response, then any numeric.  This
        # avoids false-matching "Step 1:" or similar enumerations earlier
        # in the response as the coordinate.
        x_pct: Optional[float] = None
        coord_pattern = r"(?<![\w.])(0?\.\d+|[01](?:\.\d+)?)(?![\w.])"
        last_line = text.rstrip().splitlines()[-1] if text.strip() else ""
        last_line_matches = re.findall(coord_pattern, last_line)
        all_matches = re.findall(coord_pattern, text)
        candidates: list[str] = []
        if last_line_matches:
            candidates.append(last_line_matches[-1])
        if all_matches:
            candidates.append(all_matches[-1])
        # Prefer values containing a decimal point (more likely to be a
        # real coordinate than enumerations like "Step 1").
        for raw in candidates:
            try:
                v = float(raw)
            except ValueError:
                continue
            if 0.0 <= v <= 1.0 and "." in raw:
                x_pct = v
                break
        if x_pct is None:
            # Fall back to first plausible candidate even without a decimal.
            for raw in candidates:
                try:
                    v = float(raw)
                except ValueError:
                    continue
                if 0.0 <= v <= 1.0:
                    x_pct = v
                    break
        if x_pct is None:
            logger.warning(
                "VLM crop: no parseable x_pct in response %r for %s",
                text[:200], video_path,
            )
            return None

        return x_pct

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
