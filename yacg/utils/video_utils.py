"""
Video utility functions for the YACG.

Wraps FFmpeg/FFprobe operations for metadata extraction, audio extraction,
segment cutting, and frame capture. Follows patterns from the existing
VideoProcessor in caption_generator.py.
"""

import json
import logging
import os
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded OpenCV reference — shared across modules
_cv2 = None


def get_cv2():  # type: ignore[no-untyped-def]
    """Lazy-import cv2 so modules can be imported even without OpenCV."""
    global _cv2
    if _cv2 is None:
        try:
            import cv2

            _cv2 = cv2
        except ImportError:
            raise ImportError(
                "OpenCV (cv2) is required. "
                "Install it with: pip install opencv-python-headless"
            )
    return _cv2


def translate_ffmpeg_error(stderr: str) -> str:
    """Translate raw FFmpeg stderr into a human-readable message."""
    s = stderr.lower()
    if "no such file" in s:
        return "Input file not found. Check that the video path is correct."
    if "invalid data" in s or "invalid argument" in s:
        return "The video file appears to be corrupt or in an unsupported format."
    if "permission denied" in s:
        return "Permission denied — check file/folder permissions."
    if "no space" in s:
        return "Disk is full — free up space and try again."
    if "does not contain" in s or ("stream" in s and "not found" in s):
        return "The video file is missing a required stream (video or audio)."
    return stderr.strip()[:300]


def extract_metadata(video_path: str) -> dict:
    """Extract metadata from a video file using FFprobe.

    Args:
        video_path: Path to the video file.

    Returns:
        A dict with keys: filename, filepath, duration, width, height,
        file_size, fps, codec, audio_codec, sample_rate, channels.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If FFprobe fails to parse the file.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        raise RuntimeError("ffprobe not found — install FFmpeg")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffprobe timed out on {video_path}")

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed (exit {result.returncode}): {result.stderr.strip()}")

    data = json.loads(result.stdout)

    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
        None,
    )
    audio_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "audio"),
        None,
    )
    format_info = data.get("format", {})

    # Parse frame rate safely (e.g. "30/1" -> 30.0)
    fps = 0.0
    if video_stream:
        r_frame_rate = video_stream.get("r_frame_rate", "0/1")
        parts = r_frame_rate.split("/")
        if len(parts) == 2 and int(parts[1]) != 0:
            fps = int(parts[0]) / int(parts[1])

    meta = {
        "filename": path.name,
        "filepath": str(path.resolve()),
        "duration": float(format_info.get("duration", 0)),
        "width": int(video_stream.get("width", 0)) if video_stream else 0,
        "height": int(video_stream.get("height", 0)) if video_stream else 0,
        "file_size": int(format_info.get("size", 0)),
        "fps": fps,
        "codec": video_stream.get("codec_name", "") if video_stream else "",
        "audio_codec": audio_stream.get("codec_name", "") if audio_stream else "",
        "sample_rate": int(audio_stream.get("sample_rate", 0)) if audio_stream else 0,
        "channels": int(audio_stream.get("channels", 0)) if audio_stream else 0,
    }

    logger.debug("Extracted metadata for %s: %s", path.name, meta)
    return meta


def extract_audio(
    video_path: str,
    output_path: str,
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> str:
    """Extract audio from a video file as WAV.

    Args:
        video_path: Source video file.
        output_path: Destination WAV file path.
        start: Optional start time in seconds.
        end: Optional end time in seconds.

    Returns:
        The output file path.

    Raises:
        RuntimeError: If FFmpeg fails.
    """
    cmd = ["ffmpeg", "-y", "-i", str(video_path)]

    if start is not None:
        cmd.extend(["-ss", str(start)])
    if end is not None:
        if start is not None:
            cmd.extend(["-t", str(end - start)])
        else:
            cmd.extend(["-t", str(end)])

    cmd.extend([
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "22050",
        "-ac", "1",
        str(output_path),
    ])

    duration_str = ""
    segment_dur = None
    if start is not None and end is not None:
        segment_dur = end - start
        duration_str = f" ({segment_dur:.1f}s segment)"
    elif end is not None:
        segment_dur = end
        duration_str = f" ({segment_dur:.1f}s)"
    logger.info("Extracting audio%s...", duration_str)
    if segment_dur is not None and segment_dur > 60:
        logger.warning(
            "Long audio extraction (%.0fs) — this may take a while", segment_dur,
        )
    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {result.stderr.strip()}")

    logger.info("Extracted audio to %s", output_path)
    return output_path


def extract_segment(
    video_path: str,
    start: float,
    end: float,
    output_path: str,
) -> str:
    """Extract a video segment with re-encoding.

    Uses FFmpeg to cut a segment from *start* to *end* seconds and
    re-encode to H.264/AAC for maximum compatibility.

    Args:
        video_path: Source video file.
        start: Start time in seconds.
        end: End time in seconds.
        output_path: Destination file path.

    Returns:
        The output file path.

    Raises:
        RuntimeError: If FFmpeg fails.
    """
    duration = end - start
    if duration <= 0:
        raise ValueError(f"Invalid segment: start={start}, end={end}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path),
    ]

    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        raise RuntimeError(f"Segment extraction failed: {result.stderr.strip()}")

    logger.info("Extracted segment %.1f-%.1fs to %s", start, end, output_path)
    return output_path


def get_frame_at_time(video_path: str, time_seconds: float) -> Optional[np.ndarray]:
    """Capture a single frame from a video at the given timestamp.

    Args:
        video_path: Path to video file.
        time_seconds: Timestamp in seconds.

    Returns:
        A numpy array (BGR, HxWx3) or ``None`` on failure.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Could not open video: %s", video_path)
        return None

    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000.0)
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame at %.2fs in %s", time_seconds, video_path)
            return None
        return frame
    finally:
        cap.release()


def ensure_compatible_video(video_path: str) -> str:
    """Ensure the video uses a codec OpenCV/scenedetect can read (h264).

    If the video uses an incompatible codec (AV1, VP9, etc.), transcode it
    to h264 via FFmpeg. Returns the original path if already compatible,
    or the path to a transcoded copy.

    The caller is responsible for cleaning up any transcoded file.

    Args:
        video_path: Path to the video file.

    Returns:
        Path to a compatible video file (may be same as input).
    """
    meta = extract_metadata(video_path)
    codec = meta.get("codec", "").lower()

    compatible_codecs = {"h264", "h265", "hevc", "mpeg4", "mpeg2video", "mjpeg"}
    if codec in compatible_codecs:
        logger.debug("Video codec %s is compatible", codec)
        return video_path

    # Also test if OpenCV can actually open it
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        opened = cap.isOpened()
        if opened:
            ret, _ = cap.read()
            cap.release()
            if ret:
                logger.debug("OpenCV can read the video despite codec %s", codec)
                return video_path
        else:
            cap.release()
    except Exception:
        pass

    # Transcode to h264
    logger.info(
        "Video codec '%s' is not compatible with OpenCV — transcoding to h264...",
        codec,
    )

    stem = Path(video_path).stem
    transcoded_path = str(Path(video_path).parent / f"{stem}_h264.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(transcoded_path),
    ]

    logger.info("Transcoding: %s -> %s", video_path, transcoded_path)

    # Use Popen for progress logging instead of subprocess.run with a
    # 1-hour silent timeout.  FFmpeg writes progress to stderr; we log
    # periodic updates so the user knows the transcode is still running.
    import re as _re
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    stderr_lines: list[str] = []
    last_log_time = time.time()
    try:
        assert proc.stderr is not None  # for type checker
        for line in proc.stderr:
            stderr_lines.append(line)
            # FFmpeg progress lines look like: "frame= 1234 fps=..."
            # or "time=00:01:23.45"
            now = time.time()
            if now - last_log_time >= 10:  # log every 10 seconds
                time_match = _re.search(r"time=(\S+)", line)
                if time_match:
                    logger.info("Transcoding progress: %s", time_match.group(1))
                else:
                    logger.info("Transcoding in progress...")
                last_log_time = now
        proc.wait(timeout=3600)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError(
            f"Transcoding timed out after 1 hour for {video_path}"
        )

    if proc.returncode != 0:
        stderr_text = "".join(stderr_lines).strip()
        raise RuntimeError(
            f"Transcoding failed for {video_path}: {stderr_text[-500:]}"
        )

    logger.info("Transcoding complete: %s", transcoded_path)
    return transcoded_path


@contextmanager
def temp_audio_file(suffix: str = ".wav") -> Generator[str, None, None]:
    """Context manager that yields a temporary file path and cleans up on exit.

    Args:
        suffix: File extension for the temporary file.

    Yields:
        Path to the temporary file.
    """
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        yield path
    finally:
        if os.path.exists(path):
            os.unlink(path)
            logger.debug("Cleaned up temp file: %s", path)


