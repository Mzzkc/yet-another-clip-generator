"""
Transcript bridge for integrating yt-transcriber with the clip pipeline.

Calls yt-transcriber as a subprocess to produce transcripts, then provides
methods to extract text for time ranges and detect trigger words. Falls
back gracefully when yt-transcriber is not available.
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default ASMR trigger keywords
_DEFAULT_KEYWORDS: list[str] = [
    "tingles", "relax", "sleep", "cozy", "gentle",
    "dragon", "scales", "whisper", "magic",
]


class TranscriptBridge:
    """Bridge between yt-transcriber and the clip extraction pipeline.

    Provides transcript data for trigger word detection and semantic context.

    Args:
        yt_transcriber_path: Path to the yt-transcriber directory containing
            transcribe.py.
    """

    def __init__(
        self, yt_transcriber_path: str = "/home/emzi/Projects/yt-transcriber"
    ) -> None:
        self.yt_transcriber_path = yt_transcriber_path
        self._script_path = os.path.join(yt_transcriber_path, "transcribe.py")

    def is_available(self) -> bool:
        """Check whether yt-transcriber is available.

        Returns:
            True if the transcribe.py script exists.
        """
        return os.path.isfile(self._script_path)

    def transcribe_youtube(
        self, url: str, model: str = "small"
    ) -> Optional[dict]:
        """Transcribe a YouTube video using yt-transcriber.

        Calls transcribe.py as a subprocess and returns the parsed JSON output.

        Args:
            url: YouTube video URL.
            model: Whisper model size (tiny, base, small, medium, large-v3).

        Returns:
            Parsed transcript dict (schema_version, meta, content, processing),
            or None if transcription fails.
        """
        if not self.is_available():
            logger.warning(
                "yt-transcriber not found at %s", self.yt_transcriber_path
            )
            return None

        try:
            import tempfile
            with tempfile.TemporaryDirectory(prefix="transcript_") as tmp_dir:
                cmd = [
                    "python3", self._script_path,
                    url,
                    "--model", model,
                    "--output-dir", tmp_dir,
                ]

                logger.info("Running yt-transcriber: %s", " ".join(cmd))
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=600,
                )

                if result.returncode != 0:
                    logger.error(
                        "yt-transcriber failed (exit %d): %s",
                        result.returncode, result.stderr.strip()[:500],
                    )
                    return None

                # Find the output JSON file
                json_files = list(Path(tmp_dir).glob("transcript_*.json"))
                if not json_files:
                    logger.error("No transcript JSON found in %s", tmp_dir)
                    return None

                return self.load_transcript(str(json_files[0]))

        except subprocess.TimeoutExpired:
            logger.error("yt-transcriber timed out")
            return None
        except Exception as exc:
            logger.error("yt-transcriber error: %s", exc)
            return None

    def load_transcript(self, json_path: str) -> Optional[dict]:
        """Load a transcript from a JSON file.

        Args:
            json_path: Path to a yt-transcriber output JSON file.

        Returns:
            Parsed transcript dict, or None on failure.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info("Loaded transcript from %s", json_path)
            return data
        except FileNotFoundError:
            logger.error("Transcript file not found: %s", json_path)
            return None
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON in %s: %s", json_path, exc)
            return None
        except Exception as exc:
            logger.error("Failed to load transcript %s: %s", json_path, exc)
            return None

    def get_segment_text(
        self, transcript: dict, start: float, end: float
    ) -> str:
        """Extract transcript text for a time range.

        Walks through the transcript segments and collects text from any
        segment that overlaps the [start, end] window.

        Args:
            transcript: Parsed yt-transcriber JSON (must have content.segments).
            start: Start time in seconds.
            end: End time in seconds.

        Returns:
            Concatenated text from overlapping segments, or empty string.
        """
        try:
            segments = transcript.get("content", {}).get("segments", [])
        except (AttributeError, TypeError):
            return ""

        texts: list[str] = []
        for seg in segments:
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)

            # Check for overlap
            if seg_start < end and seg_end > start:
                text = seg.get("text", "").strip()
                if text:
                    texts.append(text)

        return " ".join(texts)

    def find_trigger_words(
        self,
        transcript: dict,
        start: float,
        end: float,
        keywords: Optional[list[str]] = None,
    ) -> list[str]:
        """Find trigger words in a transcript time range.

        Args:
            transcript: Parsed yt-transcriber JSON.
            start: Start time in seconds.
            end: End time in seconds.
            keywords: Custom keyword list. Uses ASMR defaults if None.

        Returns:
            List of matched trigger words found in the segment text.
        """
        if keywords is None:
            keywords = list(_DEFAULT_KEYWORDS)

        text = self.get_segment_text(transcript, start, end).lower()
        if not text:
            return []

        found: list[str] = []
        for keyword in keywords:
            if keyword.lower() in text:
                found.append(keyword)

        if found:
            logger.debug(
                "Trigger words in %.1f-%.1fs: %s", start, end, found
            )

        return found
