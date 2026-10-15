"""Remote Whisper transcription.

Sends a video's audio to a Whisper server with the standard
``POST /v1/audio/transcriptions`` endpoint (for example faster-whisper running
on another machine's GPU) and returns word timestamps in yacg's own shape.
Used when ``whisper_backend = "remote"``; local faster-whisper stays the default.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass

import requests

from yacg.models import WordTimestamp

logger = logging.getLogger(__name__)

# (connect, read) seconds — a long video can take minutes to transcribe.
_TIMEOUT = (10, 3600)


@dataclass
class RemoteTranscript:
    """What the server told us, reduced to what yacg uses."""

    language: str | None
    language_probability: float | None
    words: list[WordTimestamp]


def transcriptions_endpoint(url: str) -> str:
    """Resolve the endpoint from a base URL (with or without /v1) or $WHISPER_API_URL."""
    base = (url or os.environ.get("WHISPER_API_URL", "")).strip().rstrip("/")
    if not base:
        raise RuntimeError(
            "Remote Whisper backend needs a server URL: set whisper_url "
            "(--whisper-url) or the WHISPER_API_URL environment variable"
        )
    if base.endswith("/v1"):
        base = base[:-3]
    return base + "/v1/audio/transcriptions"


def _extract_audio(video_path: str, out_path: str) -> None:
    """Extract 16 kHz mono FLAC — what Whisper uses anyway, and small to upload."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", video_path,
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac", out_path],
        check=True,
    )


def transcribe(
    video_path: str,
    *,
    url: str,
    model: str,
    vad_filter: bool,
    session: requests.Session | None = None,
) -> RemoteTranscript:
    """Transcribe ``video_path`` on the remote server with word timestamps.

    Raises:
        RuntimeError: on missing URL, audio extraction failure, an unreachable
            server, or a non-200 response.
    """
    endpoint = transcriptions_endpoint(url)
    headers = {}
    api_key = os.environ.get("WHISPER_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    fields = [
        ("model", model),
        ("response_format", "verbose_json"),
        ("timestamp_granularities[]", "segment"),
        ("timestamp_granularities[]", "word"),
        ("vad_filter", "true" if vad_filter else "false"),
    ]
    http = session or requests
    with tempfile.TemporaryDirectory(prefix="yacg_whisper_") as tmp:
        audio_path = os.path.join(tmp, "audio.flac")
        try:
            _extract_audio(video_path, audio_path)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise RuntimeError(
                f"Could not extract audio for remote transcription: {video_path}"
            ) from exc
        size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        logger.info("Uploading %.1f MB of audio to remote Whisper (%s)", size_mb, model)
        try:
            with open(audio_path, "rb") as audio:
                response = http.post(
                    endpoint,
                    data=fields,
                    files={"file": ("audio.flac", audio, "audio/flac")},
                    headers=headers,
                    timeout=_TIMEOUT,
                )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Could not reach the remote Whisper server at {endpoint} ({exc}). "
                "Check that it is running, or set whisper_backend back to 'local'."
            ) from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"Remote Whisper server returned HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )
    body = response.json()
    # Word timestamps live inside segments on faster-whisper servers and at
    # the top level on some others.
    raw_words = [w for seg in body.get("segments") or [] for w in seg.get("words") or []]
    if not raw_words:
        raw_words = body.get("words") or []
    words = [
        WordTimestamp(
            word=w["word"].strip(),
            start=float(w["start"]),
            end=float(w["end"]),
            # Missing confidence stays None; subtitle filtering always keeps those.
            probability=w.get("probability"),
        )
        for w in raw_words
    ]
    return RemoteTranscript(
        language=body.get("language"),
        language_probability=body.get("language_probability"),
        words=words,
    )
