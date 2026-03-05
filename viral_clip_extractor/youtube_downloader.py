"""
YouTube video downloader for the Viral Clip Extractor.

Downloads videos via yt-dlp for local processing, handling various YouTube
URL formats and providing structured metadata about the downloaded file.
Patterns adapted from yt-transcriber/transcribe.py.
"""

import logging
import os
import re
from pathlib import Path

import yt_dlp

logger = logging.getLogger(__name__)


class YouTubeDownloader:
    """Download YouTube videos for clip extraction.

    Args:
        output_dir: Directory to store downloaded videos. Created if needed.
    """

    def __init__(self, output_dir: str = "./downloads") -> None:
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def download(self, url: str) -> dict:
        """Download a YouTube video and return metadata about the result.

        Args:
            url: A YouTube URL (youtube.com/watch, youtu.be, shorts, etc.).

        Returns:
            A dict with keys:
                video_path (str): Local path to the downloaded file.
                title (str): Video title.
                duration (float): Duration in seconds.
                channel (str): Channel name.

        Raises:
            ValueError: If the URL cannot be parsed as a YouTube video.
            RuntimeError: If the download fails.
        """
        video_id = self.extract_video_id(url)
        if not video_id:
            raise ValueError(f"Could not parse YouTube video ID from URL: {url}")

        output_template = os.path.join(self.output_dir, "%(id)s.%(ext)s")

        # Force h264 video codec — AV1/VP9 can't be decoded by OpenCV/scenedetect
        ydl_opts = {
            "format": (
                "bestvideo[vcodec^=avc1]+bestaudio[ext=m4a]/"
                "bestvideo[vcodec^=avc]+bestaudio/"
                "best[vcodec^=avc1]/"
                "best[ext=mp4]/"
                "best"
            ),
            "outtmpl": output_template,
            "merge_output_format": "mp4",
            "postprocessors": [{
                # Re-encode to h264 if source codec isn't compatible
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }],
            "quiet": True,
            "no_warnings": True,
        }

        logger.info("Downloading video %s ...", video_id)

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
        except yt_dlp.utils.DownloadError as exc:
            raise RuntimeError(f"Download failed for {url}: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Unexpected error downloading {url}: {exc}") from exc

        if info is None:
            raise RuntimeError(f"yt-dlp returned no info for {url}")

        # Resolve the actual output file path
        video_path = self._find_downloaded_file(video_id)
        if video_path is None:
            raise RuntimeError(f"Download appeared to succeed but no file found for {video_id}")

        result = {
            "video_path": video_path,
            "title": info.get("title", "Unknown"),
            "duration": float(info.get("duration", 0)),
            "channel": info.get("uploader", info.get("channel", "Unknown")),
        }

        logger.info(
            "Downloaded: %s (%s, %.0fs)",
            result["title"],
            result["channel"],
            result["duration"],
        )
        return result

    def extract_video_id(self, url: str) -> str:
        """Parse a YouTube video ID from various URL formats.

        Supports:
            - youtube.com/watch?v=ID
            - youtu.be/ID
            - youtube.com/shorts/ID
            - youtube.com/embed/ID
            - youtube.com/v/ID
            - Bare 11-character ID

        Args:
            url: A YouTube URL or bare video ID.

        Returns:
            The 11-character video ID, or an empty string if unparseable.
        """
        patterns = [
            r"(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([a-zA-Z0-9_-]{11})",
            r"^([a-zA-Z0-9_-]{11})$",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return ""

    def _find_downloaded_file(self, video_id: str) -> str | None:
        """Locate the downloaded file by video ID in the output directory."""
        for entry in os.listdir(self.output_dir):
            if entry.startswith(video_id) and not entry.endswith(".part"):
                return os.path.join(self.output_dir, entry)
        return None
