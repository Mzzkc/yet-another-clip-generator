"""
Subtitle burning module for TikTok word-pop style subtitles.

Generates ASS (Advanced SubStation Alpha) subtitle files from pre-computed
word-level timestamps and burns them into clips using FFmpeg's libass filter.
Receives words from the full-video Whisper run — does NOT re-transcribe.
"""

import logging
import os
import platform
import subprocess
import tempfile
import threading

from viral_clip_extractor.models import SubtitleStyle, WordTimestamp
from viral_clip_extractor.utils.video_utils import translate_ffmpeg_error

logger = logging.getLogger(__name__)

# Font fallback chain — first match wins. Covers Linux, macOS, Windows,
# and containers. Cached after first probe.
_FONT_CANDIDATES = [
    "Liberation Sans",  # Linux (liberation-fonts)
    "Arial",            # Windows, macOS
    "Helvetica Neue",   # macOS
    "Helvetica",        # macOS fallback
    "DejaVu Sans",      # Linux fallback (dejavu-fonts)
    "Noto Sans",        # Google Noto (common on Linux/containers)
    "sans-serif",       # Ultimate fallback — libass resolves via fontconfig
]
_cached_font: str | None = None
_font_lock = threading.Lock()


def _find_system_font() -> str:
    """Probe for an available sans-serif font on this system.

    Uses ``fc-list`` on Linux/macOS and a platform check on Windows.
    Result is cached for the process lifetime. Thread-safe via lock.
    """
    global _cached_font
    if _cached_font is not None:
        return _cached_font

    with _font_lock:
        # Double-check after acquiring the lock
        if _cached_font is not None:
            return _cached_font

        # On Windows, Arial is always available
        if platform.system() == "Windows":
            _cached_font = "Arial"
            return _cached_font

        # Try fc-list (fontconfig) on Linux/macOS
        try:
            result = subprocess.run(
                ["fc-list", "--format", "%{family}\n"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                # fc-list may return multi-family entries like
                # "Liberation Sans,Liberation Sans Condensed".
                # Split on comma to match individual family names.
                available: set[str] = set()
                for line in result.stdout.strip().split("\n"):
                    for family in line.split(","):
                        available.add(family.strip())
                for candidate in _FONT_CANDIDATES:
                    if candidate in available:
                        _cached_font = candidate
                        logger.debug("Selected subtitle font: %s", _cached_font)
                        return _cached_font
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Fallback
        _cached_font = "sans-serif"
        logger.warning(
            "No preferred font found — using 'sans-serif' (subtitle "
            "metrics may be imprecise)"
        )
        return _cached_font


def _format_ass_time(seconds: float) -> str:
    """Convert seconds to ASS time format h:mm:ss.cc (centiseconds)."""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


class SubtitleBurner:
    """Generate and burn TikTok-style word-pop subtitles into video clips.

    Uses ASS (Advanced SubStation Alpha) subtitle format with FFmpeg's libass
    filter for hardburn. Font selection uses platform-aware auto-detection.
    """

    # Word grouping constants — named for clarity and tuning.
    # WORD_GROUP_GAP: Pause (seconds) that triggers a new subtitle group.
    WORD_GROUP_GAP = 0.2
    # SENTENCE_BREAK_GAP: Minimum display time for a subtitle group.
    SENTENCE_BREAK_GAP = 0.3
    # MAX_GROUP_DURATION: Maximum display time before forcing a new group.
    MAX_GROUP_DURATION = 2.0

    def _group_words(
        self, words: list[WordTimestamp],
    ) -> list[tuple[float, float, str]]:
        """Group words into 1-3 word subtitle phrases.

        Returns list of (start_time, end_time, text) tuples.
        Times should already be clip-relative (shifted by clip_start).
        """
        groups: list[tuple[float, float, str]] = []
        i = 0
        while i < len(words):
            group_words = [words[i]]
            group_start = words[i].start

            # Try to add up to 2 more words
            for j in range(1, 3):
                next_idx = i + j
                if next_idx >= len(words):
                    break
                # Check for natural pause before this word
                gap = words[next_idx].start - words[next_idx - 1].end
                if gap > self.WORD_GROUP_GAP:
                    break
                group_words.append(words[next_idx])

            group_end = group_words[-1].end
            text = " ".join(w.word.strip() for w in group_words)

            # Enforce minimum display time
            if group_end - group_start < self.SENTENCE_BREAK_GAP:
                group_end = group_start + self.SENTENCE_BREAK_GAP

            # Cap maximum display time
            if group_end - group_start > self.MAX_GROUP_DURATION:
                group_end = group_start + self.MAX_GROUP_DURATION

            groups.append((group_start, group_end, text))
            i += len(group_words)

        return groups

    def get_video_dimensions(self, video_path: str) -> tuple[int, int]:
        """Get video width and height via ffprobe.

        Args:
            video_path: Path to the video file.

        Returns:
            Tuple of (width, height).

        Raises:
            RuntimeError: If ffprobe fails or returns unexpected output.
        """
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x",
            str(video_path),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffprobe failed on {video_path}: {result.stderr[:200]}"
            )
        parts = result.stdout.strip().split("x")
        if len(parts) != 2:
            raise RuntimeError(
                f"Unexpected ffprobe output: {result.stdout.strip()}"
            )
        return int(parts[0]), int(parts[1])

    def generate_ass(
        self,
        words: list[WordTimestamp],
        frame_width: int,
        frame_height: int,
        style: SubtitleStyle | None = None,
    ) -> str:
        """Generate ASS subtitle content with TikTok word-pop styling.

        Words are pre-sliced and time-shifted for this clip.
        Groups 1-3 words at a time for display.

        Args:
            words: Word timestamps already filtered and time-shifted
                for this clip (time 0.0 = start of clip).
            frame_width: Video width in pixels.
            frame_height: Video height in pixels.
            style: Optional subtitle styling overrides. Uses sensible
                defaults when ``None`` or when individual fields are
                left at their defaults.

        Returns:
            Complete ASS file content as a string.

        Raises:
            RuntimeError: If words is empty.
        """
        if not words:
            raise RuntimeError(
                "No words provided — cannot generate subtitles"
            )

        if style is None:
            style = SubtitleStyle()

        # Calculate styling parameters scaled to frame dimensions
        font_name = style.font_name or _find_system_font()
        font_size = int(frame_height * style.font_size_pct)
        margin_l = int(frame_width * style.margin_h_pct)
        margin_r = int(frame_width * style.margin_h_pct)
        margin_v = int(frame_height * style.margin_v_pct)

        primary_color = style.primary_color
        outline_color = style.outline_color
        outline_width = style.outline_width
        shadow = style.shadow

        # Group words into 1-3 word phrases
        groups = self._group_words(words)

        # Build dialogue lines
        dialogue_lines: list[str] = []
        for start, end, text in groups:
            start_time = _format_ass_time(start)
            end_time = _format_ass_time(end)
            dialogue_lines.append(
                f"Dialogue: 0,{start_time},{end_time},WordPop,,0,0,0,,{text}"
            )

        ass_content = (
            f"[Script Info]\n"
            f"; Generated by Viral Clip Extractor\n"
            f"ScriptType: v4.00+\n"
            f"PlayResX: {frame_width}\n"
            f"PlayResY: {frame_height}\n"
            f"WrapStyle: 0\n"
            f"\n"
            f"[V4+ Styles]\n"
            f"Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour,"
            f" OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut,"
            f" ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow,"
            f" Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: WordPop,{font_name},{font_size},{primary_color},"
            f"&H000000FF,{outline_color},&H80000000,-1,0,0,0,100,100,0,0,1,"
            f"{outline_width},{shadow},2,{margin_l},{margin_r},{margin_v},1\n"
            f"\n"
            f"[Events]\n"
            f"Format: Layer, Start, End, Style, Name, MarginL, MarginR,"
            f" MarginV, Effect, Text\n"
        )
        ass_content += "\n".join(dialogue_lines) + "\n"

        return ass_content

    def burn_subtitles(self, clip_path: str, ass_content: str) -> str:
        """Write ASS to temp file, run FFmpeg hardburn.

        Args:
            clip_path: Path to the clip video file.
            ass_content: Complete ASS subtitle content.

        Returns:
            Path to subtitled clip (same as input — in-place replacement).

        Raises:
            RuntimeError: On FFmpeg failure.
        """
        ass_temp_path = None
        temp_output = clip_path + ".subtitled.mp4"
        try:
            # Write ASS content to temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".ass", delete=False, encoding="utf-8",
            ) as f:
                f.write(ass_content)
                ass_temp_path = f.name

            cmd = [
                "ffmpeg", "-y",
                "-i", clip_path,
                "-vf", f"ass={ass_temp_path}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "copy",
                "-movflags", "+faststart",
                temp_output,
            ]

            logger.info("Burning subtitles into %s", clip_path)
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
            )

            if result.returncode != 0:
                human_msg = translate_ffmpeg_error(result.stderr)
                raise RuntimeError(
                    f"Subtitle burn failed: {human_msg}"
                )

            # Atomic replace on same filesystem; fall back to
            # copy+delete for cross-filesystem scenarios.
            try:
                os.replace(temp_output, clip_path)
            except OSError:
                import shutil
                shutil.copy2(temp_output, clip_path)
                os.unlink(temp_output)
            logger.info("Subtitles burned into %s", clip_path)
            return clip_path

        finally:
            # Clean up temp files
            if ass_temp_path and os.path.exists(ass_temp_path):
                os.unlink(ass_temp_path)
            if os.path.exists(temp_output):
                os.unlink(temp_output)

    def process_clip(
        self,
        clip_path: str,
        words: list[WordTimestamp],
        frame_width: int,
        frame_height: int,
        style: SubtitleStyle | None = None,
    ) -> str:
        """Full pipeline: generate ASS then burn into clip.

        The caller is responsible for pre-slicing words to the clip's
        time range and shifting timestamps so time 0.0 = clip start.
        Frame dimensions are needed for scaling subtitle styling.

        Args:
            clip_path: Path to the extracted clip file.
            words: Word timestamps already filtered and time-shifted
                for this clip (time 0.0 = start of clip).
            frame_width: Video width in pixels.
            frame_height: Video height in pixels.
            style: Optional subtitle styling overrides.

        Returns:
            Path to the subtitled clip (same as input — in-place).

        Raises:
            RuntimeError: On any failure. No un-subtitled clips produced.
        """
        ass_content = self.generate_ass(words, frame_width, frame_height, style=style)
        return self.burn_subtitles(clip_path, ass_content)
