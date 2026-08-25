"""
Subtitle burning module for TikTok word-pop style subtitles.

Generates ASS (Advanced SubStation Alpha) subtitle files from pre-computed
word-level timestamps and burns them into clips using FFmpeg's libass filter.
Receives words from the full-video Whisper run — does NOT re-transcribe.
"""

import logging
import os
import platform
import struct
import subprocess
import tempfile
import threading
from functools import lru_cache

from yacg.models import SubtitleStyle, WordTimestamp
from yacg.utils.video_utils import translate_ffmpeg_error

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


@lru_cache(maxsize=32)
def _font_metrics(font_name: str) -> tuple[int, tuple[int, ...], bytes, int] | None:
    """Load enough TrueType metrics to measure the font libass will select.

    Fontconfig resolves the same family substitution used by libass on Linux
    and macOS.  The standard library has no text-metric API, so reading the
    advance widths directly keeps grouping independent of an optional imaging
    dependency.  Fonts or platforms that cannot expose a TrueType file fall
    back to a conservative estimate below.
    """
    try:
        result = subprocess.run(
            ["fc-match", "--format=%{file}", font_name],
            capture_output=True, text=True, timeout=5,
        )
        font_path = result.stdout.strip()
        if result.returncode != 0 or not font_path:
            return None
        with open(font_path, "rb") as font_file:
            data = font_file.read()

        num_tables = struct.unpack_from(">H", data, 4)[0]
        tables: dict[bytes, tuple[int, int]] = {}
        for index in range(num_tables):
            offset = 12 + index * 16
            tag, _, table_offset, length = struct.unpack_from(">4sIII", data, offset)
            tables[tag] = (table_offset, length)
        head_offset = tables[b"head"][0]
        hhea_offset = tables[b"hhea"][0]
        hmtx_offset = tables[b"hmtx"][0]
        cmap_offset = tables[b"cmap"][0]
        units_per_em = struct.unpack_from(">H", data, head_offset + 18)[0]
        number_of_metrics = struct.unpack_from(">H", data, hhea_offset + 34)[0]
        advances = tuple(
            struct.unpack_from(">H", data, hmtx_offset + index * 4)[0]
            for index in range(number_of_metrics)
        )

        cmap_tables = struct.unpack_from(">H", data, cmap_offset + 2)[0]
        cmap_subtable_offset = None
        for index in range(cmap_tables):
            record_offset = cmap_offset + 4 + index * 8
            platform_id, encoding_id, relative_offset = struct.unpack_from(
                ">HHI", data, record_offset,
            )
            candidate_offset = cmap_offset + relative_offset
            format_id = struct.unpack_from(">H", data, candidate_offset)[0]
            if format_id == 4 and (platform_id, encoding_id) in {(3, 1), (0, 3), (0, 4)}:
                cmap_subtable_offset = candidate_offset
                break
        if cmap_subtable_offset is None:
            return None
        return units_per_em, advances, data, cmap_subtable_offset
    except (FileNotFoundError, KeyError, OSError, struct.error, subprocess.TimeoutExpired):
        return None


def _glyph_id_for_codepoint(font_data: bytes, cmap_offset: int, codepoint: int) -> int:
    """Return a glyph id from a TrueType format-4 Unicode cmap."""
    seg_count = struct.unpack_from(">H", font_data, cmap_offset + 6)[0] // 2
    end_codes_offset = cmap_offset + 14
    start_codes_offset = end_codes_offset + seg_count * 2 + 2
    id_delta_offset = start_codes_offset + seg_count * 2
    id_range_offset = id_delta_offset + seg_count * 2
    for index in range(seg_count):
        end_code = struct.unpack_from(">H", font_data, end_codes_offset + index * 2)[0]
        if codepoint > end_code:
            continue
        start_code = struct.unpack_from(">H", font_data, start_codes_offset + index * 2)[0]
        if codepoint < start_code:
            return 0
        delta = struct.unpack_from(">h", font_data, id_delta_offset + index * 2)[0]
        range_offset_position = id_range_offset + index * 2
        range_offset = struct.unpack_from(">H", font_data, range_offset_position)[0]
        if range_offset == 0:
            return (codepoint + delta) & 0xFFFF
        glyph_position = range_offset_position + range_offset + 2 * (codepoint - start_code)
        glyph_id = struct.unpack_from(">H", font_data, glyph_position)[0]
        return (glyph_id + delta) & 0xFFFF if glyph_id else 0
    return 0


def _rendered_text_width(text: str, font_name: str, font_size: int, outline_width: float) -> float:
    """Estimate single-line ASS width from the selected font's glyph advances."""
    metrics = _font_metrics(font_name)
    if metrics is not None:
        units_per_em, advances, font_data, cmap_offset = metrics
        advance_units = 0
        for character in text:
            glyph_id = _glyph_id_for_codepoint(font_data, cmap_offset, ord(character))
            advance_units += advances[min(glyph_id, len(advances) - 1)]
        return advance_units * font_size / units_per_em + 2 * outline_width

    # Fontconfig is unavailable on Windows and minimal containers.  Preserve
    # a conservative bound there rather than allowing a phrase to overflow.
    narrow = set("ilIjtfr.,'!|:;` ")
    wide = set("MW@#%&")
    em_width = sum(0.32 if char in narrow else 0.92 if char in wide else 0.58 for char in text)
    return em_width * font_size + 2 * outline_width


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
    # MAX_GROUP_DURATION: Maximum display time before forcing a new group.
    MAX_GROUP_DURATION = 2.0

    def _group_words(
        self,
        words: list[WordTimestamp],
        min_words_per_group: int,
        max_words_per_group: int,
        available_width: int,
        font_name: str,
        font_size: int,
        outline_width: float,
    ) -> list[tuple[float, float, list[WordTimestamp]]]:
        """Group words into readable subtitle phrases.

        Returns list of (start_time, end_time, group_words) tuples.
        Times should already be clip-relative (shifted by clip_start).
        Returning the WORD list (not concatenated text) lets the caller
        emit per-word karaoke timing tags when needed.
        """
        groups: list[tuple[float, float, list[WordTimestamp]]] = []
        # The reason after each group determines whether its short tail may
        # borrow words from the preceding group.  Only a soft max-word split
        # is movable; punctuation, pauses, duration, and width are real
        # spoken/rendered boundaries.
        split_reasons: list[str] = []
        i = 0
        while i < len(words):
            group_words = [words[i]]
            group_start = words[i].start
            split_reason = "end"
            while i + len(group_words) < len(words):
                candidate = words[i + len(group_words)]
                previous = group_words[-1]
                if previous.word.strip().endswith((".", "!", "?", "…")):
                    split_reason = "punctuation"
                    break
                if candidate.start - previous.end > self.WORD_GROUP_GAP:
                    split_reason = "gap"
                    break
                if candidate.end - group_start > self.MAX_GROUP_DURATION:
                    split_reason = "duration"
                    break
                candidate_words = [*group_words, candidate]
                candidate_text = " ".join(word.word.strip() for word in candidate_words)
                if _rendered_text_width(
                    candidate_text, font_name, font_size, outline_width,
                ) > available_width:
                    split_reason = "width"
                    break
                if len(candidate_words) > max_words_per_group:
                    split_reason = "max_words"
                    break
                group_words = candidate_words

            groups.append((group_words[0].start, group_words[-1].end, group_words))
            split_reasons.append(split_reason)
            i += len(group_words)

        # Greedy groups naturally end at the maximum.  Shift words into a
        # final short group where the split was arbitrary, keeping target
        # 3-5 phrases for continuous speech without crossing a hard boundary.
        for index in range(len(groups) - 1, 0, -1):
            start, end, group_words = groups[index]
            previous_start, previous_end, previous_words = groups[index - 1]
            if len(group_words) >= min_words_per_group or split_reasons[index - 1] != "max_words":
                continue
            needed = min_words_per_group - len(group_words)
            if len(previous_words) - needed < min_words_per_group:
                continue
            moved_words = previous_words[-needed:]
            rebalanced_words = [*moved_words, *group_words]
            rebalanced_text = " ".join(word.word.strip() for word in rebalanced_words)
            if (
                rebalanced_words[-1].end - rebalanced_words[0].start > self.MAX_GROUP_DURATION
                or _rendered_text_width(
                    rebalanced_text, font_name, font_size, outline_width,
                ) > available_width
            ):
                continue
            remaining_words = previous_words[:-needed]
            groups[index - 1] = (
                remaining_words[0].start, remaining_words[-1].end, remaining_words,
            )
            groups[index] = (
                rebalanced_words[0].start, rebalanced_words[-1].end, rebalanced_words,
            )

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

        outline_color = style.outline_color
        outline_width = style.outline_width
        shadow = style.shadow
        border_style = style.border_style if style.border_style in (1, 3) else 1

        # ASS karaoke color semantics (V4+ libass):
        #   * Text not yet sung displays in SecondaryColour.
        #   * As \k<duration> fires per word, that word switches to PrimaryColour.
        # For "active word green / resting violet" we map:
        #   PrimaryColour   = karaoke_active_color (active / already-sung)
        #   SecondaryColour = primary_color       (resting / not-yet-sung)
        # When karaoke is OFF, the existing PrimaryColour-only behavior is
        # preserved by setting both to primary_color (no transition fires).
        if style.karaoke:
            ass_primary = style.karaoke_active_color
            ass_secondary = style.primary_color
        else:
            ass_primary = style.primary_color
            ass_secondary = style.primary_color

        # Group words using the final font size and margins.  This prevents a
        # phrase that was valid before style resolution from overflowing the
        # actual single-line libass rendering area.
        groups = self._group_words(
            words,
            min_words_per_group=style.min_words_per_group,
            max_words_per_group=style.max_words_per_group,
            available_width=frame_width - margin_l - margin_r,
            font_name=font_name,
            font_size=font_size,
            outline_width=outline_width,
        )

        dialogue_lines: list[str] = []
        for start, end, group_words in groups:
            start_time = _format_ass_time(start)
            end_time = _format_ass_time(end)
            if style.karaoke:
                # Per-word karaoke timing.  ASS \k expects centiseconds.
                # Use a minimum of 1cs so even very short words register.
                parts: list[str] = []
                for index, w in enumerate(group_words):
                    next_onset = (
                        group_words[index + 1].start
                        if index + 1 < len(group_words)
                        else w.end
                    )
                    cs = max(1, int(round((next_onset - w.start) * 100)))
                    parts.append(f"{{\\k{cs}}}{w.word.strip()}")
                text = " ".join(parts)
            else:
                text = " ".join(w.word.strip() for w in group_words)
            dialogue_lines.append(
                f"Dialogue: 0,{start_time},{end_time},WordPop,,0,0,0,,{text}"
            )

        # WrapStyle: 2 (none) is preferred over 0 (smart wrap) for short-
        # form vertical clips.  Smart-wrap silently breaks dialogue into
        # 2 lines when text exceeds the available width — common with long
        # ASMR words ("andragor") on 1080-1215 px wide crops at >5% font.
        # Style 2 lets text overflow horizontally, which composers can
        # control via font_size_pct + margin_h_pct + max_words_per_group
        # rather than discovering wrap surprises in production output.
        ass_content = (
            f"[Script Info]\n"
            f"; Generated by YACG\n"
            f"ScriptType: v4.00+\n"
            f"PlayResX: {frame_width}\n"
            f"PlayResY: {frame_height}\n"
            f"WrapStyle: 2\n"
            f"\n"
            f"[V4+ Styles]\n"
            f"Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour,"
            f" OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut,"
            f" ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow,"
            f" Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: WordPop,{font_name},{font_size},{ass_primary},"
            f"{ass_secondary},{outline_color},&H80000000,-1,0,0,0,100,100,0,0,"
            f"{border_style},{outline_width},{shadow},2,"
            f"{margin_l},{margin_r},{margin_v},1\n"
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
