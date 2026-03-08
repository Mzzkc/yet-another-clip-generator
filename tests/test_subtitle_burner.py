"""
Tests for the SubtitleBurner module.

Covers ASS generation, word grouping, burn_subtitles (mock FFmpeg),
and error handling.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from viral_clip_extractor.models import WordTimestamp
from viral_clip_extractor.subtitle_burner import SubtitleBurner, _format_ass_time


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def burner() -> SubtitleBurner:
    return SubtitleBurner()


@pytest.fixture
def sample_words() -> list[WordTimestamp]:
    """Word timestamps simulating 5 words (clip-relative timing)."""
    return [
        WordTimestamp(word="Never", start=0.0, end=0.3),
        WordTimestamp(word="gonna", start=0.35, end=0.6),
        WordTimestamp(word="give", start=0.65, end=0.9),
        WordTimestamp(word="you", start=0.95, end=1.1),
        WordTimestamp(word="up", start=1.8, end=2.0),
    ]


# ---------------------------------------------------------------------------
# ASS generation tests
# ---------------------------------------------------------------------------

class TestGenerateAss:
    """Tests for SubtitleBurner.generate_ass."""

    def test_subtitle_burner_generates_valid_ass(self, burner, sample_words):
        """generate_ass produces valid ASS with all required sections."""
        ass = burner.generate_ass(sample_words, 1080, 1920)

        assert "[Script Info]" in ass
        assert "[V4+ Styles]" in ass
        assert "[Events]" in ass
        assert "Dialogue:" in ass
        assert "PlayResX: 1080" in ass
        assert "PlayResY: 1920" in ass
        assert "WordPop" in ass
        # Font is dynamically detected — verify some sans-serif font is present
        from viral_clip_extractor.subtitle_burner import _find_system_font
        assert _find_system_font() in ass

    def test_ass_styling_1080x1920(self, burner, sample_words):
        """ASS styling is correct for 1080x1920."""
        ass = burner.generate_ass(sample_words, 1080, 1920)

        # Font size: int(1920 * 0.055) = 105
        font_size = int(1920 * 0.055)
        assert f",{font_size}," in ass

        # Margins: int(1080 * 0.15) = 162
        margin = int(1080 * 0.15)
        assert f",{margin},{margin}," in ass

    def test_ass_styling_720x1280(self, burner, sample_words):
        """ASS styling scales correctly for 720x1280."""
        ass = burner.generate_ass(sample_words, 720, 1280)

        font_size = int(1280 * 0.055)
        assert f",{font_size}," in ass

        margin = int(720 * 0.15)
        assert f",{margin},{margin}," in ass

    def test_ass_word_grouping(self, burner, sample_words):
        """Words are grouped into 1-3 word phrases."""
        ass = burner.generate_ass(sample_words, 1080, 1920)

        # Should have "Never gonna give" as one group (gaps < 200ms)
        assert "Never gonna give" in ass
        # "you" should be separate (gap to "up" is >200ms)
        # "up" should be separate
        dialogue_lines = [l for l in ass.split("\n") if l.startswith("Dialogue:")]
        assert len(dialogue_lines) == 3  # 3 groups

    def test_subtitle_burner_raises_on_empty_words(self, burner):
        """generate_ass raises RuntimeError on empty word list."""
        with pytest.raises(RuntimeError, match="No words provided"):
            burner.generate_ass([], 1080, 1920)


# ---------------------------------------------------------------------------
# ASS time format tests
# ---------------------------------------------------------------------------

class TestFormatAssTime:
    """Tests for _format_ass_time helper."""

    def test_zero(self):
        assert _format_ass_time(0.0) == "0:00:00.00"

    def test_normal(self):
        result = _format_ass_time(3723.46)
        assert result == "1:02:03.46"

    def test_negative_clamped(self):
        assert _format_ass_time(-1.0) == "0:00:00.00"

    def test_small_value(self):
        result = _format_ass_time(1.5)
        assert result == "0:00:01.50"


# ---------------------------------------------------------------------------
# burn_subtitles tests
# ---------------------------------------------------------------------------

class TestBurnSubtitles:
    """Tests for SubtitleBurner.burn_subtitles (mock FFmpeg)."""

    @patch("viral_clip_extractor.subtitle_burner.subprocess.run")
    def test_subtitle_burner_burns_subtitles(self, mock_run, burner, tmp_path):
        """burn_subtitles calls FFmpeg and replaces the clip in place."""
        clip_path = str(tmp_path / "clip.mp4")
        # Create a fake clip file
        with open(clip_path, "wb") as f:
            f.write(b"\x00" * 1000)

        temp_output = clip_path + ".subtitled.mp4"

        def side_effect(*args, **kwargs):
            # FFmpeg creates the output file
            with open(temp_output, "wb") as f:
                f.write(b"\x00" * 2000)
            return MagicMock(returncode=0, stderr="")

        mock_run.side_effect = side_effect

        ass_content = "[Script Info]\ntest content"
        result = burner.burn_subtitles(clip_path, ass_content)

        assert result == clip_path
        assert mock_run.called
        # Temp file should be cleaned up
        assert not os.path.exists(temp_output)

    @patch("viral_clip_extractor.subtitle_burner.subprocess.run")
    def test_burn_subtitles_raises_on_ffmpeg_failure(self, mock_run, burner, tmp_path):
        """burn_subtitles raises RuntimeError on FFmpeg failure."""
        clip_path = str(tmp_path / "clip.mp4")
        with open(clip_path, "wb") as f:
            f.write(b"\x00" * 1000)

        mock_run.return_value = MagicMock(returncode=1, stderr="encoding error")

        with pytest.raises(RuntimeError, match="Subtitle burn failed"):
            burner.burn_subtitles(clip_path, "[Script Info]\ntest")


# ---------------------------------------------------------------------------
# process_clip tests
# ---------------------------------------------------------------------------

class TestProcessClip:
    """Tests for SubtitleBurner.process_clip."""

    def test_process_clip_calls_both(self, burner, sample_words):
        """process_clip calls generate_ass then burn_subtitles."""
        with patch.object(burner, "generate_ass", return_value="ass content") as mock_gen:
            with patch.object(burner, "burn_subtitles", return_value="/path/clip.mp4") as mock_burn:
                result = burner.process_clip("/path/clip.mp4", sample_words, 1080, 1920)

                mock_gen.assert_called_once_with(sample_words, 1080, 1920, style=None)
                mock_burn.assert_called_once_with("/path/clip.mp4", "ass content")
                assert result == "/path/clip.mp4"
