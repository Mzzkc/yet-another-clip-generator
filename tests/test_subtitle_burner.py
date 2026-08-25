"""
Tests for the SubtitleBurner module.

Covers ASS generation, word grouping, burn_subtitles (mock FFmpeg),
and error handling.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from yacg.models import SubtitleStyle, WordTimestamp
from yacg.subtitle_burner import SubtitleBurner, _format_ass_time


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
        from yacg.subtitle_burner import _find_system_font
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

    def test_ass_word_grouping_defaults_to_two_words(self, burner, sample_words):
        """The vertical-video default caps subtitle groups at two words."""
        ass = burner.generate_ass(sample_words, 1080, 1920)

        assert "Never gonna" in ass
        assert "give you" in ass
        dialogue_lines = [l for l in ass.split("\n") if l.startswith("Dialogue:")]
        assert len(dialogue_lines) == 3

    def test_ass_word_grouping_allows_three_words(self, burner, sample_words):
        """Callers can retain the original three-word grouping explicitly."""
        style = SubtitleStyle(max_words_per_group=3)
        ass = burner.generate_ass(sample_words, 1920, 1080, style=style)

        assert "Never gonna give" in ass
        dialogue_lines = [l for l in ass.split("\n") if l.startswith("Dialogue:")]
        assert len(dialogue_lines) == 3

    def test_subtitle_burner_raises_on_empty_words(self, burner):
        """generate_ass raises RuntimeError on empty word list."""
        with pytest.raises(RuntimeError, match="No words provided"):
            burner.generate_ass([], 1080, 1920)


class TestAdaptiveCaptionGrouping:
    """Adaptive phrase grouping is visible in the rendered ASS dialogue."""

    @staticmethod
    def _words(*entries: tuple[str, float, float]) -> list[WordTimestamp]:
        return [WordTimestamp(word=word, start=start, end=end) for word, start, end in entries]

    @staticmethod
    def _dialogue_texts(ass: str) -> list[str]:
        return [line.rsplit(",", 1)[1] for line in ass.splitlines() if line.startswith("Dialogue:")]

    def test_adaptive_groups_rebalance_a_continuous_tail_to_three_words(self, burner):
        """Seven continuous words use 4+3 rather than leaving a two-word tail."""
        words = self._words(
            ("one", 0.00, 0.10), ("two", 0.12, 0.22), ("three", 0.24, 0.34),
            ("four", 0.36, 0.46), ("five", 0.48, 0.58), ("six", 0.60, 0.70),
            ("seven", 0.72, 0.82),
        )

        ass = burner.generate_ass(
            words, 1920, 1080,
            style=SubtitleStyle(min_words_per_group=3, max_words_per_group=5),
        )

        assert self._dialogue_texts(ass) == ["one two three four", "five six seven"]

    def test_adaptive_groups_break_early_after_attached_terminal_punctuation(self, burner):
        """A terminal token ends a phrase even when it has fewer than three words."""
        words = self._words(
            ("Wait!", 0.00, 0.15), ("Now", 0.17, 0.29), ("listen", 0.31, 0.43),
            ("very", 0.45, 0.57), ("closely", 0.59, 0.71),
        )

        ass = burner.generate_ass(
            words, 1920, 1080,
            style=SubtitleStyle(min_words_per_group=3, max_words_per_group=5),
        )

        assert self._dialogue_texts(ass) == ["Wait!", "Now listen very closely"]

    def test_adaptive_groups_break_early_for_a_real_pause(self, burner):
        """A pause over 0.2 seconds permits a short phrase at its boundary."""
        words = self._words(
            ("Please", 0.00, 0.10), ("wait", 0.12, 0.22),
            ("for", 0.50, 0.60), ("a", 0.62, 0.72), ("moment", 0.74, 0.84),
        )

        ass = burner.generate_ass(
            words, 1920, 1080,
            style=SubtitleStyle(min_words_per_group=3, max_words_per_group=5),
        )

        assert self._dialogue_texts(ass) == ["Please wait", "for a moment"]

    def test_adaptive_groups_rebalance_a_continuous_duration_tail(self, burner):
        """A duration split may move one word when the resulting 3+3 groups fit."""
        duration_words = self._words(
            ("one", 0.00, 0.35), ("two", 0.40, 0.75), ("three", 0.80, 1.15),
            ("four", 1.20, 1.55), ("five", 1.60, 2.05), ("six", 2.10, 2.45),
        )
        duration_ass = burner.generate_ass(
            duration_words, 1920, 1080,
            style=SubtitleStyle(min_words_per_group=3, max_words_per_group=5),
        )
        assert self._dialogue_texts(duration_ass) == ["one two three", "four five six"]

    def test_adaptive_groups_rebalance_a_continuous_width_tail(self, burner):
        """A width split may move one word when the resulting 3+3 groups fit."""
        width_words = self._words(
            ("one", 0.00, 0.10), ("two", 0.12, 0.22), ("three", 0.24, 0.34),
            ("four", 0.36, 0.46), ("five", 0.48, 0.58), ("six", 0.60, 0.70),
        )
        width_ass = burner.generate_ass(
            width_words, 1062, 1000,
            style=SubtitleStyle(
                font_name="Liberation Sans", font_size_pct=0.10, margin_h_pct=0.10,
                min_words_per_group=3, max_words_per_group=5,
            ),
        )
        assert self._dialogue_texts(width_ass) == ["one two three", "four five six"]

    @pytest.mark.parametrize(
        ("words", "expected"),
        [
            (
                (
                    ("one", 0.00, 0.10), ("two", 0.12, 0.22), ("three", 0.24, 0.34),
                    ("four.", 0.36, 0.46), ("five", 0.48, 0.58), ("six", 0.60, 0.70),
                ),
                ["one two three four.", "five six"],
            ),
            (
                (
                    ("one", 0.00, 0.10), ("two", 0.12, 0.22), ("three", 0.24, 0.34),
                    ("four", 0.36, 0.46), ("five", 0.80, 0.90), ("six", 0.92, 1.02),
                ),
                ["one two three four", "five six"],
            ),
        ],
    )
    def test_adaptive_groups_never_rebalance_across_punctuation_or_a_pause(
        self, burner, words, expected,
    ):
        """Speech boundaries outrank the target phrase-size range."""
        ass = burner.generate_ass(
            self._words(*words), 1920, 1080,
            style=SubtitleStyle(min_words_per_group=3, max_words_per_group=5),
        )

        assert self._dialogue_texts(ass) == expected

    def test_explicit_one_and_two_word_styles_remain_compatible(self, burner, sample_words):
        """Existing explicitly configured compact caption styles are unchanged."""
        one_word = burner.generate_ass(
            sample_words, 1080, 1920,
            style=SubtitleStyle(min_words_per_group=1, max_words_per_group=1),
        )
        two_words = burner.generate_ass(
            sample_words, 1080, 1920,
            style=SubtitleStyle(min_words_per_group=1, max_words_per_group=2),
        )

        assert self._dialogue_texts(one_word) == ["Never", "gonna", "give", "you", "up"]
        assert self._dialogue_texts(two_words) == ["Never gonna", "give you", "up"]


class TestKaraokeTiming:
    """Karaoke durations follow word onsets while dialogue bounds remain exact."""

    def test_karaoke_uses_next_word_onset_and_true_phrase_bounds(self, burner):
        words = [
            WordTimestamp(word="one", start=0.00, end=0.10),
            WordTimestamp(word="two", start=0.25, end=0.35),
            WordTimestamp(word="three", start=0.36, end=0.45),
        ]

        ass = burner.generate_ass(
            words, 1080, 1920,
            style=SubtitleStyle(karaoke=True, min_words_per_group=3, max_words_per_group=3),
        )
        dialogue = next(line for line in ass.splitlines() if line.startswith("Dialogue:"))

        assert dialogue.startswith("Dialogue: 0,0:00:00.00,0:00:00.45,")
        assert dialogue.endswith(r"{\k25}one {\k11}two {\k9}three")

    def test_karaoke_rounding_never_emits_zero_centiseconds(self, burner):
        words = [
            WordTimestamp(word="one", start=0.000, end=0.001),
            WordTimestamp(word="two", start=0.001, end=0.002),
            WordTimestamp(word="three", start=0.002, end=0.003),
        ]

        ass = burner.generate_ass(
            words, 1080, 1920,
            style=SubtitleStyle(karaoke=True, min_words_per_group=3, max_words_per_group=3),
        )

        assert r"{\k1}one {\k1}two {\k1}three" in ass


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

    @patch("yacg.subtitle_burner.subprocess.run")
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

    @patch("yacg.subtitle_burner.subprocess.run")
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
