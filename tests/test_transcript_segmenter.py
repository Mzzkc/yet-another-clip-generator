"""
Tests for the TranscriptSegmenter module.

Covers full_transcribe (with mock Whisper), segment_by_content (with mock
Ollama), refine_boundaries (pure logic), and failure-is-error contract.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from viral_clip_extractor.models import (
    SceneSegment,
    SegmentBoundary,
    WordTimestamp,
)
from viral_clip_extractor.transcript_segmenter import TranscriptSegmenter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def segmenter() -> TranscriptSegmenter:
    return TranscriptSegmenter(whisper_model="tiny")


@pytest.fixture
def sample_words() -> list[WordTimestamp]:
    """Simulated word timestamps spanning 0–30s with speech pauses.

    Includes punctuation on sentence-ending words so _find_speech_pauses
    can detect sentence boundaries.
    """
    return [
        WordTimestamp(word="We're", start=0.0, end=0.3),
        WordTimestamp(word="no", start=0.35, end=0.5),
        WordTimestamp(word="strangers", start=0.55, end=1.0),
        WordTimestamp(word="to", start=1.05, end=1.2),
        WordTimestamp(word="love.", start=1.25, end=1.8),
        # 0.7s pause here (sentence-ending: "love.")
        WordTimestamp(word="You", start=2.5, end=2.7),
        WordTimestamp(word="know", start=2.75, end=3.0),
        WordTimestamp(word="the", start=3.05, end=3.2),
        WordTimestamp(word="rules,", start=3.25, end=3.6),
        # 0.4s pause here (NOT sentence-ending: "rules,")
        WordTimestamp(word="and", start=4.0, end=4.2),
        WordTimestamp(word="so", start=4.25, end=4.4),
        WordTimestamp(word="do", start=4.45, end=4.6),
        WordTimestamp(word="I", start=4.65, end=4.8),
        # Long content continues...
        WordTimestamp(word="A", start=10.0, end=10.2),
        WordTimestamp(word="full", start=10.25, end=10.5),
        WordTimestamp(word="commitment", start=10.55, end=11.0),
        WordTimestamp(word="what", start=15.0, end=15.2),
        WordTimestamp(word="I'm", start=15.25, end=15.4),
        WordTimestamp(word="thinking", start=15.45, end=15.8),
        WordTimestamp(word="of", start=15.85, end=16.0),
        WordTimestamp(word="You", start=20.0, end=20.2),
        WordTimestamp(word="would", start=20.25, end=20.5),
        WordTimestamp(word="not", start=20.55, end=20.7),
        WordTimestamp(word="get", start=20.75, end=20.9),
        WordTimestamp(word="this", start=20.95, end=21.1),
        WordTimestamp(word="from", start=25.0, end=25.2),
        WordTimestamp(word="any", start=25.25, end=25.4),
        WordTimestamp(word="other", start=25.45, end=25.7),
        WordTimestamp(word="guy.", start=25.75, end=26.0),
    ]


# ---------------------------------------------------------------------------
# full_transcribe tests
# ---------------------------------------------------------------------------

class TestFullTranscribe:
    """Tests for TranscriptSegmenter.full_transcribe (mock Whisper)."""

    def test_transcript_segmenter_transcribes(self, segmenter):
        """full_transcribe returns WordTimestamp list with word-level timing."""
        mock_model = MagicMock()
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95

        # Simulate Whisper segment with words
        mock_word1 = MagicMock()
        mock_word1.word = " Hello"
        mock_word1.start = 0.0
        mock_word1.end = 0.5
        mock_word1.probability = 0.95

        mock_word2 = MagicMock()
        mock_word2.word = " world"
        mock_word2.start = 0.6
        mock_word2.end = 1.0
        mock_word2.probability = 0.88

        mock_segment = MagicMock()
        mock_segment.words = [mock_word1, mock_word2]
        mock_model.transcribe.return_value = (iter([mock_segment]), mock_info)

        with patch.dict("sys.modules", {"faster_whisper": MagicMock(WhisperModel=lambda *a, **kw: mock_model)}):
            words = segmenter.full_transcribe("/fake/video.mp4")

        assert len(words) == 2
        assert all(isinstance(w, WordTimestamp) for w in words)
        assert words[0].word == "Hello"
        assert words[0].start == 0.0
        assert words[0].end == 0.5
        assert words[1].word == "world"

    def test_transcript_segmenter_raises_on_silent(self, segmenter):
        """full_transcribe raises RuntimeError on empty/silent video."""
        mock_model = MagicMock()
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.9

        # Whisper returns segments with no words
        mock_segment = MagicMock()
        mock_segment.words = []
        mock_model.transcribe.return_value = (iter([mock_segment]), mock_info)

        with patch(
            "viral_clip_extractor.transcript_segmenter.WhisperModel",
            return_value=mock_model,
            create=True,
        ):
            # Patch the import inside the method
            with patch.dict("sys.modules", {"faster_whisper": MagicMock(WhisperModel=lambda *a, **kw: mock_model)}):
                with pytest.raises(RuntimeError, match="No speech detected"):
                    segmenter.full_transcribe("/fake/silent.mp4")

    def test_transcript_segmenter_raises_on_failure(self, segmenter):
        """full_transcribe raises RuntimeError when faster-whisper import fails."""
        with patch.dict("sys.modules", {"faster_whisper": None}):
            with pytest.raises(RuntimeError, match="faster-whisper is required"):
                segmenter.full_transcribe("/fake/video.mp4")


# ---------------------------------------------------------------------------
# segment_by_content tests
# ---------------------------------------------------------------------------

class TestSegmentByContent:
    """Tests for TranscriptSegmenter.segment_by_content (mock Ollama)."""

    def test_transcript_segmenter_segments_content(
        self, segmenter, sample_words
    ):
        """segment_by_content sends transcript to Ollama and parses response."""
        ollama_response = json.dumps([
            {
                "start_time": 0.0,
                "end_time": 30.0,
                "hook_summary": "Classic intro",
                "segment_type": "hook",
            },
        ])

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": ollama_response}

        with patch.object(segmenter._session, "post", return_value=mock_resp) as mock_post:
            boundaries = segmenter.segment_by_content(sample_words, "Rickroll")

        assert len(boundaries) == 1
        assert isinstance(boundaries[0], SegmentBoundary)
        assert boundaries[0].start_time == 0.0
        # end_time clamped from 30.0 to transcript end (26.0) by
        # _validate_timestamps (P3-16 fix)
        assert boundaries[0].end_time == 26.0
        assert boundaries[0].hook_summary == "Classic intro"
        assert boundaries[0].segment_type == "hook"

    def test_segment_by_content_raises_on_ollama_failure(
        self, segmenter, sample_words
    ):
        """segment_by_content raises after all retries exhausted."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500

        with patch.object(segmenter._session, "post", return_value=mock_resp):
            with patch.object(
                TranscriptSegmenter, "_backoff", return_value=None
            ):
                with pytest.raises(RuntimeError, match="Failed to parse"):
                    segmenter.segment_by_content(sample_words, "Test")


# ---------------------------------------------------------------------------
# refine_boundaries tests
# ---------------------------------------------------------------------------

class TestRefineBoundaries:
    """Tests for TranscriptSegmenter.refine_boundaries (pure logic)."""

    def test_transcript_segmenter_refines_boundaries(
        self, segmenter, sample_words
    ):
        """refine_boundaries snaps to speech pauses and returns SceneSegments."""
        boundaries = [
            SegmentBoundary(
                start_time=0.0,
                end_time=26.0,
                hook_summary="Full song",
                segment_type="hook",
            ),
        ]

        scenes = segmenter.refine_boundaries(boundaries, sample_words)

        assert len(scenes) >= 1
        for scene in scenes:
            assert isinstance(scene, SceneSegment)
            assert scene.start_time >= 0.0
            assert scene.end_time > scene.start_time
            assert scene.scene_index >= 0
            assert scene.duration >= 15.0  # min duration enforced

    def test_refine_boundaries_returns_scene_segments(
        self, segmenter, sample_words
    ):
        """Output type is list[SceneSegment] with sequential indices."""
        boundaries = [
            SegmentBoundary(
                start_time=0.0, end_time=26.0,
                hook_summary="Part 1", segment_type="hook",
            ),
        ]

        scenes = segmenter.refine_boundaries(boundaries, sample_words)
        assert all(isinstance(s, SceneSegment) for s in scenes)
        for i, s in enumerate(scenes):
            assert s.scene_index == i

    def test_refine_boundaries_empty_words(self, segmenter):
        """refine_boundaries with no words returns empty list."""
        boundaries = [
            SegmentBoundary(
                start_time=0.0, end_time=30.0,
                hook_summary="Test", segment_type="hook",
            ),
        ]
        result = segmenter.refine_boundaries(boundaries, [])
        assert result == []


# ---------------------------------------------------------------------------
# Internal helper tests
# ---------------------------------------------------------------------------

class TestHelpers:
    """Tests for internal helper methods."""

    def test_find_speech_pauses(self, segmenter, sample_words):
        """_find_speech_pauses identifies gaps > 300ms with sentence flags."""
        pauses = segmenter._find_speech_pauses(sample_words)
        assert len(pauses) > 0
        for pause_start, pause_end, is_sentence_end in pauses:
            assert pause_end - pause_start > 0.3
        # The first pause follows "love." — should be sentence-ending
        assert pauses[0][2] is True

    def test_find_speech_pauses_non_sentence(self, segmenter, sample_words):
        """_find_speech_pauses marks comma-pauses as non-sentence-ending."""
        pauses = segmenter._find_speech_pauses(sample_words)
        # The second pause follows "rules," — NOT sentence-ending
        assert pauses[1][2] is False

    def test_snap_to_nearest_pause_before(self, segmenter):
        """_snap_to_nearest_pause with prefer='before' snaps to pause <= target."""
        pauses = [(1.0, 1.5, False), (3.0, 3.5, False), (5.0, 5.5, False)]
        # Target at 3.2: pause_end 1.5 is <= 3.2 (dist 1.7), no other pause_end <= 3.2
        result = segmenter._snap_to_nearest_pause(3.2, pauses, prefer="before")
        assert result == 1.5
        assert result <= 3.2  # Must be before or at target

    def test_snap_to_nearest_pause_before_direction_enforced(self, segmenter):
        """_snap_to_nearest_pause with prefer='before' never returns after target."""
        pauses = [(5.0, 5.5, False), (9.5, 10.5, False)]
        result = segmenter._snap_to_nearest_pause(10.0, pauses, prefer="before")
        assert result <= 10.0  # Must not snap AFTER target
        assert result == 5.5  # pause_end 5.5 is the only one <= 10.0

    def test_snap_to_nearest_pause_after(self, segmenter):
        """_snap_to_nearest_pause with prefer='after' snaps correctly."""
        pauses = [(1.0, 1.5, False), (3.0, 3.5, False), (5.0, 5.5, False)]
        result = segmenter._snap_to_nearest_pause(2.5, pauses, prefer="after")
        assert result == 3.0
        assert result >= 2.5  # Must be after or at target

    def test_snap_to_nearest_pause_after_direction_enforced(self, segmenter):
        """_snap_to_nearest_pause with prefer='after' never returns before target."""
        pauses = [(9.5, 10.5, False), (20.0, 20.5, False)]
        result = segmenter._snap_to_nearest_pause(10.0, pauses, prefer="after")
        assert result >= 10.0  # Must not snap BEFORE target
        assert result == 20.0  # pause_start 20.0 is the closest >= 10.0

    def test_snap_to_nearest_pause_no_pauses(self, segmenter):
        """_snap_to_nearest_pause returns target when no pauses."""
        result = segmenter._snap_to_nearest_pause(5.0, [], prefer="before")
        assert result == 5.0

    def test_snap_prefers_sentence_ending_pause(self, segmenter):
        """_snap_to_nearest_pause prefers sentence-ending pauses within 5s tolerance."""
        # Mid-sentence pause at 3.0 (closer to target 3.5)
        # Sentence-ending pause at 5.0 (within 5s tolerance of target)
        pauses = [
            (2.5, 3.0, False),   # mid-sentence, closest
            (4.5, 5.0, True),    # sentence-ending, within 5s
            (10.0, 10.5, True),  # sentence-ending, too far
        ]
        # prefer="before": target=6.0
        # Closest pause_end <= 6.0: 5.0 (dist 1.0), 3.0 (dist 3.0)
        # Closest sentence-ending pause_end <= 6.0: 5.0 (dist 1.0)
        # Sentence-ending 5.0 is within 5s → prefer it
        result = segmenter._snap_to_nearest_pause(6.0, pauses, prefer="before")
        assert result == 5.0

    def test_snap_prefers_sentence_ending_pause_after(self, segmenter):
        """prefer='after' also prefers sentence-ending pauses within tolerance."""
        pauses = [
            (3.0, 3.5, False),   # mid-sentence, closest after target
            (5.0, 5.5, True),    # sentence-ending, within 5s
        ]
        # target=2.5, prefer="after"
        # Closest pause_start >= 2.5: 3.0 (dist 0.5)
        # Closest sentence-ending pause_start >= 2.5: 5.0 (dist 2.5)
        # Sentence-ending 5.0 is within 5s → prefer it
        result = segmenter._snap_to_nearest_pause(2.5, pauses, prefer="after")
        assert result == 5.0

    def test_snap_ignores_sentence_pause_beyond_tolerance(self, segmenter):
        """Sentence-ending pause beyond 5s tolerance falls back to closest."""
        pauses = [
            (3.0, 3.5, False),   # mid-sentence, closest
            (15.0, 15.5, True),  # sentence-ending, beyond 5s from target
        ]
        # target=4.0, prefer="before"
        # Closest pause_end <= 4.0: 3.5 (dist 0.5)
        # Sentence-ending pause_end <= 4.0: none (15.5 > 4.0)
        # Falls back to closest: 3.5
        result = segmenter._snap_to_nearest_pause(4.0, pauses, prefer="before")
        assert result == 3.5

    def test_format_transcript(self, segmenter, sample_words):
        """_format_transcript produces timestamped text blocks."""
        text = segmenter._format_transcript(sample_words)
        assert "[" in text
        assert "]" in text
        assert "We're" in text

    def test_format_transcript_empty(self, segmenter):
        """_format_transcript returns empty string for empty words."""
        assert segmenter._format_transcript([]) == ""
