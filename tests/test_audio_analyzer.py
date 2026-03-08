"""
Tests for the AudioAnalyzer module.

All external dependencies (librosa, faster-whisper, soundfile) are mocked
so tests run without hardware or heavy libraries installed.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from viral_clip_extractor.core.audio_analyzer import AudioAnalyzer, _DEFAULT_ASMR_KEYWORDS
from viral_clip_extractor.models import AudioFeatures


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer() -> AudioAnalyzer:
    """An AudioAnalyzer with default keywords."""
    return AudioAnalyzer()


@pytest.fixture
def custom_analyzer() -> AudioAnalyzer:
    """An AudioAnalyzer with custom keywords."""
    return AudioAnalyzer(asmr_keywords=["custom", "words"])


def _fake_audio(sr: int = 22050, duration: float = 5.0) -> np.ndarray:
    """Generate a fake audio signal (sine wave) for testing."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests: Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_keywords(self, analyzer: AudioAnalyzer) -> None:
        """Default keyword list matches the module-level default."""
        assert analyzer.asmr_keywords == _DEFAULT_ASMR_KEYWORDS

    def test_custom_keywords(self, custom_analyzer: AudioAnalyzer) -> None:
        """Custom keyword list is used when provided."""
        assert custom_analyzer.asmr_keywords == ["custom", "words"]

    def test_none_keywords_uses_default(self) -> None:
        """Passing None explicitly still yields the default list."""
        a = AudioAnalyzer(asmr_keywords=None)
        assert a.asmr_keywords == _DEFAULT_ASMR_KEYWORDS


# ---------------------------------------------------------------------------
# Tests: analyze_segment — happy path
# ---------------------------------------------------------------------------

class TestAnalyzeSegment:
    @patch("viral_clip_extractor.core.audio_analyzer.AudioAnalyzer._detect_trigger_words")
    def test_returns_audio_features(self, mock_trigger: MagicMock, analyzer: AudioAnalyzer) -> None:
        """analyze_segment returns an AudioFeatures dataclass."""
        mock_trigger.return_value = []
        fake_y = _fake_audio()

        with patch("librosa.load", return_value=(fake_y, 22050)):
            with patch("librosa.feature.rms", return_value=[np.array([0.1, 0.2, 0.3, 0.4, 0.5])]):
                with patch("librosa.feature.spectral_centroid", return_value=[np.array([3000.0, 5000.0, 6000.0])]):
                    with patch("librosa.feature.zero_crossing_rate", return_value=[np.array([0.05, 0.1])]):
                        with patch("librosa.onset.onset_strength", return_value=np.array([0.1])):
                            with patch("librosa.onset.onset_detect", return_value=np.array([1.0, 2.0])):
                                with patch("librosa.stft", return_value=np.ones((1025, 10))):
                                    with patch("librosa.fft_frequencies", return_value=np.linspace(0, 11025, 1025)):
                                        result = analyzer.analyze_segment("/fake/video.mp4", 0.0, 5.0)

        assert isinstance(result, AudioFeatures)
        assert result.audio_peak_score > 0
        assert result.overall_energy > 0

    @patch("viral_clip_extractor.core.audio_analyzer.AudioAnalyzer._detect_trigger_words")
    def test_high_freq_score_computation(self, mock_trigger: MagicMock, analyzer: AudioAnalyzer) -> None:
        """High-freq score reflects fraction of spectral centroid > 4 kHz."""
        mock_trigger.return_value = []
        fake_y = _fake_audio()

        # All centroids above 4 kHz → high_freq_score close to 1.0
        with patch("librosa.load", return_value=(fake_y, 22050)):
            with patch("librosa.feature.rms", return_value=[np.array([0.1])]):
                with patch("librosa.feature.spectral_centroid", return_value=[np.array([5000.0, 6000.0, 7000.0])]):
                    with patch("librosa.feature.zero_crossing_rate", return_value=[np.array([0.05])]):
                        with patch("librosa.onset.onset_strength", return_value=np.array([0.0])):
                            with patch("librosa.onset.onset_detect", return_value=np.array([])):
                                with patch("librosa.stft", return_value=np.ones((1025, 10))):
                                    with patch("librosa.fft_frequencies", return_value=np.linspace(0, 11025, 1025)):
                                        result = analyzer.analyze_segment("/fake/video.mp4", 0.0, 5.0)

        # All centroid frames > 4000, so base score = 1.0 (may get crinkle boost capped at 1.0)
        assert result.high_freq_score >= 0.9

    @patch("viral_clip_extractor.core.audio_analyzer.AudioAnalyzer._detect_trigger_words")
    def test_dynamic_range_and_zcr(self, mock_trigger: MagicMock, analyzer: AudioAnalyzer) -> None:
        """Dynamic range is std of RMS, zcr_score is mean of zero-crossing rate."""
        mock_trigger.return_value = []
        fake_y = _fake_audio()
        rms_vals = np.array([0.1, 0.5, 0.1, 0.5])
        zcr_vals = np.array([0.2, 0.4])

        with patch("librosa.load", return_value=(fake_y, 22050)):
            with patch("librosa.feature.rms", return_value=[rms_vals]):
                with patch("librosa.feature.spectral_centroid", return_value=[np.array([3000.0])]):
                    with patch("librosa.feature.zero_crossing_rate", return_value=[zcr_vals]):
                        with patch("librosa.onset.onset_strength", return_value=np.array([0.0])):
                            with patch("librosa.onset.onset_detect", return_value=np.array([])):
                                with patch("librosa.stft", return_value=np.ones((1025, 10))):
                                    with patch("librosa.fft_frequencies", return_value=np.linspace(0, 11025, 1025)):
                                        result = analyzer.analyze_segment("/fake/video.mp4", 0.0, 5.0)

        assert abs(result.dynamic_range - float(np.std(rms_vals))) < 1e-6
        assert abs(result.zcr_score - float(np.mean(zcr_vals))) < 0.05  # allow small ASMR boost


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_invalid_duration_returns_zeros(self, analyzer: AudioAnalyzer) -> None:
        """Negative duration (end < start) produces zero-valued features."""
        result = analyzer.analyze_segment("/fake/video.mp4", 10.0, 5.0)
        assert isinstance(result, AudioFeatures)
        assert result.audio_peak_score == 0.0
        assert result.overall_energy == 0.0
        assert result.trigger_words == []

    def test_zero_duration_returns_zeros(self, analyzer: AudioAnalyzer) -> None:
        """Zero duration (start == end) produces zero-valued features."""
        result = analyzer.analyze_segment("/fake/video.mp4", 5.0, 5.0)
        assert isinstance(result, AudioFeatures)
        assert result.audio_peak_score == 0.0

    @patch("viral_clip_extractor.core.audio_analyzer.AudioAnalyzer._detect_trigger_words")
    def test_empty_audio_returns_zeros(self, mock_trigger: MagicMock, analyzer: AudioAnalyzer) -> None:
        """Empty audio array from librosa.load yields zeros."""
        mock_trigger.return_value = []
        with patch("librosa.load", return_value=(np.array([]), 22050)):
            result = analyzer.analyze_segment("/fake/video.mp4", 0.0, 5.0)

        assert isinstance(result, AudioFeatures)
        assert result.audio_peak_score == 0.0
        assert result.high_freq_score == 0.0

    @patch("viral_clip_extractor.core.audio_analyzer.AudioAnalyzer._detect_trigger_words")
    def test_librosa_load_exception(self, mock_trigger: MagicMock, analyzer: AudioAnalyzer) -> None:
        """If librosa.load raises, raise RuntimeError (failure = error)."""
        mock_trigger.return_value = []
        with patch("librosa.load", side_effect=RuntimeError("corrupt file")):
            with pytest.raises(RuntimeError, match="Failed to load audio"):
                analyzer.analyze_segment("/fake/corrupt.mp4", 0.0, 5.0)

    def test_librosa_import_failure(self, analyzer: AudioAnalyzer) -> None:
        """If librosa is completely missing, raise RuntimeError (failure = error)."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "librosa":
                raise ImportError("No module named 'librosa'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(RuntimeError, match="librosa is required"):
                analyzer.analyze_segment("/fake/video.mp4", 0.0, 5.0)


# ---------------------------------------------------------------------------
# Tests: Trigger word detection
# ---------------------------------------------------------------------------

class TestTriggerWords:
    def test_trigger_words_found(self, analyzer: AudioAnalyzer) -> None:
        """Trigger words present in transcription are returned."""
        mock_seg = MagicMock()
        mock_seg.text = "This is so relaxing, like gentle whisper tingles"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([mock_seg]), MagicMock())

        mock_whisper_module = MagicMock()
        mock_whisper_module.WhisperModel.return_value = mock_model

        with patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}):
            with patch(
                "viral_clip_extractor.utils.video_utils.extract_audio"
            ):
                result = analyzer._detect_trigger_words("/fake/audio.wav", 0.0, 2.0)

        assert "relax" in result
        assert "gentle" in result
        assert "whisper" in result
        assert "tingles" in result

    def test_trigger_words_none_found(self, analyzer: AudioAnalyzer) -> None:
        """No trigger words if transcript doesn't match."""
        mock_seg = MagicMock()
        mock_seg.text = "Hello everyone welcome to the stream"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([mock_seg]), MagicMock())

        mock_whisper_module = MagicMock()
        mock_whisper_module.WhisperModel.return_value = mock_model

        with patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}):
            with patch(
                "viral_clip_extractor.utils.video_utils.extract_audio"
            ):
                result = analyzer._detect_trigger_words("/fake/audio.wav", 0.0, 2.0)

        assert result == []

    def test_trigger_words_without_faster_whisper(self, analyzer: AudioAnalyzer) -> None:
        """Returns empty list gracefully when faster-whisper is not installed."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "faster_whisper":
                raise ImportError("No module named 'faster_whisper'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = analyzer._detect_trigger_words("/fake/audio.wav", 0.0, 2.0)

        assert result == []

    def test_custom_keywords_used_for_detection(self, custom_analyzer: AudioAnalyzer) -> None:
        """Custom keyword list is used for matching."""
        mock_seg = MagicMock()
        mock_seg.text = "This has custom words in it"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([mock_seg]), MagicMock())

        mock_whisper_module = MagicMock()
        mock_whisper_module.WhisperModel.return_value = mock_model

        with patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}):
            with patch(
                "viral_clip_extractor.utils.video_utils.extract_audio"
            ):
                result = custom_analyzer._detect_trigger_words("/fake/audio.wav", 0.0, 2.0)

        assert "custom" in result
        assert "words" in result


# ---------------------------------------------------------------------------
# Tests: ASMR-specific detections
# ---------------------------------------------------------------------------

class TestASMRDetections:
    @patch("viral_clip_extractor.core.audio_analyzer.AudioAnalyzer._detect_trigger_words")
    def test_tapping_detection_boosts_peak(self, mock_trigger: MagicMock, analyzer: AudioAnalyzer) -> None:
        """Dense onsets (tapping) boost the audio_peak_score."""
        mock_trigger.return_value = []
        fake_y = _fake_audio()

        base_rms = np.array([0.1])
        # Many onsets in short duration → tapping boost
        many_onsets = np.linspace(0.0, 4.0, 50)

        with patch("librosa.load", return_value=(fake_y, 22050)):
            with patch("librosa.feature.rms", return_value=[base_rms]):
                with patch("librosa.feature.spectral_centroid", return_value=[np.array([3000.0])]):
                    with patch("librosa.feature.zero_crossing_rate", return_value=[np.array([0.05])]):
                        with patch("librosa.onset.onset_strength", return_value=np.ones(50)):
                            with patch("librosa.onset.onset_detect", return_value=many_onsets):
                                with patch("librosa.stft", return_value=np.ones((1025, 10))):
                                    with patch("librosa.fft_frequencies", return_value=np.linspace(0, 11025, 1025)):
                                        result = analyzer.analyze_segment("/fake/video.mp4", 0.0, 5.0)

        # Base peak = percentile(90) of [0.1] = 0.1, tapping should add a boost
        assert result.audio_peak_score > float(np.percentile(base_rms, 90))

    @patch("viral_clip_extractor.core.audio_analyzer.AudioAnalyzer._detect_trigger_words")
    def test_onset_detect_failure_does_not_crash(self, mock_trigger: MagicMock, analyzer: AudioAnalyzer) -> None:
        """If onset detection raises, analysis still completes."""
        mock_trigger.return_value = []
        fake_y = _fake_audio()

        with patch("librosa.load", return_value=(fake_y, 22050)):
            with patch("librosa.feature.rms", return_value=[np.array([0.2])]):
                with patch("librosa.feature.spectral_centroid", return_value=[np.array([3000.0])]):
                    with patch("librosa.feature.zero_crossing_rate", return_value=[np.array([0.05])]):
                        with patch("librosa.onset.onset_strength", side_effect=RuntimeError("boom")):
                            with patch("librosa.stft", return_value=np.ones((1025, 10))):
                                with patch("librosa.fft_frequencies", return_value=np.linspace(0, 11025, 1025)):
                                    result = analyzer.analyze_segment("/fake/video.mp4", 0.0, 5.0)

        assert isinstance(result, AudioFeatures)
        assert result.audio_peak_score > 0  # Still got basic features
