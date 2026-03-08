"""
Tests for the virality scorer and clip extractor modules.

All external dependencies (FFmpeg, cv2, librosa, scenedetect) are mocked
so tests run without hardware or media files.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from viral_clip_extractor.models import (
    AudioFeatures,
    ClipData,
    PipelineConfig,
    SceneSegment,
    SemanticFeatures,
    ViralityScore,
    VisualFeatures,
)
from viral_clip_extractor.core.virality_scorer import ViralityScorer
from viral_clip_extractor.extractors.clip_extractor import ClipExtractor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def audio() -> AudioFeatures:
    """Audio features with moderate values."""
    return AudioFeatures(
        audio_peak_score=0.7,
        high_freq_score=0.5,
        dynamic_range=0.3,
        zcr_score=0.2,
        trigger_words=["tingles"],
        overall_energy=0.6,
    )


@pytest.fixture
def visual() -> VisualFeatures:
    """Visual features with moderate values."""
    return VisualFeatures(
        motion_score=15.0,
        face_presence=0.8,
        visual_interest=50.0,
        composition_score=7.0,
    )


@pytest.fixture
def semantic() -> SemanticFeatures:
    """Semantic features at mid-range."""
    return SemanticFeatures(
        emotional_intensity=7.0,
        narrative_interest=6.0,
        hook_potential=8.0,
        asmr_quality=9.0,
        visual_appeal=7.5,
        uniqueness=5.0,
        description="Dragon ASMR tapping sequence",
    )


@pytest.fixture
def scorer() -> ViralityScorer:
    """Default scorer with ASMR weights."""
    return ViralityScorer()


@pytest.fixture
def segment() -> SceneSegment:
    return SceneSegment(start_time=10.0, end_time=25.0, scene_index=0)


# ---------------------------------------------------------------------------
# ViralityScorer — calculate_score
# ---------------------------------------------------------------------------

class TestViralityScorerCalculateScore:
    """Tests for ViralityScorer.calculate_score."""

    def test_returns_virality_score_type(
        self, scorer: ViralityScorer, audio: AudioFeatures,
        visual: VisualFeatures, semantic: SemanticFeatures,
    ) -> None:
        result = scorer.calculate_score(audio, visual, semantic, duration=15.0)
        assert isinstance(result, ViralityScore)

    def test_score_in_valid_range(
        self, scorer: ViralityScorer, audio: AudioFeatures,
        visual: VisualFeatures, semantic: SemanticFeatures,
    ) -> None:
        result = scorer.calculate_score(audio, visual, semantic, duration=15.0)
        assert 0.0 <= result.total_score <= 100.0

    def test_component_scores_present(
        self, scorer: ViralityScorer, audio: AudioFeatures,
        visual: VisualFeatures, semantic: SemanticFeatures,
    ) -> None:
        result = scorer.calculate_score(audio, visual, semantic, duration=15.0)
        expected_keys = {
            "audio_peaks", "high_freq", "motion", "visual", "composition",
            "hook", "emotional", "asmr", "narrative", "uniqueness", "duration",
        }
        assert set(result.component_scores.keys()) == expected_keys

    def test_confidence_full_with_semantic(
        self, scorer: ViralityScorer, audio: AudioFeatures,
        visual: VisualFeatures, semantic: SemanticFeatures,
    ) -> None:
        result = scorer.calculate_score(audio, visual, semantic, duration=15.0)
        assert result.confidence == 1.0

    def test_confidence_reduced_without_semantic(
        self, scorer: ViralityScorer, audio: AudioFeatures,
        visual: VisualFeatures,
    ) -> None:
        result = scorer.calculate_score(audio, visual, None, duration=15.0)
        assert result.confidence == 0.5

    def test_semantic_none_redistributes_weights(
        self, scorer: ViralityScorer, audio: AudioFeatures,
        visual: VisualFeatures,
    ) -> None:
        """When semantic is None, the score should still be non-zero
        if audio/visual inputs are non-zero."""
        result = scorer.calculate_score(audio, visual, None, duration=15.0)
        assert result.total_score > 0.0
        # Semantic components should be zero
        for key in ("hook", "emotional", "asmr", "narrative", "uniqueness"):
            assert result.component_scores[key] == 0.0

    def test_higher_inputs_produce_higher_score(
        self, scorer: ViralityScorer,
    ) -> None:
        low_audio = AudioFeatures(
            audio_peak_score=0.1, high_freq_score=0.1,
            dynamic_range=0.05, zcr_score=0.05,
        )
        high_audio = AudioFeatures(
            audio_peak_score=0.9, high_freq_score=0.9,
            dynamic_range=0.4, zcr_score=0.3,
        )
        low_visual = VisualFeatures(
            motion_score=2.0, face_presence=0.1,
            visual_interest=10.0, composition_score=2.0,
        )
        high_visual = VisualFeatures(
            motion_score=40.0, face_presence=0.9,
            visual_interest=90.0, composition_score=9.0,
        )
        low_sem = SemanticFeatures(
            emotional_intensity=1.0, narrative_interest=1.0,
            hook_potential=1.0, asmr_quality=1.0,
            visual_appeal=1.0, uniqueness=1.0,
        )
        high_sem = SemanticFeatures(
            emotional_intensity=9.0, narrative_interest=9.0,
            hook_potential=9.0, asmr_quality=9.0,
            visual_appeal=9.0, uniqueness=9.0,
        )

        low_score = scorer.calculate_score(low_audio, low_visual, low_sem, 15.0)
        high_score = scorer.calculate_score(high_audio, high_visual, high_sem, 15.0)
        assert high_score.total_score > low_score.total_score

    def test_custom_weights(
        self, audio: AudioFeatures, visual: VisualFeatures,
        semantic: SemanticFeatures,
    ) -> None:
        """Custom weights should affect scoring."""
        custom = {
            "hook": 0.0, "emotional": 0.0, "audio_peaks": 0.0,
            "asmr": 0.0, "motion": 0.0, "narrative": 0.0,
            "high_freq": 0.0, "uniqueness": 0.0, "visual": 0.0,
            "duration": 1.0,  # Only duration matters
        }
        scorer = ViralityScorer(weights=custom)
        result = scorer.calculate_score(audio, visual, semantic, duration=15.0)
        # Duration 15s is optimal → score of 10 → scaled to 100
        assert result.total_score == 100.0


# ---------------------------------------------------------------------------
# ViralityScorer — normalization and duration
# ---------------------------------------------------------------------------

class TestViralityScorerHelpers:
    """Tests for ViralityScorer._normalize and _duration_score."""

    def test_normalize_mid_value(self, scorer: ViralityScorer) -> None:
        assert scorer._normalize(0.5, 0.0, 1.0) == pytest.approx(5.0)

    def test_normalize_at_min(self, scorer: ViralityScorer) -> None:
        assert scorer._normalize(0.0, 0.0, 1.0) == 0.0

    def test_normalize_at_max(self, scorer: ViralityScorer) -> None:
        assert scorer._normalize(1.0, 0.0, 1.0) == 10.0

    def test_normalize_clamps_above_max(self, scorer: ViralityScorer) -> None:
        assert scorer._normalize(2.0, 0.0, 1.0) == 10.0

    def test_normalize_clamps_below_min(self, scorer: ViralityScorer) -> None:
        assert scorer._normalize(-1.0, 0.0, 1.0) == 0.0

    def test_normalize_equal_min_max(self, scorer: ViralityScorer) -> None:
        assert scorer._normalize(5.0, 5.0, 5.0) == 0.0

    def test_duration_optimal_range(self, scorer: ViralityScorer) -> None:
        for d in [7.0, 15.0, 30.0]:
            assert scorer._duration_score(d) == 10.0

    def test_duration_short_linear(self, scorer: ViralityScorer) -> None:
        assert scorer._duration_score(6.0) == pytest.approx(7.5)
        assert scorer._duration_score(5.0) == pytest.approx(5.0)

    def test_duration_long_linear(self, scorer: ViralityScorer) -> None:
        assert scorer._duration_score(60.0) == pytest.approx(5.0)

    def test_duration_very_short(self, scorer: ViralityScorer) -> None:
        assert scorer._duration_score(2.5) == pytest.approx(2.5)
        assert scorer._duration_score(0.0) == 0.0

    def test_duration_very_long_decays(self, scorer: ViralityScorer) -> None:
        score_70 = scorer._duration_score(70.0)
        score_100 = scorer._duration_score(100.0)
        assert score_70 < 5.0
        assert score_100 < score_70

    def test_duration_negative(self, scorer: ViralityScorer) -> None:
        assert scorer._duration_score(-5.0) == 0.0

    def test_public_duration_score_alias(self, scorer: ViralityScorer) -> None:
        """Public duration_score() should match private _duration_score()."""
        for d in [3.0, 6.0, 15.0, 45.0, 90.0]:
            assert scorer.duration_score(d) == scorer._duration_score(d)


# ---------------------------------------------------------------------------
# ViralityScorer — edge cases
# ---------------------------------------------------------------------------

class TestViralityScorerEdgeCases:
    """Edge cases and boundary conditions."""

    def test_zero_audio_features(self, scorer: ViralityScorer) -> None:
        audio = AudioFeatures(
            audio_peak_score=0.0, high_freq_score=0.0,
            dynamic_range=0.0, zcr_score=0.0,
        )
        visual = VisualFeatures(
            motion_score=0.0, face_presence=0.0,
            visual_interest=0.0, composition_score=0.0,
        )
        result = scorer.calculate_score(audio, visual, None, duration=0.0)
        assert isinstance(result, ViralityScore)
        assert result.total_score == 0.0

    def test_max_features(self, scorer: ViralityScorer) -> None:
        audio = AudioFeatures(
            audio_peak_score=2.0, high_freq_score=1.0,
            dynamic_range=0.5, zcr_score=0.5,
        )
        visual = VisualFeatures(
            motion_score=50.0, face_presence=1.0,
            visual_interest=100.0, composition_score=10.0,
        )
        semantic = SemanticFeatures(
            emotional_intensity=10.0, narrative_interest=10.0,
            hook_potential=10.0, asmr_quality=10.0,
            visual_appeal=10.0, uniqueness=10.0,
        )
        result = scorer.calculate_score(audio, visual, semantic, duration=15.0)
        assert result.total_score == 100.0

    def test_config_weights_used(self) -> None:
        config = PipelineConfig(
            scoring_weights={
                "hook": 0.5, "emotional": 0.0, "audio_peaks": 0.0,
                "asmr": 0.0, "motion": 0.0, "narrative": 0.0,
                "high_freq": 0.0, "uniqueness": 0.0, "visual": 0.0,
                "duration": 0.5,
            }
        )
        scorer = ViralityScorer(config=config)
        audio = AudioFeatures(
            audio_peak_score=0.5, high_freq_score=0.5,
            dynamic_range=0.2, zcr_score=0.1,
        )
        visual = VisualFeatures(
            motion_score=10.0, face_presence=0.5,
            visual_interest=50.0, composition_score=5.0,
        )
        semantic = SemanticFeatures(
            emotional_intensity=5.0, narrative_interest=5.0,
            hook_potential=10.0, asmr_quality=5.0,
            visual_appeal=5.0, uniqueness=5.0,
        )
        result = scorer.calculate_score(audio, visual, semantic, duration=15.0)
        # hook = 10/10 * 10 = 10, duration = 10, weighted avg = 10, * 10 = 100
        assert result.total_score == 100.0


# ---------------------------------------------------------------------------
# ClipExtractor — extract_clip
# ---------------------------------------------------------------------------

@patch(
    "viral_clip_extractor.extractors.clip_extractor.ClipExtractor._get_vertical_filter",
    return_value="crop=608:1080:236:0",
)
class TestClipExtractorExtractClip:
    """Tests for ClipExtractor.extract_clip (FFmpeg mocked)."""

    def _make_extractor(self) -> ClipExtractor:
        return ClipExtractor(context_padding=2.0)

    @patch("viral_clip_extractor.extractors.clip_extractor.subprocess.run")
    @patch(
        "viral_clip_extractor.extractors.clip_extractor.ClipExtractor._get_video_duration",
        return_value=120.0,
    )
    def test_successful_extraction(
        self, mock_dur: MagicMock, mock_run: MagicMock,
        mock_vf: MagicMock, tmp_path: Path,
    ) -> None:
        output = tmp_path / "out.mp4"
        # Simulate FFmpeg creating a file
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        def side_effect(*args, **kwargs):
            output.write_bytes(b"\x00" * 20000)  # >10KB
            return MagicMock(returncode=0, stderr="")

        mock_run.side_effect = side_effect

        ext = self._make_extractor()
        result = ext.extract_clip("/fake/video.mp4", 10.0, 25.0, str(output))
        assert result is True
        assert mock_run.called

    @patch("viral_clip_extractor.extractors.clip_extractor.subprocess.run")
    @patch(
        "viral_clip_extractor.extractors.clip_extractor.ClipExtractor._get_video_duration",
        return_value=120.0,
    )
    def test_ffmpeg_failure_returns_false(
        self, mock_dur: MagicMock, mock_run: MagicMock,
        mock_vf: MagicMock, tmp_path: Path,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=1, stderr="encoding error")
        output = tmp_path / "out.mp4"
        ext = self._make_extractor()
        result = ext.extract_clip("/fake/video.mp4", 10.0, 25.0, str(output))
        assert result is False

    @patch(
        "viral_clip_extractor.extractors.clip_extractor.subprocess.run",
        side_effect=FileNotFoundError("ffmpeg not found"),
    )
    @patch(
        "viral_clip_extractor.extractors.clip_extractor.ClipExtractor._get_video_duration",
        return_value=120.0,
    )
    def test_ffmpeg_not_found(
        self, mock_dur: MagicMock, mock_run: MagicMock,
        mock_vf: MagicMock, tmp_path: Path,
    ) -> None:
        output = tmp_path / "out.mp4"
        ext = self._make_extractor()
        result = ext.extract_clip("/fake/video.mp4", 10.0, 25.0, str(output))
        assert result is False

    @patch("viral_clip_extractor.extractors.clip_extractor.subprocess.run")
    @patch(
        "viral_clip_extractor.extractors.clip_extractor.ClipExtractor._get_video_duration",
        return_value=120.0,
    )
    def test_output_too_small_returns_false(
        self, mock_dur: MagicMock, mock_run: MagicMock,
        mock_vf: MagicMock, tmp_path: Path,
    ) -> None:
        output = tmp_path / "out.mp4"

        def side_effect(*args, **kwargs):
            output.write_bytes(b"\x00" * 100)  # < 10KB
            return MagicMock(returncode=0, stderr="")

        mock_run.side_effect = side_effect
        ext = self._make_extractor()
        result = ext.extract_clip("/fake/video.mp4", 10.0, 25.0, str(output))
        assert result is False

    @patch("viral_clip_extractor.extractors.clip_extractor.subprocess.run")
    @patch(
        "viral_clip_extractor.extractors.clip_extractor.ClipExtractor._get_video_duration",
        return_value=120.0,
    )
    def test_context_padding_applied(
        self, mock_dur: MagicMock, mock_run: MagicMock,
        mock_vf: MagicMock, tmp_path: Path,
    ) -> None:
        output = tmp_path / "out.mp4"

        def side_effect(*args, **kwargs):
            output.write_bytes(b"\x00" * 20000)
            return MagicMock(returncode=0, stderr="")

        mock_run.side_effect = side_effect

        ext = ClipExtractor(context_padding=3.0)
        ext.extract_clip("/fake/video.mp4", 10.0, 20.0, str(output))

        # Check that -ss and -t reflect padding: start=7.0, duration=16.0
        call_args = mock_run.call_args[0][0]
        ss_idx = call_args.index("-ss")
        t_idx = call_args.index("-t")
        assert float(call_args[ss_idx + 1]) == pytest.approx(7.0)
        assert float(call_args[t_idx + 1]) == pytest.approx(16.0)

    @patch("viral_clip_extractor.extractors.clip_extractor.subprocess.run")
    @patch(
        "viral_clip_extractor.extractors.clip_extractor.ClipExtractor._get_video_duration",
        return_value=12.0,
    )
    def test_padding_clamped_to_video_bounds(
        self, mock_dur: MagicMock, mock_run: MagicMock,
        mock_vf: MagicMock, tmp_path: Path,
    ) -> None:
        output = tmp_path / "out.mp4"

        def side_effect(*args, **kwargs):
            output.write_bytes(b"\x00" * 20000)
            return MagicMock(returncode=0, stderr="")

        mock_run.side_effect = side_effect

        ext = ClipExtractor(context_padding=5.0)
        ext.extract_clip("/fake/video.mp4", 1.0, 11.0, str(output))

        call_args = mock_run.call_args[0][0]
        ss_idx = call_args.index("-ss")
        t_idx = call_args.index("-t")
        # Padded start clamped to 0, padded end clamped to 12
        assert float(call_args[ss_idx + 1]) == pytest.approx(0.0)
        assert float(call_args[t_idx + 1]) == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# ClipExtractor — batch_extract
# ---------------------------------------------------------------------------

class TestClipExtractorBatchExtract:
    """Tests for ClipExtractor.batch_extract."""

    @patch.object(ClipExtractor, "extract_clip", return_value=True)
    def test_batch_creates_files(
        self, mock_extract: MagicMock, tmp_path: Path,
    ) -> None:
        segments = [
            SceneSegment(start_time=0.0, end_time=10.0, scene_index=0),
            SceneSegment(start_time=15.0, end_time=30.0, scene_index=1),
        ]
        scores = [85.0, 72.0]
        ext = ClipExtractor()

        paths = ext.batch_extract(
            "/fake/video.mp4", segments, str(tmp_path), scores,
        )
        assert len(paths) == 2
        assert "clip_01_score85.mp4" in paths[0]
        assert "clip_02_score72.mp4" in paths[1]

    @patch.object(ClipExtractor, "extract_clip", side_effect=[True, False, True])
    def test_batch_skips_failures(
        self, mock_extract: MagicMock, tmp_path: Path,
    ) -> None:
        segments = [
            SceneSegment(start_time=0.0, end_time=10.0, scene_index=0),
            SceneSegment(start_time=10.0, end_time=20.0, scene_index=1),
            SceneSegment(start_time=20.0, end_time=30.0, scene_index=2),
        ]
        ext = ClipExtractor()
        paths = ext.batch_extract("/fake/video.mp4", segments, str(tmp_path))
        assert len(paths) == 2

    @patch.object(ClipExtractor, "extract_clip", return_value=True)
    def test_batch_empty_segments(
        self, mock_extract: MagicMock, tmp_path: Path,
    ) -> None:
        ext = ClipExtractor()
        paths = ext.batch_extract("/fake/video.mp4", [], str(tmp_path))
        assert paths == []
        assert not mock_extract.called


# ---------------------------------------------------------------------------
# ClipExtractor — extract_batch (backward compat)
# ---------------------------------------------------------------------------

class TestClipExtractorExtractBatch:
    """Tests for the backward-compatible extract_batch method."""

    @patch.object(ClipExtractor, "extract_clip", return_value=True)
    def test_extract_batch_with_clip_data(
        self, mock_extract: MagicMock, tmp_path: Path,
    ) -> None:
        clips = [
            ClipData(
                scene=SceneSegment(start_time=5.0, end_time=20.0, scene_index=0),
                audio=AudioFeatures(
                    audio_peak_score=0.5, high_freq_score=0.3,
                    dynamic_range=0.2, zcr_score=0.1,
                ),
                visual=VisualFeatures(
                    motion_score=10.0, face_presence=0.5,
                    visual_interest=50.0, composition_score=5.0,
                ),
                semantic=SemanticFeatures(
                    emotional_intensity=5.0, narrative_interest=5.0,
                    hook_potential=5.0, asmr_quality=5.0,
                    visual_appeal=5.0, uniqueness=5.0,
                ),
                virality=ViralityScore(total_score=88.5),
            ),
        ]
        config = PipelineConfig(output_dir=str(tmp_path))
        ext = ClipExtractor(config=config)
        paths = ext.extract_batch("/fake/video.mp4", clips)
        assert len(paths) == 1
        assert mock_extract.called
