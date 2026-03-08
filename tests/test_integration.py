"""
Integration tests for the Viral Clip Extractor pipeline.

These tests exercise real code paths with real video fixtures — no mocking
of core libraries (OpenCV, librosa, scenedetect, FFmpeg). Fixtures live at
/tmp/vce_test_fixtures/ and are created by conftest.py or the preprocessing stage.

Note: The pipeline is now transcript-first (Whisper + Ollama segmentation).
Tests that use synthetic videos (no speech) will fail at transcription.
These tests either mock the transcript segmenter or verify the error behavior.
"""

import csv
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from viral_clip_extractor.cli import main as cli_main
from viral_clip_extractor.models import (
    AudioFeatures,
    PipelineConfig,
    ProcessingResult,
    SceneSegment,
    SegmentBoundary,
    VisualFeatures,
    WordTimestamp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_cli(args: list[str]) -> int:
    """Run the CLI and return the exit code."""
    return cli_main(args)


def _make_config(**overrides) -> PipelineConfig:
    """Create a PipelineConfig with sensible test defaults."""
    defaults = {
        "min_virality_score": 0.0,
    }
    defaults.update(overrides)
    return PipelineConfig(**defaults)


# ---------------------------------------------------------------------------
# Scene detection with real video
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSceneDetectionReal:
    def test_scene_detection_real_video(self, synthetic_1s):
        """SceneDetector.detect_scenes() with no mocks on a real video."""
        from viral_clip_extractor.core.scene_detector import SceneDetector

        detector = SceneDetector(threshold=3.0, min_scene_len=0.5, max_scene_len=60.0)
        scenes = detector.detect_scenes(str(synthetic_1s))

        assert isinstance(scenes, list)
        for s in scenes:
            assert isinstance(s, SceneSegment)
            assert s.start_time >= 0
            assert s.end_time > s.start_time

    def test_scene_detection_rickroll(self, rickroll_30s):
        """SceneDetector handles a real 30s video clip."""
        from viral_clip_extractor.core.scene_detector import SceneDetector

        detector = SceneDetector(threshold=3.0, min_scene_len=3.0, max_scene_len=60.0)
        scenes = detector.detect_scenes(str(rickroll_30s))

        assert len(scenes) >= 1
        total_coverage = sum(s.duration for s in scenes)
        assert total_coverage > 25.0  # should cover most of the 30s


# ---------------------------------------------------------------------------
# Audio analysis with real video
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAudioAnalysisReal:
    def test_audio_analysis_real_video(self, synthetic_1s):
        """AudioAnalyzer.analyze_segment() with no mocks on real video."""
        from viral_clip_extractor.core.audio_analyzer import AudioAnalyzer

        analyzer = AudioAnalyzer()
        result = analyzer.analyze_segment(str(synthetic_1s), 0.0, 1.0)

        assert isinstance(result, AudioFeatures)
        # Synthetic silent audio — values should be near zero but not crash
        assert result.audio_peak_score >= 0.0
        assert result.high_freq_score >= 0.0

    def test_audio_analysis_no_audio_track(self, synthetic_noaudio):
        """AudioAnalyzer raises RuntimeError for video with no audio track."""
        from viral_clip_extractor.core.audio_analyzer import AudioAnalyzer

        analyzer = AudioAnalyzer()
        with pytest.raises(RuntimeError, match="Failed to load audio"):
            analyzer.analyze_segment(str(synthetic_noaudio), 0.0, 5.0)


# ---------------------------------------------------------------------------
# Visual analysis with real video
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestVisualAnalysisReal:
    def test_visual_analysis_real_video(self, synthetic_1s):
        """VisualAnalyzer.analyze_segment() with no mocks."""
        from viral_clip_extractor.core.visual_analyzer import VisualAnalyzer

        analyzer = VisualAnalyzer(PipelineConfig())
        result = analyzer.analyze_segment(str(synthetic_1s), 0.0, 1.0)

        assert isinstance(result, VisualFeatures)
        assert 0.0 <= result.motion_score <= 1.0
        assert 0.0 <= result.face_presence <= 1.0
        assert 0.0 <= result.visual_interest <= 1.0
        assert 0.0 <= result.composition_score <= 1.0

    def test_visual_analysis_singleframe(self, synthetic_singleframe):
        """VisualAnalyzer handles near-zero duration without crash."""
        from viral_clip_extractor.core.visual_analyzer import VisualAnalyzer

        analyzer = VisualAnalyzer(PipelineConfig())
        result = analyzer.analyze_segment(str(synthetic_singleframe), 0.0, 0.04)

        assert isinstance(result, VisualFeatures)
        # Just verify it doesn't crash — values may be zero for single frame
        assert result.motion_score >= 0.0


# ---------------------------------------------------------------------------
# Scoring with real features
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestScoringWithRealFeatures:
    def test_scoring_with_real_features(self, synthetic_1s):
        """Full chain: scene detect -> audio -> visual -> scorer."""
        from viral_clip_extractor.core.audio_analyzer import AudioAnalyzer
        from viral_clip_extractor.core.scene_detector import SceneDetector
        from viral_clip_extractor.core.virality_scorer import ViralityScorer
        from viral_clip_extractor.core.visual_analyzer import VisualAnalyzer

        detector = SceneDetector(threshold=3.0, min_scene_len=0.5, max_scene_len=60.0)
        scenes = detector.detect_scenes(str(synthetic_1s))
        assert len(scenes) >= 1

        scene = scenes[0]
        audio_analyzer = AudioAnalyzer()
        audio = audio_analyzer.analyze_segment(
            str(synthetic_1s), scene.start_time, scene.end_time
        )

        visual_analyzer = VisualAnalyzer(PipelineConfig())
        visual = visual_analyzer.analyze_segment(
            str(synthetic_1s), scene.start_time, scene.end_time
        )

        scorer = ViralityScorer()
        score = scorer.calculate_score(
            audio=audio, visual=visual, semantic=None, duration=scene.duration
        )

        assert score.total_score >= 0
        assert "audio_peaks" in score.component_scores
        assert "motion" in score.component_scores
        assert "visual" in score.component_scores
        assert "duration" in score.component_scores


# ---------------------------------------------------------------------------
# Clip extraction with real FFmpeg
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestClipExtractionReal:
    def test_clip_extraction_real_video(self, rickroll_30s):
        """ClipExtractor.extract_clip() with real FFmpeg."""
        from viral_clip_extractor.extractors.clip_extractor import ClipExtractor

        config = _make_config()
        extractor = ClipExtractor(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_clip.mp4")
            success = extractor.extract_clip(
                str(rickroll_30s), 0.0, 10.0, output_path
            )

            assert success is True
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 10_000  # > 10 KB


# ---------------------------------------------------------------------------
# Pipeline end-to-end tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPipelineE2E:
    def test_pipeline_process_minimal(self, synthetic_1s):
        """Pipeline returns error for synthetic video with no speech (transcript-first)."""
        from viral_clip_extractor.pipeline import ViralClipPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(output_dir=tmpdir)
            pipeline = ViralClipPipeline(config=config)

            result = pipeline.process_video(
                video_path=str(synthetic_1s),
                title="Test Synthetic",
                top_n=2,
                min_score=0.0,
            )

            assert isinstance(result, ProcessingResult)
            assert result.video_path == str(synthetic_1s)
            # Transcript-first pipeline fails on synthetic silence
            assert len(result.errors) > 0
            assert result.clips == []

    @pytest.mark.slow
    def test_pipeline_rickroll_basic(self, rickroll_30s):
        """Full pipeline on 30s rickroll with mocked transcript segmenter."""
        from viral_clip_extractor.pipeline import ViralClipPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(output_dir=tmpdir)
            pipeline = ViralClipPipeline(config=config)

            # Mock the transcript segmenter to avoid Whisper model dependency
            mock_segmenter = MagicMock()
            mock_segmenter.full_transcribe.return_value = [
                WordTimestamp(word="We're", start=18.0, end=19.0),
                WordTimestamp(word="no", start=19.0, end=19.3),
                WordTimestamp(word="strangers", start=19.3, end=19.7),
                WordTimestamp(word="to", start=19.7, end=20.2),
                WordTimestamp(word="love", start=20.2, end=22.1),
            ]
            mock_segmenter.segment_by_content.return_value = [
                SegmentBoundary(
                    start_time=0.0, end_time=30.0,
                    hook_summary="Classic intro", segment_type="hook",
                ),
            ]
            mock_segmenter.refine_boundaries.return_value = [
                SceneSegment(start_time=0.0, end_time=30.0, scene_index=0),
            ]
            pipeline._transcript_segmenter = mock_segmenter

            result = pipeline.process_video(
                video_path=str(rickroll_30s),
                title="Rick Astley - Never Gonna Give You Up",
                top_n=2,
                min_score=0.0,
            )

            assert isinstance(result, ProcessingResult)
            assert result.total_scenes >= 1

            # CSV should exist
            csv_path = os.path.join(tmpdir, "clips_report.csv")
            assert os.path.exists(csv_path)

    def test_process_nonexistent_file(self):
        """Processing a nonexistent file returns error result."""
        from viral_clip_extractor.pipeline import ViralClipPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(output_dir=tmpdir)
            pipeline = ViralClipPipeline(config=config)

            result = pipeline.process_video(
                video_path="/nonexistent_video_file.mp4",
                title="test",
                top_n=1,
                min_score=0.0,
            )

            assert isinstance(result, ProcessingResult)
            assert len(result.errors) > 0
            assert result.clips == []


# ---------------------------------------------------------------------------
# CSV output structure
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCSVOutput:
    def test_csv_has_expected_columns(self, synthetic_1s):
        """CSV report contains all expected column headers (with mocked segmenter)."""
        from viral_clip_extractor.pipeline import ViralClipPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(output_dir=tmpdir)
            pipeline = ViralClipPipeline(config=config)

            # Mock transcript segmenter — synthetic_1s has no speech
            mock_segmenter = MagicMock()
            mock_segmenter.full_transcribe.return_value = [
                WordTimestamp(word="test", start=0.0, end=0.5),
            ]
            mock_segmenter.segment_by_content.return_value = [
                SegmentBoundary(
                    start_time=0.0, end_time=1.0,
                    hook_summary="Test", segment_type="hook",
                ),
            ]
            mock_segmenter.refine_boundaries.return_value = [
                SceneSegment(start_time=0.0, end_time=1.0, scene_index=0),
            ]
            pipeline._transcript_segmenter = mock_segmenter

            pipeline.process_video(
                video_path=str(synthetic_1s),
                title="CSV Test",
                top_n=2,
                min_score=0.0,
            )

            csv_path = os.path.join(tmpdir, "clips_report.csv")
            assert os.path.exists(csv_path)

            with open(csv_path) as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames

            expected_cols = [
                "Clip_Filename", "Start_Time", "End_Time", "Duration",
                "Virality_Score", "Hook", "Description", "Hashtags",
                "Full_Caption", "Category", "Audio_Peak", "Motion_Score",
                "Face_Presence", "ASMR_Quality", "Processing_Timestamp",
            ]
            for col in expected_cols:
                assert col in headers, f"Missing CSV column: {col}"


# ---------------------------------------------------------------------------
# CLI check command
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCheckCommand:
    def test_check_command_runs(self):
        """check subcommand executes and returns exit code."""
        exit_code = _run_cli(["check"])
        # On a system with deps installed, should be 0
        assert exit_code == 0


