"""
Adversarial tests for the Viral Clip Extractor.

Edge-case inputs, malformed data, and resource cleanup verification.
These tests verify graceful degradation rather than correct output.
"""

import os
import shutil
import tempfile

import pytest

from viral_clip_extractor.models import (
    AudioFeatures,
    PipelineConfig,
    ProcessingResult,
    VisualFeatures,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> PipelineConfig:
    defaults = {
        "min_virality_score": 0.0,
    }
    defaults.update(overrides)
    return PipelineConfig(**defaults)


# ---------------------------------------------------------------------------
# Invalid file inputs
# ---------------------------------------------------------------------------

@pytest.mark.adversarial
class TestInvalidInputs:
    def test_zero_byte_file(self):
        """Pipeline handles a zero-byte file gracefully."""
        from viral_clip_extractor.pipeline import ViralClipPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            empty_file = os.path.join(tmpdir, "empty.mp4")
            with open(empty_file, "wb") as f:
                pass  # zero bytes

            config = _make_config(output_dir=tmpdir)
            pipeline = ViralClipPipeline(config=config)

            result = pipeline.process_video(
                video_path=empty_file,
                title="Empty File",
                top_n=1,
                min_score=0.0,
            )

            assert isinstance(result, ProcessingResult)
            assert len(result.errors) > 0
            assert result.clips == []

    def test_corrupt_file(self):
        """Pipeline handles random bytes as .mp4 gracefully."""
        from viral_clip_extractor.pipeline import ViralClipPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            corrupt_file = os.path.join(tmpdir, "corrupt.mp4")
            with open(corrupt_file, "wb") as f:
                f.write(os.urandom(1024))  # random bytes

            config = _make_config(output_dir=tmpdir)
            pipeline = ViralClipPipeline(config=config)

            result = pipeline.process_video(
                video_path=corrupt_file,
                title="Corrupt File",
                top_n=1,
                min_score=0.0,
            )

            assert isinstance(result, ProcessingResult)
            assert len(result.errors) > 0
            assert result.clips == []

    def test_nonexistent_file(self):
        """Pipeline returns an error for nonexistent paths."""
        from viral_clip_extractor.pipeline import ViralClipPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(output_dir=tmpdir)
            pipeline = ViralClipPipeline(config=config)

            result = pipeline.process_video(
                video_path="/absolutely/nonexistent/path/video.mp4",
                title="Ghost",
                top_n=1,
                min_score=0.0,
            )

            assert isinstance(result, ProcessingResult)
            assert len(result.errors) > 0
            assert "not found" in result.errors[0].lower() or "Video not found" in result.errors[0]

    def test_directory_as_video_path(self):
        """Pipeline doesn't crash when given a directory instead of a file."""
        from viral_clip_extractor.pipeline import ViralClipPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(output_dir=tmpdir)
            pipeline = ViralClipPipeline(config=config)

            result = pipeline.process_video(
                video_path=tmpdir,  # a directory, not a file
                title="Dir Input",
                top_n=1,
                min_score=0.0,
            )

            assert isinstance(result, ProcessingResult)
            assert len(result.errors) > 0


# ---------------------------------------------------------------------------
# Audio edge cases
# ---------------------------------------------------------------------------

@pytest.mark.adversarial
class TestAudioEdgeCases:
    def test_no_audio_track_full_pipeline(self, synthetic_noaudio):
        """Pipeline completes with video that has no audio track."""
        from viral_clip_extractor.pipeline import ViralClipPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(output_dir=tmpdir)
            pipeline = ViralClipPipeline(config=config)

            result = pipeline.process_video(
                video_path=str(synthetic_noaudio),
                title="No Audio",
                top_n=2,
                min_score=0.0,
            )

            assert isinstance(result, ProcessingResult)
            # Pipeline should complete without crashing
            # Visual features should still be populated for any clips
            for clip in result.clips:
                assert isinstance(clip.visual, VisualFeatures)
                assert isinstance(clip.audio, AudioFeatures)

    def test_audio_analyzer_negative_times(self):
        """AudioAnalyzer raises RuntimeError for unloadable audio."""
        from viral_clip_extractor.core.audio_analyzer import AudioAnalyzer

        analyzer = AudioAnalyzer()
        with pytest.raises(RuntimeError, match="Failed to load audio"):
            analyzer.analyze_segment("/dev/null", -5.0, -1.0)


# ---------------------------------------------------------------------------
# Single-frame edge case
# ---------------------------------------------------------------------------

@pytest.mark.adversarial
class TestSingleFrameEdgeCases:
    def test_single_frame_video_pipeline(self, synthetic_singleframe):
        """Pipeline handles a single-frame video without crash."""
        from viral_clip_extractor.pipeline import ViralClipPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(output_dir=tmpdir)
            pipeline = ViralClipPipeline(config=config)

            result = pipeline.process_video(
                video_path=str(synthetic_singleframe),
                title="Single Frame",
                top_n=2,
                min_score=0.0,
            )

            assert isinstance(result, ProcessingResult)
            # May produce 0 clips (acceptable for near-zero duration)


# ---------------------------------------------------------------------------
# Title edge cases
# ---------------------------------------------------------------------------

@pytest.mark.adversarial
class TestTitleEdgeCases:
    def test_unicode_in_title(self, synthetic_1s):
        """Unicode characters in title don't crash the pipeline."""
        from viral_clip_extractor.pipeline import ViralClipPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(output_dir=tmpdir)
            pipeline = ViralClipPipeline(config=config)

            result = pipeline.process_video(
                video_path=str(synthetic_1s),
                title="T\u00ebst V\u00efd\u00e9o \U0001f3ac",
                top_n=2,
                min_score=0.0,
            )

            assert isinstance(result, ProcessingResult)
            assert result.video_title == "T\u00ebst V\u00efd\u00e9o \U0001f3ac"

    def test_empty_title(self, synthetic_1s):
        """Empty title is handled by falling back to filename."""
        from viral_clip_extractor.pipeline import ViralClipPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(output_dir=tmpdir)
            pipeline = ViralClipPipeline(config=config)

            result = pipeline.process_video(
                video_path=str(synthetic_1s),
                title="",
                top_n=2,
                min_score=0.0,
            )

            assert isinstance(result, ProcessingResult)
            # Title should be auto-detected from filename
            assert len(result.video_title) > 0

    def test_very_long_title(self, synthetic_1s):
        """Very long title doesn't crash the pipeline."""
        from viral_clip_extractor.pipeline import ViralClipPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(output_dir=tmpdir)
            pipeline = ViralClipPipeline(config=config)

            long_title = "A" * 5000
            result = pipeline.process_video(
                video_path=str(synthetic_1s),
                title=long_title,
                top_n=2,
                min_score=0.0,
            )

            assert isinstance(result, ProcessingResult)


# ---------------------------------------------------------------------------
# Output directory edge cases
# ---------------------------------------------------------------------------

@pytest.mark.adversarial
class TestOutputDirEdgeCases:
    def test_output_dir_does_not_exist(self, synthetic_1s):
        """Pipeline with no-speech video returns error (transcript-first)."""
        from viral_clip_extractor.pipeline import ViralClipPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "nested", "deep", "output")
            config = _make_config(output_dir=new_dir)
            pipeline = ViralClipPipeline(config=config)

            result = pipeline.process_video(
                video_path=str(synthetic_1s),
                title="Dir Create Test",
                top_n=2,
                min_score=0.0,
            )

            assert isinstance(result, ProcessingResult)
            # Transcript-first pipeline fails on synthetic silence
            assert len(result.errors) > 0


# ---------------------------------------------------------------------------
# Config edge cases
# ---------------------------------------------------------------------------

@pytest.mark.adversarial
class TestConfigEdgeCases:
    def test_min_score_100(self, synthetic_1s):
        """min_score=100 yields zero clips (nothing scores 100)."""
        from viral_clip_extractor.pipeline import ViralClipPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(output_dir=tmpdir)
            pipeline = ViralClipPipeline(config=config)

            result = pipeline.process_video(
                video_path=str(synthetic_1s),
                title="High Bar",
                top_n=10,
                min_score=100.0,
            )

            assert isinstance(result, ProcessingResult)
            assert result.clips == []

    def test_top_n_zero(self, synthetic_1s):
        """top_n=0 yields zero clips."""
        from viral_clip_extractor.pipeline import ViralClipPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(output_dir=tmpdir)
            pipeline = ViralClipPipeline(config=config)

            result = pipeline.process_video(
                video_path=str(synthetic_1s),
                title="Zero Clips",
                top_n=0,
                min_score=0.0,
            )

            assert isinstance(result, ProcessingResult)
            assert result.clips == []

    def test_negative_min_score(self, synthetic_1s):
        """Negative min_score doesn't crash — all clips pass."""
        from viral_clip_extractor.pipeline import ViralClipPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(output_dir=tmpdir)
            pipeline = ViralClipPipeline(config=config)

            result = pipeline.process_video(
                video_path=str(synthetic_1s),
                title="Negative Score",
                top_n=10,
                min_score=-50.0,
            )

            assert isinstance(result, ProcessingResult)


# ---------------------------------------------------------------------------
# Scorer edge cases
# ---------------------------------------------------------------------------

@pytest.mark.adversarial
class TestScorerEdgeCases:
    def test_scorer_all_zero_inputs(self):
        """Scorer handles all-zero features without crashing."""
        from viral_clip_extractor.core.virality_scorer import ViralityScorer

        scorer = ViralityScorer()
        audio = AudioFeatures(
            audio_peak_score=0.0, high_freq_score=0.0,
            dynamic_range=0.0, zcr_score=0.0,
        )
        visual = VisualFeatures(
            motion_score=0.0, face_presence=0.0,
            visual_interest=0.0, composition_score=0.0,
        )

        score = scorer.calculate_score(audio=audio, visual=visual, semantic=None, duration=10.0)
        assert score.total_score >= 0
        assert score.total_score <= 100

    def test_scorer_extreme_values(self):
        """Scorer clamps extreme input values properly."""
        from viral_clip_extractor.core.virality_scorer import ViralityScorer

        scorer = ViralityScorer()
        audio = AudioFeatures(
            audio_peak_score=999.0, high_freq_score=999.0,
            dynamic_range=999.0, zcr_score=999.0,
        )
        visual = VisualFeatures(
            motion_score=999.0, face_presence=999.0,
            visual_interest=999.0, composition_score=999.0,
        )

        score = scorer.calculate_score(audio=audio, visual=visual, semantic=None, duration=10.0)
        assert 0 <= score.total_score <= 100

    def test_scorer_zero_duration(self):
        """Scorer handles zero duration gracefully."""
        from viral_clip_extractor.core.virality_scorer import ViralityScorer

        scorer = ViralityScorer()
        audio = AudioFeatures(
            audio_peak_score=0.5, high_freq_score=0.5,
            dynamic_range=0.1, zcr_score=0.1,
        )
        visual = VisualFeatures(
            motion_score=0.5, face_presence=0.3,
            visual_interest=0.4, composition_score=0.5,
        )

        score = scorer.calculate_score(audio=audio, visual=visual, semantic=None, duration=0.0)
        assert 0 <= score.total_score <= 100
        assert score.component_scores["duration"] == 0.0


# ---------------------------------------------------------------------------
# CLI edge cases
# ---------------------------------------------------------------------------

@pytest.mark.adversarial
class TestCLIEdgeCases:
    def test_cli_no_args(self):
        """CLI with no arguments returns non-zero exit code."""
        from viral_clip_extractor.cli import main as cli_main
        exit_code = cli_main([])
        assert exit_code == 1

    def test_cli_invalid_command(self):
        """CLI with invalid subcommand exits non-zero."""
        from viral_clip_extractor.cli import main as cli_main
        # argparse will exit with SystemExit for unrecognized commands
        with pytest.raises(SystemExit):
            cli_main(["nonexistent_command"])

    def test_cli_process_missing_required_args(self):
        """process subcommand without --video fails."""
        from viral_clip_extractor.cli import main as cli_main
        with pytest.raises(SystemExit):
            cli_main(["process"])

    def test_cli_help_exits_zero(self):
        """--help exits with code 0."""
        from viral_clip_extractor.cli import main as cli_main
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["--help"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Scene detector edge cases
# ---------------------------------------------------------------------------

@pytest.mark.adversarial
class TestSceneDetectorEdgeCases:
    def test_scene_detector_nonexistent_file(self):
        """SceneDetector raises for nonexistent file."""
        from viral_clip_extractor.core.scene_detector import SceneDetector

        detector = SceneDetector()
        with pytest.raises(FileNotFoundError):
            detector.detect_scenes("/nonexistent/path/video.mp4")

    def test_scene_detector_zero_byte_file(self):
        """SceneDetector raises for zero-byte file."""
        from viral_clip_extractor.core.scene_detector import SceneDetector

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name

        try:
            detector = SceneDetector()
            with pytest.raises(RuntimeError):
                detector.detect_scenes(path)
        finally:
            os.unlink(path)
