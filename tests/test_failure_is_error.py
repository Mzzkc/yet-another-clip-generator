"""
Tests verifying the failure-is-error contract.

Every feature is mandatory. If semantic analysis fails, the clip fails.
If caption generation fails, the clip fails. No silent fallbacks.
"""

from unittest.mock import MagicMock, patch

import pytest

from viral_clip_extractor.models import PipelineConfig


# ---------------------------------------------------------------------------
# Semantic failure raises
# ---------------------------------------------------------------------------

class TestSemanticFailureRaises:
    """Verify semantic analysis raises on Ollama failure."""

    def test_semantic_failure_raises(self):
        """SemanticAnalyzer.analyze_segment raises RuntimeError on Ollama 500."""
        from viral_clip_extractor.core.semantic_analyzer import SemanticAnalyzer

        analyzer = SemanticAnalyzer(
            model="test-model", ollama_host="http://localhost:11434"
        )

        # Mock frame extraction to return valid frames
        with patch.object(
            analyzer, "_extract_frames_base64", return_value=["base64data"]
        ):
            # Mock the session's post to return 500 on every attempt
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            mock_resp.text = "Internal Server Error"
            with patch.object(analyzer._session, "post", return_value=mock_resp):
                with pytest.raises(RuntimeError):
                    analyzer.analyze_segment(
                        "/fake/video.mp4", 0.0, 15.0, title="Test"
                    )


# ---------------------------------------------------------------------------
# Caption failure raises
# ---------------------------------------------------------------------------

class TestCaptionFailureRaises:
    """Verify caption generation raises on failure."""

    def test_caption_failure_raises(self):
        """OllamaVideoAnalyzer.analyze_video raises RuntimeError on failure."""
        from viral_clip_extractor.caption_generator import OllamaVideoAnalyzer

        analyzer = OllamaVideoAnalyzer(
            model="test-model", ollama_host="http://localhost:11434"
        )

        # Mock frame extraction to return empty list → should raise
        with patch.object(analyzer, "_extract_frames_base64", return_value=[]):
            with pytest.raises(RuntimeError, match="Could not extract any frames"):
                analyzer.analyze_video("/fake/clip.mp4", "Test Title")


# ---------------------------------------------------------------------------
# Caption failure deletes clip and thumbnail
# ---------------------------------------------------------------------------

class TestCaptionFailureDeletesClip:
    """Verify caption failure deletes the clip and its thumbnail."""

    def test_caption_failure_deletes_clip_and_thumbnail(self, tmp_path):
        """Pipeline._generate_captions deletes clip + thumbnail on failure."""
        from viral_clip_extractor.models import (
            AudioFeatures, ClipData, SceneSegment, SemanticFeatures,
            VisualFeatures, ViralityScore,
        )
        from viral_clip_extractor.pipeline import ViralClipPipeline

        # Create fake clip and thumbnail files
        clip_file = tmp_path / "clip_01.mp4"
        clip_file.write_text("fake video")
        thumb_file = tmp_path / "clip_01_thumb.jpg"
        thumb_file.write_text("fake thumb")

        clip = ClipData(
            scene=SceneSegment(start_time=0.0, end_time=10.0, scene_index=0),
            audio=AudioFeatures(0.0, 0.0, 0.0, 0.0),
            visual=VisualFeatures(0.0, 0.0, 0.0, 0.0),
            semantic=SemanticFeatures(5.0, 5.0, 5.0, 5.0, 5.0, 5.0),
            virality=ViralityScore(total_score=80.0),
            output_path=str(clip_file),
            thumbnail_path=str(thumb_file),
        )

        pipeline = ViralClipPipeline()
        errors: list[str] = []

        # Mock caption analyzer to always raise
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_video.side_effect = RuntimeError("LLM down")
        pipeline._caption_analyzer = mock_analyzer

        pipeline._generate_captions([clip], "Test", errors)

        # Clip and thumbnail should be deleted
        assert not clip_file.exists()
        assert not thumb_file.exists()
        assert clip.output_path is None
        assert clip.thumbnail_path is None
        assert len(errors) == 1
        assert "Caption generation failed" in errors[0]


# ---------------------------------------------------------------------------
# Pipeline is transcript-first
# ---------------------------------------------------------------------------

class TestPipelineIsTranscriptFirst:
    """Verify pipeline uses TranscriptSegmenter, not SceneDetector."""

    def test_pipeline_is_transcript_first(self):
        """Pipeline._get_transcript_segmenter exists and is used."""
        from viral_clip_extractor.pipeline import ViralClipPipeline

        pipeline = ViralClipPipeline()

        # Verify the pipeline has transcript segmenter accessor
        assert hasattr(pipeline, "_get_transcript_segmenter")
        assert hasattr(pipeline, "_get_subtitle_burner")

        # Verify pipeline creates TranscriptSegmenter
        segmenter = pipeline._get_transcript_segmenter()
        from viral_clip_extractor.transcript_segmenter import TranscriptSegmenter
        assert isinstance(segmenter, TranscriptSegmenter)

    def test_pipeline_has_no_feature_toggles(self):
        """PipelineConfig has no enable_semantic, enable_captions, vertical_crop."""
        config = PipelineConfig()
        assert not hasattr(config, "enable_semantic")
        assert not hasattr(config, "enable_captions")
        assert not hasattr(config, "vertical_crop")

    def test_pipeline_has_whisper_model(self):
        """PipelineConfig has whisper_model field."""
        config = PipelineConfig()
        assert hasattr(config, "whisper_model")
        assert config.whisper_model == "small"

    def test_clip_extractor_no_vertical_toggle(self):
        """ClipExtractor has no vertical parameter."""
        from viral_clip_extractor.extractors.clip_extractor import ClipExtractor
        import inspect

        sig = inspect.signature(ClipExtractor.__init__)
        assert "vertical" not in sig.parameters
