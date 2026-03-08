"""
Integration tests for the YACG pipeline.

Tests pipeline initialization, config loading, CLI argument parsing,
YouTubeDownloader URL parsing, CSV generation, and mock-based pipeline
orchestration with the transcript-first flow.
"""

import csv
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from yacg.models import (
    AudioFeatures,
    CaptionData,
    ClipData,
    ContentProfile,
    PipelineConfig,
    ProcessingResult,
    SceneSegment,
    SegmentBoundary,
    SemanticFeatures,
    ViralityScore,
    VisualFeatures,
    WordTimestamp,
)


# ------------------------------------------------------------------
# Test PipelineConfig loading from INI
# ------------------------------------------------------------------


class TestConfigLoading:
    """Test loading PipelineConfig from INI files."""

    def test_load_default_config(self):
        """Loading with no path returns defaults."""
        from yacg.utils.config import load_config

        config = load_config(None)
        assert isinstance(config, PipelineConfig)
        assert config.model_name == "qwen2.5-vl:7b"
        assert config.scene_threshold == 3.0
        assert config.min_scene_len == 7.0
        assert config.top_n_clips == 10
        assert config.whisper_model == "small"

    def test_load_missing_file_returns_defaults(self):
        """Loading a non-existent file returns defaults gracefully."""
        from yacg.utils.config import load_config

        config = load_config("/nonexistent/path.ini")
        assert isinstance(config, PipelineConfig)
        assert config.model_name == "qwen2.5-vl:7b"

    def test_load_custom_config(self, tmp_path):
        """Loading a valid INI file overrides defaults."""
        from yacg.utils.config import load_config

        ini_content = """\
[Model]
model_name = test-model:latest
ollama_host = http://example.com:1234
whisper_model = small

[SceneDetection]
threshold = 5.0
min_scene_len = 10.0
max_scene_len = 45.0

[ClipSelection]
top_n_clips = 5
min_virality_score = 80.0
"""
        config_path = tmp_path / "test_config.ini"
        config_path.write_text(ini_content)

        config = load_config(str(config_path))
        assert config.model_name == "test-model:latest"
        assert config.ollama_host == "http://example.com:1234"
        assert config.scene_threshold == 5.0
        assert config.min_scene_len == 10.0
        assert config.top_n_clips == 5
        assert config.min_virality_score == 80.0
        assert config.whisper_model == "small"


# ------------------------------------------------------------------
# Test CLI argument parsing
# ------------------------------------------------------------------


class TestCLIParsing:
    """Test argparse-based CLI argument parsing."""

    def test_process_subcommand(self):
        """Parse 'process' subcommand with required args."""
        from yacg.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "process", "--video", "/path/to/video.mp4",
            "--title", "Test Video",
            "--output-dir", "./output",
            "--top-n", "15",
            "--min-score", "80",
        ])

        assert args.command == "process"
        assert args.video == "/path/to/video.mp4"
        assert args.title == "Test Video"
        assert args.output_dir == "./output"
        assert args.top_n == 15
        assert args.min_score == 80.0

    def test_youtube_subcommand(self):
        """Parse 'youtube' subcommand with URL."""
        from yacg.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "youtube", "--url", "https://www.youtube.com/watch?v=test12345ab",
            "--top-n", "5",
        ])

        assert args.command == "youtube"
        assert args.url == "https://www.youtube.com/watch?v=test12345ab"
        assert args.top_n == 5

    def test_batch_subcommand(self):
        """Parse 'batch' subcommand."""
        from yacg.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "batch", "--videos-dir", "/path/to/videos",
        ])

        assert args.command == "batch"
        assert args.videos_dir == "/path/to/videos"

    def test_check_subcommand(self):
        """Parse 'check' subcommand."""
        from yacg.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["check"])
        assert args.command == "check"

    def test_no_command_returns_error(self):
        """main() with no args returns error exit code."""
        from yacg.cli import main

        exit_code = main([])
        assert exit_code == 1

    def test_whisper_model_flag(self):
        """Parse --whisper-model flag."""
        from yacg.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "process", "--video", "test.mp4",
            "--whisper-model", "large-v3",
        ])

        assert args.whisper_model == "large-v3"


# ------------------------------------------------------------------
# Test YouTubeDownloader URL parsing
# ------------------------------------------------------------------


class TestYouTubeURLParsing:
    """Test YouTubeDownloader.extract_video_id with various URL formats."""

    def test_standard_url(self):
        from yacg.youtube_downloader import YouTubeDownloader

        dl = YouTubeDownloader.__new__(YouTubeDownloader)
        assert dl.extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_short_url(self):
        from yacg.youtube_downloader import YouTubeDownloader

        dl = YouTubeDownloader.__new__(YouTubeDownloader)
        assert dl.extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_shorts_url(self):
        from yacg.youtube_downloader import YouTubeDownloader

        dl = YouTubeDownloader.__new__(YouTubeDownloader)
        assert dl.extract_video_id("https://youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_embed_url(self):
        from yacg.youtube_downloader import YouTubeDownloader

        dl = YouTubeDownloader.__new__(YouTubeDownloader)
        assert dl.extract_video_id("https://youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_bare_id(self):
        from yacg.youtube_downloader import YouTubeDownloader

        dl = YouTubeDownloader.__new__(YouTubeDownloader)
        assert dl.extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_invalid_url(self):
        from yacg.youtube_downloader import YouTubeDownloader

        dl = YouTubeDownloader.__new__(YouTubeDownloader)
        assert dl.extract_video_id("not_a_valid_url") == ""


# ------------------------------------------------------------------
# Test Pipeline initialization
# ------------------------------------------------------------------


class TestPipelineInit:
    """Test pipeline instantiation and default config."""

    def test_default_init(self):
        """Pipeline initializes with default config."""
        from yacg.pipeline import ViralClipPipeline

        pipeline = ViralClipPipeline()
        assert pipeline.config is not None
        assert isinstance(pipeline.config, PipelineConfig)
        assert pipeline.config.model_name == "qwen2.5-vl:7b"

    def test_custom_config(self):
        """Pipeline respects custom config."""
        from yacg.pipeline import ViralClipPipeline

        config = PipelineConfig(
            model_name="custom:latest",
            top_n_clips=5,
        )
        pipeline = ViralClipPipeline(config=config)
        assert pipeline.config.model_name == "custom:latest"
        assert pipeline.config.top_n_clips == 5

    def test_process_video_missing_file(self):
        """Processing a non-existent file returns error result."""
        from yacg.pipeline import ViralClipPipeline

        pipeline = ViralClipPipeline()
        result = pipeline.process_video("/nonexistent/video.mp4")
        assert isinstance(result, ProcessingResult)
        assert len(result.errors) > 0
        assert "not found" in result.errors[0].lower()


# ------------------------------------------------------------------
# Test CSV generation
# ------------------------------------------------------------------


class TestCSVGeneration:
    """Test _generate_csv produces correct output."""

    def test_csv_columns(self, tmp_path):
        """CSV has the expected column headers."""
        from yacg.pipeline import ViralClipPipeline

        pipeline = ViralClipPipeline()

        clip = ClipData(
            scene=SceneSegment(start_time=10.0, end_time=25.0, scene_index=0),
            audio=AudioFeatures(
                audio_peak_score=0.8, high_freq_score=0.3,
                dynamic_range=0.2, zcr_score=0.1,
            ),
            visual=VisualFeatures(
                motion_score=0.5, face_presence=0.7,
                visual_interest=0.4, composition_score=0.6,
            ),
            semantic=SemanticFeatures(
                emotional_intensity=7.0, narrative_interest=6.0,
                hook_potential=8.0, asmr_quality=9.0,
                visual_appeal=7.5, uniqueness=6.5,
                description="Test clip",
            ),
            virality=ViralityScore(total_score=85.0),
            output_path="/tmp/clip_01.mp4",
            caption=CaptionData(
                hook="Watch this!",
                description="Amazing clip",
                hashtags=["#asmr", "#viral"],
                full_caption="Watch this!\n\nAmazing clip\n\n#asmr #viral",
                category="ASMR",
                virality_score=85,
            ),
        )

        result = ProcessingResult(
            video_path="/tmp/test.mp4",
            video_title="Test",
            clips=[clip],
            total_scenes=5,
            processing_time_seconds=10.0,
        )

        csv_path = str(tmp_path / "test_report.csv")
        pipeline._generate_csv(result, csv_path)

        assert os.path.exists(csv_path)

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]

        # Check expected columns exist
        expected_cols = [
            "Clip_Filename", "Start_Time", "End_Time", "Duration",
            "Virality_Score", "Hook", "Description", "Hashtags",
            "Full_Caption", "Category", "Audio_Peak", "Motion_Score",
            "Face_Presence", "Zero_Crossing_Rate", "Composition",
            "ASMR_Quality", "Processing_Timestamp",
        ]
        for col in expected_cols:
            assert col in row, f"Missing column: {col}"

        assert row["Hook"] == "Watch this!"
        assert row["Category"] == "ASMR"
        assert float(row["Virality_Score"]) == 85.0

    def test_csv_empty_clips(self, tmp_path):
        """CSV with no clips still has headers."""
        from yacg.pipeline import ViralClipPipeline

        pipeline = ViralClipPipeline()
        result = ProcessingResult(
            video_path="/tmp/test.mp4",
            video_title="Test",
            clips=[],
            total_scenes=0,
            processing_time_seconds=0.0,
        )

        csv_path = str(tmp_path / "empty_report.csv")
        pipeline._generate_csv(result, csv_path)

        assert os.path.exists(csv_path)
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 0


# ------------------------------------------------------------------
# Mock-based pipeline orchestration test (transcript-first flow)
# ------------------------------------------------------------------


class TestPipelineOrchestration:
    """Mock all components and verify transcript-first pipeline flow."""

    @patch("yacg.pipeline.ViralClipPipeline._get_caption_analyzer")
    @patch("yacg.pipeline.ViralClipPipeline._get_subtitle_burner")
    @patch("yacg.pipeline.ViralClipPipeline._get_clip_extractor")
    @patch("yacg.pipeline.ViralClipPipeline._get_virality_scorer")
    @patch("yacg.pipeline.ViralClipPipeline._get_semantic_analyzer")
    @patch("yacg.pipeline.ViralClipPipeline._get_visual_analyzer")
    @patch("yacg.pipeline.ViralClipPipeline._get_audio_analyzer")
    @patch("yacg.pipeline.ViralClipPipeline._get_transcript_segmenter")
    def test_transcript_first_pipeline(
        self,
        mock_segmenter,
        mock_audio,
        mock_visual,
        mock_semantic,
        mock_scorer,
        mock_extractor,
        mock_burner,
        mock_caption,
        tmp_path,
    ):
        """Verify pipeline uses transcript-first flow with all mandatory steps."""
        from yacg.pipeline import ViralClipPipeline

        # Create a dummy video file
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video data")

        # Mock transcript segmenter
        segmenter = MagicMock()
        segmenter.full_transcribe.return_value = [
            WordTimestamp(word="Hello", start=0.0, end=0.5),
            WordTimestamp(word="world", start=0.5, end=1.0),
        ]
        segmenter.segment_by_content.return_value = [
            SegmentBoundary(start_time=0.0, end_time=30.0,
                          hook_summary="Test", segment_type="hook"),
        ]
        segmenter.refine_boundaries.return_value = [
            SceneSegment(start_time=0.0, end_time=30.0, scene_index=0),
        ]
        mock_segmenter.return_value = segmenter

        # Mock audio analyzer
        audio_analyzer = MagicMock()
        audio_analyzer.analyze_segment.return_value = AudioFeatures(
            audio_peak_score=0.5, high_freq_score=0.3,
            dynamic_range=0.2, zcr_score=0.1,
        )
        mock_audio.return_value = audio_analyzer

        # Mock visual analyzer
        visual_analyzer = MagicMock()
        visual_analyzer.analyze_segment.return_value = VisualFeatures(
            motion_score=0.4, face_presence=0.6,
            visual_interest=0.3, composition_score=0.5,
        )
        mock_visual.return_value = visual_analyzer

        # Mock semantic analyzer
        semantic_analyzer = MagicMock()
        semantic_analyzer.analyze_segment.return_value = SemanticFeatures(
            emotional_intensity=7.0, narrative_interest=6.0,
            hook_potential=8.0, asmr_quality=5.0,
            visual_appeal=7.0, uniqueness=6.0,
        )
        mock_semantic.return_value = semantic_analyzer

        # Mock virality scorer
        scorer = MagicMock()
        scorer.calculate_score.return_value = ViralityScore(
            total_score=75.0, component_scores={}, confidence=0.5,
        )
        mock_scorer.return_value = scorer

        # Mock clip extractor
        extractor = MagicMock()
        extractor.extract_clip.return_value = True
        mock_extractor.return_value = extractor

        # Mock subtitle burner
        burner = MagicMock()
        burner.get_video_dimensions.return_value = (1080, 1920)
        mock_burner.return_value = burner

        # Mock caption analyzer
        caption = MagicMock()
        caption.analyze_video.return_value = None  # No caption data
        mock_caption.return_value = caption

        config = PipelineConfig(
            output_dir=str(tmp_path / "output"),
        )
        pipeline = ViralClipPipeline(config=config)

        result = pipeline.process_video(
            video_path=str(video_file),
            title="Test Video",
            top_n=10,
            min_score=50.0,
        )

        # Verify transcript-first flow
        segmenter.full_transcribe.assert_called_once_with(str(video_file))
        segmenter.segment_by_content.assert_called_once()
        segmenter.refine_boundaries.assert_called_once()

        # Verify all analyzers called (semantic is mandatory)
        assert audio_analyzer.analyze_segment.call_count == 1
        assert visual_analyzer.analyze_segment.call_count == 1
        assert semantic_analyzer.analyze_segment.call_count == 1

        # Verify scorer called
        assert scorer.calculate_score.call_count == 1

        # Verify extractor called
        assert extractor.extract_clip.call_count == 1

        # Verify result
        assert isinstance(result, ProcessingResult)
        assert result.total_scenes == 1
        assert len(result.clips) == 1


# ------------------------------------------------------------------
# Test Pipeline with ContentProfile
# ------------------------------------------------------------------


class TestPipelineContentProfile:
    """Test pipeline construction with ContentProfile."""

    def test_pipeline_with_gaming_profile(self):
        """Pipeline accepts a PipelineConfig with a gaming ContentProfile."""
        from yacg.pipeline import ViralClipPipeline

        profile = ContentProfile(
            content_type="gaming",
            target_audience="gamers",
            tone="energetic",
        )
        config = PipelineConfig(content_profile=profile)
        pipeline = ViralClipPipeline(config=config)
        assert pipeline.config.content_profile.content_type == "gaming"
        assert pipeline.config.content_profile.tone == "energetic"

    def test_pipeline_default_profile_is_general(self):
        """Default pipeline has general content profile."""
        from yacg.pipeline import ViralClipPipeline

        pipeline = ViralClipPipeline()
        assert pipeline.config.content_profile.content_type == "general"

    def test_semantic_analyzer_receives_profile_fields(self):
        """SemanticAnalyzer gets content_type, channel_description, etc. from pipeline."""
        from yacg.pipeline import ViralClipPipeline

        profile = ContentProfile(
            content_type="cooking",
            channel_description="Chef's Kitchen",
            target_audience="home cooks",
            tone="casual",
            custom_instructions="Focus on technique",
        )
        config = PipelineConfig(content_profile=profile)
        pipeline = ViralClipPipeline(config=config)
        sa = pipeline._get_semantic_analyzer()
        assert sa.content_type == "cooking"
        assert sa.channel_description == "Chef's Kitchen"
        assert sa.target_audience == "home cooks"
        assert sa.tone == "casual"
        assert sa.custom_instructions == "Focus on technique"

    def test_caption_analyzer_receives_profile_fields(self):
        """OllamaVideoAnalyzer gets all ContentProfile fields from pipeline."""
        from yacg.pipeline import ViralClipPipeline

        profile = ContentProfile(
            content_type="educational",
            channel_description="Science channel",
            target_audience="students",
            tone="professional",
            platform="shorts",
            caption_length="long",
            hashtag_count=4,
            custom_instructions="Include citations",
        )
        config = PipelineConfig(content_profile=profile)
        pipeline = ViralClipPipeline(config=config)
        ca = pipeline._get_caption_analyzer()
        assert ca.content_type == "educational"
        assert ca.channel_description == "Science channel"
        assert ca.target_audience == "students"
        assert ca.tone == "professional"
        assert ca.platform == "shorts"
        assert ca.caption_length == "long"
        assert ca.hashtag_count == 4
        assert ca.custom_instructions == "Include citations"

    def test_transcript_segmenter_receives_profile_fields(self):
        """TranscriptSegmenter gets content_type, channel_description, etc. from pipeline."""
        from yacg.pipeline import ViralClipPipeline

        profile = ContentProfile(
            content_type="gaming",
            channel_description="Pro Gamer Channel",
            target_audience="gamers",
            custom_instructions="Only clutch moments",
        )
        config = PipelineConfig(content_profile=profile)
        pipeline = ViralClipPipeline(config=config)
        ts = pipeline._get_transcript_segmenter()
        assert ts.content_type == "gaming"
        assert ts.channel_description == "Pro Gamer Channel"
        assert ts.target_audience == "gamers"
        assert ts.custom_instructions == "Only clutch moments"

    def test_audio_analyzer_asmr_uses_default_keywords(self):
        """ASMR content type creates AudioAnalyzer with default ASMR keywords."""
        from yacg.pipeline import ViralClipPipeline

        profile = ContentProfile(content_type="asmr")
        config = PipelineConfig(content_profile=profile)
        pipeline = ViralClipPipeline(config=config)
        aa = pipeline._get_audio_analyzer()
        # Default ASMR keywords (from AudioAnalyzer.__init__) include ASMR terms
        assert "tingles" in aa.asmr_keywords or "whisper" in aa.asmr_keywords

    def test_audio_analyzer_general_uses_engagement_keywords(self):
        """Non-ASMR content type creates AudioAnalyzer with general engagement keywords."""
        from yacg.pipeline import ViralClipPipeline

        profile = ContentProfile(content_type="gaming")
        config = PipelineConfig(content_profile=profile)
        pipeline = ViralClipPipeline(config=config)
        aa = pipeline._get_audio_analyzer()
        assert "amazing" in aa.asmr_keywords or "incredible" in aa.asmr_keywords
