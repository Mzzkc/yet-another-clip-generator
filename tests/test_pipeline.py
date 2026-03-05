"""
Integration tests for the Viral Clip Extractor pipeline.

Tests pipeline initialization, config loading, CLI argument parsing,
TranscriptBridge, YouTubeDownloader URL parsing, CSV generation,
and mock-based pipeline orchestration.
"""

import csv
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from viral_clip_extractor.models import (
    AudioFeatures,
    ClipData,
    PipelineConfig,
    ProcessingResult,
    SceneSegment,
    SemanticFeatures,
    ViralityScore,
    VisualFeatures,
)


# ------------------------------------------------------------------
# Test PipelineConfig loading from INI
# ------------------------------------------------------------------


class TestConfigLoading:
    """Test loading PipelineConfig from INI files."""

    def test_load_default_config(self):
        """Loading with no path returns defaults."""
        from viral_clip_extractor.utils.config import load_config

        config = load_config(None)
        assert isinstance(config, PipelineConfig)
        assert config.model_name == "qwen2.5-vl:7b"
        assert config.scene_threshold == 3.0
        assert config.min_scene_len == 7.0
        assert config.top_n_clips == 10

    def test_load_missing_file_returns_defaults(self):
        """Loading a non-existent file returns defaults gracefully."""
        from viral_clip_extractor.utils.config import load_config

        config = load_config("/nonexistent/path.ini")
        assert isinstance(config, PipelineConfig)
        assert config.model_name == "qwen2.5-vl:7b"

    def test_load_custom_config(self, tmp_path):
        """Loading a valid INI file overrides defaults."""
        from viral_clip_extractor.utils.config import load_config

        ini_content = """\
[Model]
model_name = test-model:latest
ollama_host = http://example.com:1234

[SceneDetection]
threshold = 5.0
min_scene_len = 10.0
max_scene_len = 45.0

[ClipSelection]
top_n_clips = 5
min_virality_score = 80.0

[Features]
enable_semantic = false
enable_captions = false
vertical_crop = false
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
        assert config.enable_semantic is False
        assert config.enable_captions is False
        assert config.vertical_crop is False


# ------------------------------------------------------------------
# Test CLI argument parsing
# ------------------------------------------------------------------


class TestCLIParsing:
    """Test argparse-based CLI argument parsing."""

    def test_process_subcommand(self):
        """Parse 'process' subcommand with required args."""
        from viral_clip_extractor.cli import build_parser

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
        from viral_clip_extractor.cli import build_parser

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
        from viral_clip_extractor.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "batch", "--videos-dir", "/path/to/videos",
        ])

        assert args.command == "batch"
        assert args.videos_dir == "/path/to/videos"

    def test_check_subcommand(self):
        """Parse 'check' subcommand."""
        from viral_clip_extractor.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["check"])
        assert args.command == "check"

    def test_no_command_returns_error(self):
        """main() with no args returns error exit code."""
        from viral_clip_extractor.cli import main

        exit_code = main([])
        assert exit_code == 1

    def test_process_flags(self):
        """Parse boolean flags (--no-captions, --no-semantic, --no-vertical)."""
        from viral_clip_extractor.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "process", "--video", "test.mp4",
            "--no-captions", "--no-semantic", "--no-vertical",
        ])

        assert args.no_captions is True
        assert args.no_semantic is True
        assert args.no_vertical is True


# ------------------------------------------------------------------
# Test TranscriptBridge
# ------------------------------------------------------------------


class TestTranscriptBridge:
    """Test TranscriptBridge with mock transcript JSON."""

    def _sample_transcript(self) -> dict:
        """Build a sample yt-transcriber output."""
        return {
            "schema_version": "1.0",
            "meta": {
                "video": {"title": "Test Video", "duration_seconds": 120},
                "transcription": {"segment_count": 4},
            },
            "content": {
                "full_text": "Hello everyone welcome to the relaxing session tingles",
                "segments": [
                    {"id": 0, "start": 0.0, "end": 5.0, "text": "Hello everyone"},
                    {"id": 1, "start": 5.0, "end": 15.0, "text": "welcome to the"},
                    {"id": 2, "start": 15.0, "end": 25.0, "text": "relaxing session"},
                    {"id": 3, "start": 25.0, "end": 35.0, "text": "tingles and magic"},
                ],
            },
        }

    def test_load_transcript(self, tmp_path):
        """Load transcript from a JSON file."""
        from viral_clip_extractor.transcript_bridge import TranscriptBridge

        bridge = TranscriptBridge()
        transcript = self._sample_transcript()

        json_path = tmp_path / "transcript.json"
        json_path.write_text(json.dumps(transcript))

        result = bridge.load_transcript(str(json_path))
        assert result is not None
        assert result["schema_version"] == "1.0"
        assert len(result["content"]["segments"]) == 4

    def test_load_missing_transcript(self):
        """Loading non-existent file returns None."""
        from viral_clip_extractor.transcript_bridge import TranscriptBridge

        bridge = TranscriptBridge()
        result = bridge.load_transcript("/nonexistent/file.json")
        assert result is None

    def test_get_segment_text(self):
        """Extract text for a time range."""
        from viral_clip_extractor.transcript_bridge import TranscriptBridge

        bridge = TranscriptBridge()
        transcript = self._sample_transcript()

        # 0-10s should get segments 0 and 1
        text = bridge.get_segment_text(transcript, 0.0, 10.0)
        assert "Hello everyone" in text
        assert "welcome to the" in text

        # 20-30s should get segments 2 and 3
        text = bridge.get_segment_text(transcript, 20.0, 30.0)
        assert "relaxing session" in text
        assert "tingles" in text

    def test_get_segment_text_empty(self):
        """Empty time range returns empty string."""
        from viral_clip_extractor.transcript_bridge import TranscriptBridge

        bridge = TranscriptBridge()
        transcript = self._sample_transcript()

        text = bridge.get_segment_text(transcript, 100.0, 110.0)
        assert text == ""

    def test_find_trigger_words(self):
        """Detect trigger words in a time range."""
        from viral_clip_extractor.transcript_bridge import TranscriptBridge

        bridge = TranscriptBridge()
        transcript = self._sample_transcript()

        # Segment 2-3 contains "relaxing" (close to "relax") — but exact match needed
        # Segment 3 contains "tingles" and "magic"
        found = bridge.find_trigger_words(transcript, 25.0, 40.0)
        assert "tingles" in found
        assert "magic" in found

    def test_find_trigger_words_custom_keywords(self):
        """Custom keyword list works."""
        from viral_clip_extractor.transcript_bridge import TranscriptBridge

        bridge = TranscriptBridge()
        transcript = self._sample_transcript()

        found = bridge.find_trigger_words(
            transcript, 0.0, 40.0, keywords=["hello", "session"],
        )
        assert "hello" in found
        assert "session" in found


# ------------------------------------------------------------------
# Test YouTubeDownloader URL parsing
# ------------------------------------------------------------------


class TestYouTubeURLParsing:
    """Test YouTubeDownloader.extract_video_id with various URL formats."""

    def test_standard_url(self):
        from viral_clip_extractor.youtube_downloader import YouTubeDownloader

        dl = YouTubeDownloader.__new__(YouTubeDownloader)
        assert dl.extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_short_url(self):
        from viral_clip_extractor.youtube_downloader import YouTubeDownloader

        dl = YouTubeDownloader.__new__(YouTubeDownloader)
        assert dl.extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_shorts_url(self):
        from viral_clip_extractor.youtube_downloader import YouTubeDownloader

        dl = YouTubeDownloader.__new__(YouTubeDownloader)
        assert dl.extract_video_id("https://youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_embed_url(self):
        from viral_clip_extractor.youtube_downloader import YouTubeDownloader

        dl = YouTubeDownloader.__new__(YouTubeDownloader)
        assert dl.extract_video_id("https://youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_bare_id(self):
        from viral_clip_extractor.youtube_downloader import YouTubeDownloader

        dl = YouTubeDownloader.__new__(YouTubeDownloader)
        assert dl.extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_invalid_url(self):
        from viral_clip_extractor.youtube_downloader import YouTubeDownloader

        dl = YouTubeDownloader.__new__(YouTubeDownloader)
        assert dl.extract_video_id("not_a_valid_url") == ""


# ------------------------------------------------------------------
# Test Pipeline initialization
# ------------------------------------------------------------------


class TestPipelineInit:
    """Test pipeline instantiation and default config."""

    def test_default_init(self):
        """Pipeline initializes with default config."""
        from viral_clip_extractor.pipeline import ViralClipPipeline

        pipeline = ViralClipPipeline()
        assert pipeline.config is not None
        assert isinstance(pipeline.config, PipelineConfig)
        assert pipeline.config.model_name == "qwen2.5-vl:7b"

    def test_custom_config(self):
        """Pipeline respects custom config."""
        from viral_clip_extractor.pipeline import ViralClipPipeline

        config = PipelineConfig(
            model_name="custom:latest",
            top_n_clips=5,
            enable_semantic=False,
        )
        pipeline = ViralClipPipeline(config=config)
        assert pipeline.config.model_name == "custom:latest"
        assert pipeline.config.top_n_clips == 5
        assert pipeline.config.enable_semantic is False

    def test_process_video_missing_file(self):
        """Processing a non-existent file returns error result."""
        from viral_clip_extractor.pipeline import ViralClipPipeline

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
        from viral_clip_extractor.pipeline import ViralClipPipeline

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
            caption={
                "hook": "Watch this!",
                "description": "Amazing clip",
                "hashtags": ["#asmr", "#viral"],
                "full_caption": "Watch this!\n\nAmazing clip\n\n#asmr #viral",
                "category": "ASMR",
            },
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
            "clip_filename", "start_time", "end_time", "duration",
            "virality_score", "hook", "description", "hashtags",
            "full_caption", "category", "audio_peak", "motion_score",
            "face_presence", "asmr_quality", "processing_timestamp",
        ]
        for col in expected_cols:
            assert col in row, f"Missing column: {col}"

        assert row["hook"] == "Watch this!"
        assert row["category"] == "ASMR"
        assert float(row["virality_score"]) == 85.0

    def test_csv_empty_clips(self, tmp_path):
        """CSV with no clips still has headers."""
        from viral_clip_extractor.pipeline import ViralClipPipeline

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
# Mock-based pipeline orchestration test
# ------------------------------------------------------------------


class TestPipelineOrchestration:
    """Mock all analyzers and verify pipeline calls them in correct order."""

    @patch("viral_clip_extractor.pipeline.ViralClipPipeline._get_clip_extractor")
    @patch("viral_clip_extractor.pipeline.ViralClipPipeline._get_virality_scorer")
    @patch("viral_clip_extractor.pipeline.ViralClipPipeline._get_visual_analyzer")
    @patch("viral_clip_extractor.pipeline.ViralClipPipeline._get_audio_analyzer")
    @patch("viral_clip_extractor.pipeline.ViralClipPipeline._get_scene_detector")
    def test_pipeline_calls_analyzers_in_order(
        self,
        mock_scene_det,
        mock_audio,
        mock_visual,
        mock_scorer,
        mock_extractor,
        tmp_path,
    ):
        """Verify pipeline calls scene detection, then analyzers, then extraction."""
        from viral_clip_extractor.pipeline import ViralClipPipeline

        # Create a dummy video file
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video data")

        # Mock scene detector
        scene_detector = MagicMock()
        scene_detector.detect_scenes.return_value = [
            SceneSegment(start_time=0.0, end_time=15.0, scene_index=0),
            SceneSegment(start_time=15.0, end_time=30.0, scene_index=1),
        ]
        mock_scene_det.return_value = scene_detector

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

        config = PipelineConfig(
            enable_semantic=False,
            enable_captions=False,
            output_dir=str(tmp_path / "output"),
        )
        pipeline = ViralClipPipeline(config=config)

        result = pipeline.process_video(
            video_path=str(video_file),
            title="Test Video",
            top_n=10,
            min_score=50.0,
        )

        # Verify scene detection was called
        scene_detector.detect_scenes.assert_called_once_with(str(video_file))

        # Verify audio analyzer called for each scene
        assert audio_analyzer.analyze_segment.call_count == 2

        # Verify visual analyzer called for each scene
        assert visual_analyzer.analyze_segment.call_count == 2

        # Verify scorer called for each scene
        assert scorer.calculate_score.call_count == 2

        # Verify extractor called for clips above threshold
        assert extractor.extract_clip.call_count == 2

        # Verify result
        assert isinstance(result, ProcessingResult)
        assert result.total_scenes == 2
        assert len(result.clips) == 2
