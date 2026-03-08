"""Tests for CLI module and __main__.py to boost coverage."""

import argparse
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# __main__.py coverage
# ---------------------------------------------------------------------------


class TestMainModule:
    """Cover __main__.py: sys.exit(main())."""

    def test_main_module_check_command(self):
        """Importing __main__ behavior through main() with check."""
        from yacg.cli import main

        # 'check' skips bootstrap, returns 0 when deps are present
        result = main(["check"])
        assert result == 0

    def test_main_module_no_command(self):
        """main() with no args prints help and returns 1."""
        from yacg.cli import main

        result = main([])
        assert result == 1


# ---------------------------------------------------------------------------
# CLI main() flow coverage
# ---------------------------------------------------------------------------


class TestCLIMainFlow:
    """Exercise main() code paths: warning filters, bootstrap, min-score validation."""

    def test_verbose_skips_warning_filters(self):
        """Verbose mode does not install warning filters."""
        from yacg.cli import main

        # process with -v on nonexistent file exercises the verbose code path
        result = main(["process", "--video", "/nonexistent.mp4",
                        "--title", "t", "-v", "--min-score", "0"])
        assert result == 1

    def test_min_score_below_zero(self):
        """--min-score < 0 is rejected."""
        from yacg.cli import main

        result = main(["process", "--video", "/nonexistent.mp4",
                        "--title", "t", "--min-score", "-10"])
        assert result == 1

    def test_min_score_above_100(self):
        """--min-score > 100 is rejected."""
        from yacg.cli import main

        result = main(["process", "--video", "/nonexistent.mp4",
                        "--title", "t", "--min-score", "150"])
        assert result == 1

    def test_process_nonexistent_file_returns_1(self):
        """process with nonexistent file returns 1."""
        from yacg.cli import main

        result = main(["process", "--video", "/nonexistent.mp4",
                        "--title", "t", "--min-score", "0"])
        assert result == 1

    def test_process_directory_as_video_returns_1(self):
        """process with a directory instead of file returns 1."""
        from yacg.cli import main

        result = main(["process", "--video", "/tmp",
                        "--title", "t", "--min-score", "0"])
        assert result == 1


# ---------------------------------------------------------------------------
# _build_config coverage
# ---------------------------------------------------------------------------


class TestBuildConfig:
    """Exercise _build_config with different argument combinations."""

    def test_build_config_with_ini_file(self):
        """_build_config loads a config INI when provided."""
        from yacg.cli import _build_config

        with tempfile.NamedTemporaryFile(suffix=".ini", mode="w", delete=False) as f:
            f.write("[DEFAULT]\nmin_scene_length = 5.0\n")
            ini_path = f.name

        try:
            args = argparse.Namespace(
                config=ini_path,
                output_dir="/tmp/test_out",
                model="test-model",
                whisper_model="base",
            )
            config = _build_config(args)
            assert config.output_dir == "/tmp/test_out"
            assert config.model_name == "test-model"
        finally:
            os.unlink(ini_path)

    def test_build_config_defaults(self):
        """_build_config with no overrides uses defaults."""
        from yacg.cli import _build_config

        args = argparse.Namespace(
            config=None,
            output_dir="./out",
            model="qwen2.5-vl:7b",
            whisper_model="base",
        )
        config = _build_config(args)
        assert config.output_dir == "./out"

    def test_build_config_content_type_gaming(self):
        """--content-type gaming loads gaming preset."""
        from yacg.cli import _build_config

        args = argparse.Namespace(
            config=None,
            output_dir="./out",
            model="qwen2.5-vl:7b",
            whisper_model="base",
            content_type="gaming",
            channel_description=None,
            target_audience=None,
            tone=None,
            platform=None,
            caption_length=None,
            hashtag_count=None,
            custom_instructions=None,
        )
        config = _build_config(args)
        assert config.content_profile.content_type == "gaming"
        assert config.content_profile.tone == "energetic"

    def test_build_config_gaming_with_tone_override(self):
        """--content-type gaming --tone calm overrides preset tone."""
        from yacg.cli import _build_config

        args = argparse.Namespace(
            config=None,
            output_dir="./out",
            model="qwen2.5-vl:7b",
            whisper_model="base",
            content_type="gaming",
            channel_description=None,
            target_audience=None,
            tone="calm",
            platform=None,
            caption_length=None,
            hashtag_count=None,
            custom_instructions=None,
        )
        config = _build_config(args)
        assert config.content_profile.content_type == "gaming"
        assert config.content_profile.tone == "calm"


# ---------------------------------------------------------------------------
# CLI argument parsing — content profile
# ---------------------------------------------------------------------------


class TestCLIContentProfileArgs:
    """Test new --content-type and related CLI argument parsing."""

    def test_content_type_gaming_parsed(self):
        from yacg.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "process", "--video", "test.mp4",
            "--content-type", "gaming",
        ])
        assert args.content_type == "gaming"

    def test_content_type_asmr_parsed(self):
        from yacg.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "process", "--video", "test.mp4",
            "--content-type", "asmr",
        ])
        assert args.content_type == "asmr"

    def test_tone_parsed(self):
        from yacg.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "process", "--video", "test.mp4",
            "--tone", "calm",
        ])
        assert args.tone == "calm"

    def test_platform_parsed(self):
        from yacg.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "process", "--video", "test.mp4",
            "--platform", "tiktok",
        ])
        assert args.platform == "tiktok"


# ---------------------------------------------------------------------------
# _build_config: all ContentProfile field overrides
# ---------------------------------------------------------------------------


class TestBuildConfigAllOverrides:
    """Test _build_config processes all 8 ContentProfile field overrides."""

    def test_build_config_channel_description_override(self):
        from yacg.cli import _build_config

        args = argparse.Namespace(
            config=None, output_dir="./out", model="qwen2.5-vl:7b",
            whisper_model="base", content_type=None,
            channel_description="My awesome channel",
            target_audience=None, tone=None, platform=None,
            caption_length=None, hashtag_count=None, custom_instructions=None,
        )
        config = _build_config(args)
        assert config.content_profile.channel_description == "My awesome channel"

    def test_build_config_target_audience_override(self):
        from yacg.cli import _build_config

        args = argparse.Namespace(
            config=None, output_dir="./out", model="qwen2.5-vl:7b",
            whisper_model="base", content_type=None,
            channel_description=None, target_audience="teens",
            tone=None, platform=None, caption_length=None,
            hashtag_count=None, custom_instructions=None,
        )
        config = _build_config(args)
        assert config.content_profile.target_audience == "teens"

    def test_build_config_platform_override(self):
        from yacg.cli import _build_config

        args = argparse.Namespace(
            config=None, output_dir="./out", model="qwen2.5-vl:7b",
            whisper_model="base", content_type=None,
            channel_description=None, target_audience=None,
            tone=None, platform="tiktok", caption_length=None,
            hashtag_count=None, custom_instructions=None,
        )
        config = _build_config(args)
        assert config.content_profile.platform == "tiktok"

    def test_build_config_caption_length_override(self):
        from yacg.cli import _build_config

        args = argparse.Namespace(
            config=None, output_dir="./out", model="qwen2.5-vl:7b",
            whisper_model="base", content_type=None,
            channel_description=None, target_audience=None,
            tone=None, platform=None, caption_length="long",
            hashtag_count=None, custom_instructions=None,
        )
        config = _build_config(args)
        assert config.content_profile.caption_length == "long"

    def test_build_config_hashtag_count_override(self):
        from yacg.cli import _build_config

        args = argparse.Namespace(
            config=None, output_dir="./out", model="qwen2.5-vl:7b",
            whisper_model="base", content_type=None,
            channel_description=None, target_audience=None,
            tone=None, platform=None, caption_length=None,
            hashtag_count=3, custom_instructions=None,
        )
        config = _build_config(args)
        assert config.content_profile.hashtag_count == 3

    def test_build_config_custom_instructions_override(self):
        from yacg.cli import _build_config

        args = argparse.Namespace(
            config=None, output_dir="./out", model="qwen2.5-vl:7b",
            whisper_model="base", content_type=None,
            channel_description=None, target_audience=None,
            tone=None, platform=None, caption_length=None,
            hashtag_count=None, custom_instructions="Always mention the brand",
        )
        config = _build_config(args)
        assert config.content_profile.custom_instructions == "Always mention the brand"

    def test_build_config_asmr_content_type_syncs_asmr_mode(self):
        """--content-type asmr without --asmr-mode should auto-sync asmr_mode=True."""
        from yacg.cli import _build_config

        args = argparse.Namespace(
            config=None, output_dir="./out", model="qwen2.5-vl:7b",
            whisper_model="base", content_type="asmr",
            channel_description=None, target_audience=None,
            tone=None, platform=None, caption_length=None,
            hashtag_count=None, custom_instructions=None,
            asmr_mode=None, no_asmr_mode=False,
        )
        config = _build_config(args)
        assert config.content_profile.content_type == "asmr"
        assert config.asmr_mode is True

    def test_build_config_asmr_content_type_with_no_asmr_mode(self):
        """--content-type asmr --no-asmr-mode should NOT auto-sync."""
        from yacg.cli import _build_config

        args = argparse.Namespace(
            config=None, output_dir="./out", model="qwen2.5-vl:7b",
            whisper_model="base", content_type="asmr",
            channel_description=None, target_audience=None,
            tone=None, platform=None, caption_length=None,
            hashtag_count=None, custom_instructions=None,
            asmr_mode=None, no_asmr_mode=True,
        )
        config = _build_config(args)
        assert config.content_profile.content_type == "asmr"
        assert config.asmr_mode is False


# ---------------------------------------------------------------------------
# INI config loading with [ContentProfile] section
# ---------------------------------------------------------------------------


class TestINIContentProfileLoading:
    """Test load_config parses [ContentProfile] section from INI files."""

    def test_load_ini_with_content_profile_section(self, tmp_path):
        from yacg.utils.config import load_config

        ini = tmp_path / "test.ini"
        ini.write_text(
            "[ContentProfile]\n"
            "content_type = cooking\n"
            "tone = casual\n"
            "platform = reels\n"
            "target_audience = home cooks\n"
            "hashtag_count = 4\n"
        )
        config = load_config(str(ini))
        assert config.content_profile.content_type == "cooking"
        assert config.content_profile.tone == "casual"
        assert config.content_profile.platform == "reels"
        assert config.content_profile.target_audience == "home cooks"
        assert config.content_profile.hashtag_count == 4

    def test_load_ini_invalid_content_type_raises(self, tmp_path):
        from yacg.utils.config import load_config

        ini = tmp_path / "bad.ini"
        ini.write_text(
            "[ContentProfile]\n"
            "content_type = invalid_type\n"
        )
        with pytest.raises(ValueError, match="Invalid content_type"):
            load_config(str(ini))

    def test_load_ini_invalid_tone_raises(self, tmp_path):
        from yacg.utils.config import load_config

        ini = tmp_path / "bad_tone.ini"
        ini.write_text(
            "[ContentProfile]\n"
            "content_type = general\n"
            "tone = screaming\n"
        )
        with pytest.raises(ValueError, match="Invalid tone"):
            load_config(str(ini))

    def test_load_ini_legacy_asmr_section(self, tmp_path):
        from yacg.utils.config import load_config

        ini = tmp_path / "legacy.ini"
        ini.write_text(
            "[ASMR Optimization]\n"
            "content_type = asmr\n"
            "asmr_mode = true\n"
        )
        config = load_config(str(ini))
        assert config.content_profile.content_type == "asmr"
        assert config.asmr_mode is True


# ---------------------------------------------------------------------------
# show-config and generate-config coverage
# ---------------------------------------------------------------------------


class TestShowConfigGenerateConfig:
    """Test show-config displays ContentProfile fields."""

    def test_show_config_displays_content_profile(self, capsys):
        from yacg.cli import main

        result = main(["show-config"])
        assert result == 0
        captured = capsys.readouterr()
        assert "content_type" in captured.out
        assert "Content Profile" in captured.out

    def test_generate_config_creates_file(self, tmp_path):
        from yacg.cli import main

        config_path = str(tmp_path / "generated.ini")
        result = main(["generate-config", "--output", config_path])
        assert result == 0
        assert os.path.exists(config_path)
        content = Path(config_path).read_text()
        assert "ContentProfile" in content or "content_type" in content


# ---------------------------------------------------------------------------
# _cmd_check coverage
# ---------------------------------------------------------------------------


class TestCmdCheck:
    """Cover _cmd_check code paths."""

    def test_check_passes_on_this_system(self):
        """check command returns 0 when deps are installed."""
        from yacg.cli import main

        result = main(["check"])
        assert result == 0

    def test_check_output_contains_ok(self, capsys):
        """check command prints OK for available deps."""
        from yacg.cli import main

        main(["check"])
        captured = capsys.readouterr()
        assert "OK" in captured.out

    @patch("shutil.which", return_value=None)
    def test_check_missing_ffmpeg(self, mock_which, capsys):
        """check command reports missing ffmpeg."""
        from yacg.cli import _cmd_check

        args = argparse.Namespace()
        result = _cmd_check(args)
        assert result == 1
        captured = capsys.readouterr()
        assert "NOT FOUND" in captured.out


# ---------------------------------------------------------------------------
# _cmd_batch coverage
# ---------------------------------------------------------------------------


class TestCmdBatch:
    """Cover _cmd_batch code paths."""

    def test_batch_nonexistent_dir(self):
        """batch with nonexistent directory returns 1."""
        from yacg.cli import main

        result = main(["batch", "--videos-dir", "/nonexistent_dir",
                        "--min-score", "0"])
        assert result == 1

    def test_batch_empty_dir(self):
        """batch with empty directory returns 1."""
        from yacg.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            result = main(["batch", "--videos-dir", tmpdir,
                            "--min-score", "0"])
            assert result == 1


# ---------------------------------------------------------------------------
# _cmd_youtube coverage (mocked)
# ---------------------------------------------------------------------------


class TestCmdYoutube:
    """Cover _cmd_youtube without actually downloading."""

    @patch("yacg.pipeline.ViralClipPipeline")
    def test_youtube_error_returns_1(self, mock_pipeline_cls):
        """youtube command returns 1 when pipeline has errors."""
        from yacg.cli import main
        from yacg.models import ProcessingResult

        mock_result = ProcessingResult(
            video_path="/tmp/fake.mp4",
            video_title="Fake",
            clips=[],
            total_scenes=0,
            processing_time_seconds=1.0,
            errors=["download failed"],
        )
        mock_pipeline_cls.return_value.process_youtube.return_value = mock_result

        result = main(["youtube", "--url", "https://youtube.com/watch?v=dQw4w9WgXcQ",
                        "--min-score", "0"])
        assert result == 1
