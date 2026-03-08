"""
Command-line interface for the Viral Clip Extractor.

Provides ``process``, ``youtube``, ``batch``, and ``check`` subcommands
for running the pipeline on local files, YouTube URLs, or entire directories.
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by process/youtube/batch subcommands."""
    parser.add_argument(
        "--output-dir", default="./clip_output",
        help="Output directory for clips and reports (default: ./clip_output)",
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="Number of top clips to extract (default: 20)",
    )
    parser.add_argument(
        "--min-score", type=float, default=70.0,
        help="Virality score threshold (0-100). Typical range: 50-80. "
             "Scores below 50 rarely indicate viral potential; 60-70 is "
             "moderate, 70-85 is strong, 85+ is exceptional. Lower values "
             "produce more clips, higher values are more selective "
             "(default: 70)",
    )
    parser.add_argument(
        "--model", default="qwen2.5-vl:7b",
        help="Ollama VLM model name for semantic analysis and captions "
             "(default: qwen2.5-vl:7b)",
    )
    parser.add_argument(
        "--whisper-model", default="small",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size: tiny=fastest/lowest quality, "
             "base=fast/good, small=balanced (recommended), "
             "medium=slow/high quality, large-v3=slowest/best "
             "(default: small)",
    )
    parser.add_argument(
        "--segmentation-model", default=None,
        help="Ollama model for text-only transcript segmentation. A text "
             "model (e.g. llama3:8b) avoids loading the vision encoder "
             "into VRAM. Falls back to --model if omitted",
    )
    parser.add_argument(
        "--scoring-weights", default=None,
        help="JSON string of scoring weight overrides, e.g. "
             "'{\"hook\": 0.25, \"asmr\": 0.15}'. "
             "Unspecified keys keep defaults",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to config INI file (uses defaults if omitted)",
    )
    parser.add_argument(
        "--ollama-host", default=None,
        help="Ollama API base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--whisper-device", default=None,
        choices=["auto", "cpu", "cuda"],
        help="Whisper inference device: auto=CUDA if available else CPU "
             "(default: auto)",
    )
    parser.add_argument(
        "--whisper-compute-type", default=None,
        choices=["auto", "int8", "float16", "float32"],
        help="Whisper compute precision: auto selects best for device "
             "(default: auto)",
    )
    parser.add_argument(
        "--num-frames", type=int, default=None,
        choices=[1, 2, 3, 4, 5],
        help="Number of frames per segment for VLM analysis. Set to 1 "
             "if your VLM produces garbled output with multiple images "
             "(default: 3)",
    )
    parser.add_argument(
        "--scene-threshold", type=float, default=None,
        help="Scene detection threshold — lower is more sensitive "
             "(default: 3.0)",
    )
    parser.add_argument(
        "--min-scene-len", type=float, default=None,
        help="Minimum scene length in seconds (default: 7.0)",
    )
    parser.add_argument(
        "--max-scene-len", type=float, default=None,
        help="Maximum scene length in seconds (default: 60.0)",
    )
    parser.add_argument(
        "--asmr-mode", action="store_true", default=None,
        help="Enable ASMR-optimized scoring (default: off)",
    )
    parser.add_argument(
        "--no-asmr-mode", action="store_true", default=None,
        help="Disable ASMR-optimized scoring",
    )
    parser.add_argument(
        "--content-type", default=None,
        choices=["general", "gaming", "cooking", "asmr", "educational",
                 "fitness", "comedy", "music", "beauty", "tech", "vlog"],
        help="Content type preset (default: general). Sets tone, audience, "
             "and platform defaults which can be overridden individually",
    )
    parser.add_argument(
        "--channel-description", default=None,
        help="Description of the channel/creator (injected into LLM prompts)",
    )
    parser.add_argument(
        "--target-audience", default=None,
        help="Target audience description (injected into LLM prompts)",
    )
    parser.add_argument(
        "--tone", default=None,
        choices=["energetic", "calm", "professional", "casual",
                 "humorous", "inspirational", "dramatic", "engaging"],
        help="Tone of generated captions (default: engaging)",
    )
    parser.add_argument(
        "--platform", default=None,
        choices=["tiktok", "reels", "shorts", "all"],
        help="Target platform (default: all)",
    )
    parser.add_argument(
        "--caption-length", default=None,
        choices=["short", "medium", "long"],
        help="Caption description length (default: medium)",
    )
    parser.add_argument(
        "--hashtag-count", type=int, default=None,
        choices=range(3, 8),
        help="Number of hashtags to generate (3-7, default: 5)",
    )
    parser.add_argument(
        "--custom-instructions", default=None,
        help="Custom instructions injected into all LLM prompts",
    )
    parser.add_argument(
        "--context-padding", type=float, default=None,
        help="Seconds of context padding around clip boundaries (default: 2.0)",
    )
    parser.add_argument(
        "--pause-threshold", type=float, default=None,
        help="Pause duration (seconds) that triggers a segment boundary "
             "(default: 0.3)",
    )
    parser.add_argument(
        "--min-segment-duration", type=float, default=None,
        help="Minimum segment duration in seconds (default: 15.0)",
    )
    parser.add_argument(
        "--max-segment-duration", type=float, default=None,
        help="Maximum segment duration in seconds (default: 45.0)",
    )
    parser.add_argument(
        "--vad-filter", action="store_true", default=None,
        help="Enable Voice Activity Detection filter for Whisper "
             "(default: on; disable for ASMR/ambient content)",
    )
    parser.add_argument(
        "--no-vad-filter", action="store_true", default=None,
        help="Disable Voice Activity Detection filter (for ASMR/ambient content "
             "where non-speech audio is the primary content)",
    )
    # Subtitle styling
    parser.add_argument(
        "--subtitle-font", default=None,
        help="Font name for subtitles (default: auto-detect system font). "
             "Examples: 'Arial', 'Liberation Sans', 'DejaVu Sans'",
    )
    parser.add_argument(
        "--subtitle-font-size", type=float, default=None,
        help="Subtitle font size as fraction of frame height "
             "(default: 0.055). Larger = bigger text",
    )
    parser.add_argument(
        "--subtitle-color", default=None,
        help="Subtitle text color in ASS &HAABBGGRR hex format "
             "(default: &H00FFFFFF = white). Common: "
             "&H0000FFFF=yellow, &H000000FF=red",
    )
    parser.add_argument(
        "--subtitle-outline-color", default=None,
        help="Subtitle outline color in ASS &HAABBGGRR hex format "
             "(default: &H00000000 = black)",
    )
    parser.add_argument(
        "--subtitle-outline-width", type=float, default=None,
        help="Subtitle outline thickness in pixels (default: 3.0)",
    )
    parser.add_argument(
        "--subtitle-shadow", type=float, default=None,
        help="Subtitle shadow depth in pixels (default: 1.5)",
    )
    parser.add_argument(
        "--subtitle-margin-v", type=float, default=None,
        help="Subtitle vertical margin as fraction of frame height "
             "(default: 0.38 = ~62%% from top, clears TikTok/Reels UI)",
    )
    parser.add_argument(
        "--subtitle-margin-h", type=float, default=None,
        help="Subtitle horizontal margin as fraction of frame width "
             "(default: 0.15)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run analysis only — show ranked clips then stop (no extraction, "
             "subtitles, or captions)",
    )


def _build_config(args: argparse.Namespace):
    """Build PipelineConfig from parsed CLI arguments."""
    import json as _json

    from viral_clip_extractor.utils.config import load_config

    config = load_config(getattr(args, "config", None))

    # CLI overrides
    if hasattr(args, "output_dir") and args.output_dir:
        config.output_dir = args.output_dir
    if hasattr(args, "model") and args.model:
        config.model_name = args.model
    if hasattr(args, "whisper_model") and args.whisper_model:
        config.whisper_model = args.whisper_model
    if getattr(args, "dry_run", False):
        config.dry_run = True
    seg_model = getattr(args, "segmentation_model", None)
    if seg_model:
        config.segmentation_model = seg_model
    raw_weights = getattr(args, "scoring_weights", None)
    if raw_weights:
        try:
            overrides = _json.loads(raw_weights)
            config.scoring_weights.update(overrides)
            # Re-validate after mutation — catches typo'd keys like "hooks"
            from viral_clip_extractor.models import VALID_SCORING_KEYS
            invalid_keys = set(config.scoring_weights.keys()) - VALID_SCORING_KEYS
            if invalid_keys:
                raise ValueError(
                    f"Invalid scoring weight key(s): {sorted(invalid_keys)}. "
                    f"Valid keys: {sorted(VALID_SCORING_KEYS)}"
                )
        except _json.JSONDecodeError:
            logging.getLogger(__name__).warning(
                "Invalid JSON in --scoring-weights — ignoring"
            )

    # Additional CLI overrides for all PipelineConfig fields
    if getattr(args, "ollama_host", None):
        config.ollama_host = args.ollama_host
    if getattr(args, "whisper_device", None):
        config.whisper_device = args.whisper_device
    if getattr(args, "whisper_compute_type", None):
        config.whisper_compute_type = args.whisper_compute_type
    if getattr(args, "scene_threshold", None) is not None:
        config.scene_threshold = args.scene_threshold
    if getattr(args, "min_scene_len", None) is not None:
        config.min_scene_len = args.min_scene_len
    if getattr(args, "max_scene_len", None) is not None:
        config.max_scene_len = args.max_scene_len
    if getattr(args, "no_asmr_mode", False):
        config.asmr_mode = False
    elif getattr(args, "asmr_mode", None):
        config.asmr_mode = True
    # ContentProfile: preset resolution + individual overrides
    content_type_arg = getattr(args, "content_type", None)
    if content_type_arg:
        from copy import deepcopy
        from viral_clip_extractor.models import CONTENT_PRESETS
        preset = CONTENT_PRESETS.get(content_type_arg)
        if preset:
            config.content_profile = deepcopy(preset)
        else:
            config.content_profile.content_type = content_type_arg
    if getattr(args, "channel_description", None) is not None:
        config.content_profile.channel_description = args.channel_description
    if getattr(args, "target_audience", None) is not None:
        config.content_profile.target_audience = args.target_audience
    if getattr(args, "tone", None) is not None:
        config.content_profile.tone = args.tone
    if getattr(args, "platform", None) is not None:
        config.content_profile.platform = args.platform
    if getattr(args, "caption_length", None) is not None:
        config.content_profile.caption_length = args.caption_length
    if getattr(args, "hashtag_count", None) is not None:
        config.content_profile.hashtag_count = args.hashtag_count
    if getattr(args, "custom_instructions", None) is not None:
        config.content_profile.custom_instructions = args.custom_instructions
    # Auto-sync asmr_mode when content_type is "asmr" and user didn't
    # explicitly pass --no-asmr-mode.  Must run AFTER content_type
    # processing because PipelineConfig.__post_init__ already ran at
    # construction time and won't fire again on attribute reassignment.
    if config.content_profile.content_type == "asmr" and not getattr(args, "no_asmr_mode", False):
        config.asmr_mode = True
    if getattr(args, "num_frames", None) is not None:
        config.num_frames = args.num_frames
    if getattr(args, "context_padding", None) is not None:
        config.context_padding = args.context_padding
    if getattr(args, "pause_threshold", None) is not None:
        config.pause_threshold = args.pause_threshold
    if getattr(args, "min_segment_duration", None) is not None:
        config.min_segment_duration = args.min_segment_duration
    if getattr(args, "max_segment_duration", None) is not None:
        config.max_segment_duration = args.max_segment_duration
    if getattr(args, "no_vad_filter", False):
        config.vad_filter = False
    elif getattr(args, "vad_filter", None):
        config.vad_filter = True

    # Subtitle style overrides
    if getattr(args, "subtitle_font", None):
        config.subtitle_style.font_name = args.subtitle_font
    if getattr(args, "subtitle_font_size", None) is not None:
        config.subtitle_style.font_size_pct = args.subtitle_font_size
    if getattr(args, "subtitle_color", None):
        config.subtitle_style.primary_color = args.subtitle_color
    if getattr(args, "subtitle_outline_color", None):
        config.subtitle_style.outline_color = args.subtitle_outline_color
    if getattr(args, "subtitle_outline_width", None) is not None:
        config.subtitle_style.outline_width = args.subtitle_outline_width
    if getattr(args, "subtitle_shadow", None) is not None:
        config.subtitle_style.shadow = args.subtitle_shadow
    if getattr(args, "subtitle_margin_v", None) is not None:
        config.subtitle_style.margin_v_pct = args.subtitle_margin_v
    if getattr(args, "subtitle_margin_h", None) is not None:
        config.subtitle_style.margin_h_pct = args.subtitle_margin_h

    return config


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands.

    Returns:
        A configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="viral-clip-extractor",
        description="Extract viral-potential clips from long-form videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m viral_clip_extractor process --video video.mp4 --title "My Video"
  python -m viral_clip_extractor youtube --url https://youtube.com/watch?v=XXXXX
  python -m viral_clip_extractor batch --videos-dir /path/to/videos/
  python -m viral_clip_extractor check
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- process subcommand --
    process_parser = subparsers.add_parser(
        "process", help="Process a local video file",
    )
    process_parser.add_argument(
        "--video", required=True, help="Path to local video file",
    )
    process_parser.add_argument(
        "--title", default="", help="Video title (auto-detected if omitted)",
    )
    _add_common_args(process_parser)

    # -- youtube subcommand --
    youtube_parser = subparsers.add_parser(
        "youtube", help="Process a YouTube video",
    )
    youtube_parser.add_argument(
        "--url", required=True, help="YouTube video URL",
    )
    _add_common_args(youtube_parser)

    # -- batch subcommand --
    batch_parser = subparsers.add_parser(
        "batch", help="Process all videos in a directory",
    )
    batch_parser.add_argument(
        "--videos-dir", required=True,
        help="Directory containing video files",
    )
    _add_common_args(batch_parser)

    # -- check subcommand --
    subparsers.add_parser(
        "check", help="Verify system dependencies",
    )

    # -- show-config subcommand --
    show_config_parser = subparsers.add_parser(
        "show-config",
        help="Print current configuration with field descriptions",
    )
    show_config_parser.add_argument(
        "--config", default=None,
        help="Path to config INI file (shows defaults if omitted)",
    )

    # -- generate-config subcommand --
    gen_config_parser = subparsers.add_parser(
        "generate-config",
        help="Write a default INI config file with all fields documented",
    )
    gen_config_parser.add_argument(
        "--output", default="config.ini",
        help="Destination path for the generated config file (default: config.ini)",
    )

    return parser


def _result_exit_code(result) -> int:
    """Compute exit code from a ProcessingResult.

    Returns:
        0 — no errors.
        1 — total failure (errors and zero clips produced).
        2 — partial success (some clips produced but with errors).
    """
    if not result.errors:
        return 0
    has_clips = any(c.output_path for c in result.clips)
    return 2 if has_clips else 1


def _print_error_summary(errors: list[str]) -> None:
    """Print a highlighted error summary block."""
    if not errors:
        return
    print(f"\n{'=' * 60}")
    print(f"=== ERRORS ({len(errors)}) ===")
    print(f"{'=' * 60}")
    for err in errors:
        print(f"  - {err}")
    print(f"{'=' * 60}")


def _cmd_process(args: argparse.Namespace) -> int:
    """Handle the 'process' subcommand."""
    from viral_clip_extractor.pipeline import ViralClipPipeline

    config = _build_config(args)
    logger = logging.getLogger(__name__)
    logger.info("Output will be saved to: %s", os.path.abspath(config.output_dir))
    pipeline = ViralClipPipeline(config=config)

    result = pipeline.process_video(
        video_path=args.video,
        title=args.title,
        top_n=args.top_n,
        min_score=args.min_score,
    )

    _print_error_summary(result.errors)
    return _result_exit_code(result)


def _cmd_youtube(args: argparse.Namespace) -> int:
    """Handle the 'youtube' subcommand."""
    from viral_clip_extractor.pipeline import ViralClipPipeline

    config = _build_config(args)
    yt_logger = logging.getLogger(__name__)
    yt_logger.info("Output will be saved to: %s", os.path.abspath(config.output_dir))

    # Basic URL validation before attempting download
    url = args.url
    if not (url.startswith("http://") or url.startswith("https://")):
        yt_logger.error("Invalid URL: %s — must start with http:// or https://", url)
        return 1

    pipeline = ViralClipPipeline(config=config)

    result = pipeline.process_youtube(
        url=args.url,
        top_n=args.top_n,
        min_score=args.min_score,
    )

    _print_error_summary(result.errors)
    return _result_exit_code(result)


def _cmd_batch(args: argparse.Namespace) -> int:
    """Handle the 'batch' subcommand."""
    from viral_clip_extractor.pipeline import ViralClipPipeline

    batch_logger = logging.getLogger(__name__)
    videos_dir = Path(args.videos_dir)
    if not videos_dir.is_dir():
        batch_logger.error("Not a directory: %s", args.videos_dir)
        return 1

    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    video_files = sorted(
        f for f in videos_dir.iterdir()
        if f.suffix.lower() in video_extensions
    )

    if not video_files:
        batch_logger.error("No video files found in %s", args.videos_dir)
        return 1

    batch_logger.info("Found %d video files", len(video_files))
    config = _build_config(args)
    batch_logger.info("Output will be saved to: %s", os.path.abspath(config.output_dir))

    # Create a shared transcript segmenter so the Whisper model is loaded
    # once and reused across all videos (5-15s saved per video).
    shared_pipeline = ViralClipPipeline(config=config)
    shared_segmenter = shared_pipeline._get_transcript_segmenter()

    success = 0
    failed = 0

    for i, video_path in enumerate(video_files):
        batch_logger.info(
            "\n%s\nProcessing %d/%d: %s\n%s",
            "=" * 60, i + 1, len(video_files), video_path.name, "=" * 60,
        )

        # Per-video output directory — reuse base config, only change output_dir
        video_output = os.path.join(config.output_dir, video_path.stem)
        from copy import deepcopy
        video_config = deepcopy(config)
        video_config.output_dir = video_output

        pipeline = ViralClipPipeline(
            config=video_config,
            transcript_segmenter=shared_segmenter,
        )
        title = video_path.stem.replace("_", " ").replace("-", " ").title()

        result = pipeline.process_video(
            video_path=str(video_path),
            title=title,
            top_n=args.top_n,
            min_score=args.min_score,
        )

        if not result.errors:
            success += 1
        else:
            failed += 1

    total = len(video_files)
    batch_logger.info(
        "\n%s\nProcessed %d/%d videos successfully.",
        "=" * 60, success, total,
    )
    if failed > 0:
        batch_logger.info("  %d video(s) had errors.", failed)
    batch_logger.info("=" * 60)

    # Exit codes: 0 = all success, 1 = all fail, 2 = partial
    if failed == 0:
        return 0
    elif success == 0:
        return 1
    else:
        return 2


def _cmd_check(_args: argparse.Namespace) -> int:
    """Handle the 'check' subcommand — verify system dependencies."""
    all_ok = True

    # FFmpeg
    if shutil.which("ffmpeg"):
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, text=True, timeout=5,
            )
            version_line = result.stdout.split("\n")[0] if result.stdout else "unknown"
            print(f"  ffmpeg: OK ({version_line})")
        except Exception as exc:
            print(f"  ffmpeg: ERROR (found but version check failed: {exc})")
            all_ok = False
    else:
        print("  ffmpeg: NOT FOUND")
        all_ok = False

    # FFprobe
    if shutil.which("ffprobe"):
        print("  ffprobe: OK")
    else:
        print("  ffprobe: NOT FOUND")
        all_ok = False

    # Python packages
    packages = [
        ("numpy", "numpy"),
        ("cv2", "opencv-python-headless"),
        ("librosa", "librosa"),
        ("requests", "requests"),
        ("yt_dlp", "yt-dlp"),
        ("scenedetect", "scenedetect[opencv]"),
        ("faster_whisper", "faster-whisper"),
    ]

    for import_name, pip_name in packages:
        try:
            __import__(import_name)
            print(f"  {pip_name}: OK")
        except ImportError:
            print(f"  {pip_name}: NOT FOUND (pip install {pip_name})")
            all_ok = False

    # Whisper model cache check
    whisper_model_size = "small"
    try:
        from faster_whisper.utils import download_model
        from huggingface_hub import try_to_load_from_cache
        # faster-whisper uses ct2 converted models from HuggingFace
        repo_id = f"guillaumekln/faster-whisper-{whisper_model_size}"
        cached = try_to_load_from_cache(repo_id, "model.bin")
        if cached is not None:
            print(f"  Whisper model '{whisper_model_size}': CACHED")
        else:
            print(
                f"  Whisper model '{whisper_model_size}': NOT CACHED "
                f"— first run will download it"
            )
    except ImportError:
        # faster_whisper not installed — already flagged above
        pass
    except Exception:
        print(
            f"  Whisper model '{whisper_model_size}': NOT CACHED "
            f"— first run will download it"
        )

    # Ollama
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            print(f"  Ollama: OK ({len(models)} models loaded)")
            if "qwen2.5-vl:7b" in models:
                print("  qwen2.5-vl:7b: OK")
            else:
                print("  qwen2.5-vl:7b: NOT LOADED (ollama pull qwen2.5-vl:7b)")
            # Check segmentation model (default: qwen2.5:7b, text-only variant)
            seg_model = "qwen2.5:7b"
            if seg_model in models:
                print(f"  {seg_model} (segmentation): OK")
            else:
                print(
                    f"  {seg_model} (segmentation): NOT LOADED "
                    f"(ollama pull {seg_model})"
                )
        else:
            print("  Ollama: ERROR (unexpected response)")
    except Exception:
        print("  Ollama: NOT RUNNING (start with: ollama serve)")

    # DNN face detection model check
    dnn_model_dirs = [
        str(Path(__file__).resolve().parent / "models"),
        str(Path.home() / ".vce" / "models"),
    ]
    dnn_found = False
    for d in dnn_model_dirs:
        proto = Path(d) / "deploy.prototxt"
        model_file = Path(d) / "res10_300x300_ssd_iter_140000.caffemodel"
        if proto.exists() and model_file.exists():
            print(f"  DNN face detection: OK (models in {d})")
            dnn_found = True
            break
    if not dnn_found:
        print(
            "  DNN face detection: NOT FOUND (using Haar cascade fallback). "
            "For better face detection, download deploy.prototxt and "
            "res10_300x300_ssd_iter_140000.caffemodel to "
            f"{dnn_model_dirs[1]}"
        )

    if all_ok:
        print("\nAll required dependencies satisfied.")
    else:
        print("\nSome required dependencies are missing. Install them and try again.")

    return 0 if all_ok else 1


def _cmd_show_config(args: argparse.Namespace) -> int:
    """Handle the 'show-config' subcommand — display current config."""
    from viral_clip_extractor.utils.config import load_config

    config = load_config(getattr(args, "config", None))
    descriptions = {
        "model_name": "Ollama VLM model for semantic analysis and captions",
        "ollama_host": "Ollama API base URL",
        "whisper_model": "Whisper model size (tiny/base/small/medium/large-v3)",
        "segmentation_model": "Ollama model for transcript segmentation (empty = use model_name)",
        "scene_threshold": "Scene detection threshold (lower = more sensitive)",
        "min_scene_len": "Minimum scene length in seconds",
        "max_scene_len": "Maximum scene length in seconds",
        "top_n_clips": "Maximum number of clips to extract",
        "min_virality_score": "Minimum virality score threshold (0-100)",
        "asmr_mode": "Enable ASMR-optimized scoring",
        "output_dir": "Output directory for clips and reports",
        "context_padding": "Seconds of context padding around clip boundaries",
        "scoring_weights": "Virality scoring weights",
        "whisper_device": "Whisper inference device (auto/cpu/cuda)",
        "whisper_compute_type": "Whisper compute precision (auto/int8/float16/float32)",
        "num_frames": "Number of frames per segment for VLM analysis",
        "dry_run": "Analysis only — no extraction, subtitles, or captions",
        "pause_threshold": "Pause duration (seconds) that triggers a segment boundary",
        "min_segment_duration": "Minimum segment duration in seconds",
        "max_segment_duration": "Maximum segment duration in seconds",
        "vad_filter": "Voice Activity Detection filter for Whisper",
    }
    print("Current configuration:")
    print(f"  (source: {getattr(args, 'config', None) or 'defaults'})")
    print()
    for field_name, desc in descriptions.items():
        value = getattr(config, field_name, "N/A")
        print(f"  {field_name}: {value}")
        print(f"    # {desc}")

    # ContentProfile section
    cp = config.content_profile
    cp_descriptions = {
        "content_type": "Content type preset (controls genre-specific prompt behavior)",
        "channel_description": "Channel/creator identity (injected into LLM prompts)",
        "target_audience": "Target audience description (calibrates tone and vocabulary)",
        "tone": "Caption voice and persona style",
        "platform": "Target platform (influences caption formatting and hashtags)",
        "caption_length": "Caption description length (short/medium/long)",
        "hashtag_count": "Number of hashtags to generate (3-7)",
        "custom_instructions": "User-defined instructions appended to all LLM prompts",
    }
    print()
    print("  Content Profile:")
    for field_name, desc in cp_descriptions.items():
        value = getattr(cp, field_name, "N/A")
        display = value if value else "(not set)"
        print(f"    {field_name}: {display}")
        print(f"      # {desc}")

    # SubtitleStyle section
    ss = config.subtitle_style
    ss_descriptions = {
        "font_name": "Font name for subtitles",
        "font_size_pct": "Font size as fraction of frame height",
        "primary_color": "Text color in ASS &HAABBGGRR format",
        "outline_color": "Outline color in ASS &HAABBGGRR format",
        "outline_width": "Outline thickness in pixels",
        "shadow": "Shadow depth in pixels",
        "margin_v_pct": "Vertical margin as fraction of frame height",
        "margin_h_pct": "Horizontal margin as fraction of frame width",
    }
    print()
    print("  Subtitle Style:")
    for field_name, desc in ss_descriptions.items():
        value = getattr(ss, field_name, "N/A")
        print(f"    {field_name}: {value}")
        print(f"      # {desc}")
    return 0


def _cmd_generate_config(args: argparse.Namespace) -> int:
    """Handle the 'generate-config' subcommand — write a default INI config."""
    from viral_clip_extractor.utils.config import save_default_config

    output_path = args.output
    if os.path.exists(output_path):
        print(f"Config file already exists: {output_path}")
        print("  Remove it first or specify a different --output path.")
        return 1

    save_default_config(output_path)
    print(f"Default config written to {output_path}")
    print("  Edit it, then pass with: --config " + output_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    verbose = getattr(args, "verbose", False)

    # Suppress known third-party warnings in non-verbose mode
    if not verbose:
        # pynvml is a transitive dep of torch; emits FutureWarning about package rename
        warnings.filterwarnings(
            "ignore", message=".*pynvml.*deprecated.*", category=FutureWarning
        )
        # librosa falls back to audioread when PySoundFile can't handle .mp4
        warnings.filterwarnings(
            "ignore",
            message=".*librosa.core.audio.__audioread_load.*",
            category=FutureWarning,
        )
        # PySoundFile emits UserWarning before falling back to audioread
        warnings.filterwarnings(
            "ignore", message="PySoundFile failed.*", category=UserWarning
        )

    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Bootstrap: check deps before running pipeline commands.
    # Cache result via marker file to avoid ~100ms startup penalty on
    # every invocation. Marker is valid for 24 hours.
    if args.command not in ("check", "show-config", "generate-config"):
        marker = Path.home() / ".vce_bootstrap_ok"
        skip_bootstrap = False
        if marker.exists():
            import time as _time
            age = _time.time() - marker.stat().st_mtime
            if age < 86400:  # 24 hours
                skip_bootstrap = True

        if not skip_bootstrap:
            from viral_clip_extractor.bootstrap import ensure_ready

            if not ensure_ready(verbose=verbose):
                print("Some required dependencies are missing. Run 'check' for details.")
                return 1
            try:
                marker.touch()
            except OSError:
                pass  # non-critical — just means next run rechecks

    # Validate --min-score range
    min_score = getattr(args, "min_score", 70.0)
    if min_score < 0 or min_score > 100:
        print(f"Error: --min-score must be between 0 and 100, got {min_score}")
        return 1

    handlers = {
        "process": _cmd_process,
        "youtube": _cmd_youtube,
        "batch": _cmd_batch,
        "check": _cmd_check,
        "show-config": _cmd_show_config,
        "generate-config": _cmd_generate_config,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
