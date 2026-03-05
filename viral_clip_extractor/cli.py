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
from pathlib import Path


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by process/youtube/batch subcommands."""
    parser.add_argument(
        "--output-dir", default="./clip_output",
        help="Output directory for clips and reports (default: ./clip_output)",
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Number of top clips to extract (default: 10)",
    )
    parser.add_argument(
        "--min-score", type=float, default=70.0,
        help="Minimum virality score threshold 0-100 (default: 70)",
    )
    parser.add_argument(
        "--model", default="qwen2.5-vl:7b",
        help="Ollama model name (default: qwen2.5-vl:7b)",
    )
    parser.add_argument(
        "--no-captions", action="store_true",
        help="Disable caption generation",
    )
    parser.add_argument(
        "--no-semantic", action="store_true",
        help="Disable semantic (LLM) analysis",
    )
    parser.add_argument(
        "--no-vertical", action="store_true",
        help="Keep original aspect ratio (skip 9:16 crop)",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to config INI file (uses defaults if omitted)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )


def _build_config(args: argparse.Namespace):
    """Build PipelineConfig from parsed CLI arguments."""
    from viral_clip_extractor.utils.config import load_config

    config = load_config(getattr(args, "config", None))

    # CLI overrides
    if hasattr(args, "output_dir") and args.output_dir:
        config.output_dir = args.output_dir
    if hasattr(args, "model") and args.model:
        config.model_name = args.model
    if getattr(args, "no_captions", False):
        config.enable_captions = False
    if getattr(args, "no_semantic", False):
        config.enable_semantic = False
    if getattr(args, "no_vertical", False):
        config.vertical_crop = False

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

    return parser


def _cmd_process(args: argparse.Namespace) -> int:
    """Handle the 'process' subcommand."""
    from viral_clip_extractor.pipeline import ViralClipPipeline

    config = _build_config(args)
    pipeline = ViralClipPipeline(config=config)

    result = pipeline.process_video(
        video_path=args.video,
        title=args.title,
        top_n=args.top_n,
        min_score=args.min_score,
    )

    if result.errors:
        for err in result.errors:
            print(f"  Error: {err}")

    return 0 if not result.errors or result.clips else 1


def _cmd_youtube(args: argparse.Namespace) -> int:
    """Handle the 'youtube' subcommand."""
    from viral_clip_extractor.pipeline import ViralClipPipeline

    config = _build_config(args)
    pipeline = ViralClipPipeline(config=config)

    result = pipeline.process_youtube(
        url=args.url,
        top_n=args.top_n,
        min_score=args.min_score,
    )

    if result.errors:
        for err in result.errors:
            print(f"  Error: {err}")

    return 0 if not result.errors or result.clips else 1


def _cmd_batch(args: argparse.Namespace) -> int:
    """Handle the 'batch' subcommand."""
    from viral_clip_extractor.pipeline import ViralClipPipeline

    videos_dir = Path(args.videos_dir)
    if not videos_dir.is_dir():
        print(f"Error: {args.videos_dir} is not a directory")
        return 1

    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    video_files = sorted(
        f for f in videos_dir.iterdir()
        if f.suffix.lower() in video_extensions
    )

    if not video_files:
        print(f"No video files found in {args.videos_dir}")
        return 1

    print(f"Found {len(video_files)} video files")
    config = _build_config(args)

    success = 0
    failed = 0

    for i, video_path in enumerate(video_files):
        print(f"\n{'=' * 60}")
        print(f"Processing {i + 1}/{len(video_files)}: {video_path.name}")
        print(f"{'=' * 60}")

        # Per-video output directory
        video_output = os.path.join(config.output_dir, video_path.stem)
        video_config = _build_config(args)
        video_config.output_dir = video_output

        pipeline = ViralClipPipeline(config=video_config)
        title = video_path.stem.replace("_", " ").replace("-", " ").title()

        result = pipeline.process_video(
            video_path=str(video_path),
            title=title,
            top_n=args.top_n,
            min_score=args.min_score,
        )

        if result.clips:
            success += 1
        else:
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Batch complete: {success} succeeded, {failed} failed")
    print(f"{'=' * 60}")

    return 0


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
        except Exception:
            print("  ffmpeg: OK")
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
    ]

    optional_packages = [
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

    for import_name, pip_name in optional_packages:
        try:
            __import__(import_name)
            print(f"  {pip_name}: OK (optional)")
        except ImportError:
            print(f"  {pip_name}: NOT FOUND (optional, pip install {pip_name})")

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
        else:
            print("  Ollama: ERROR (unexpected response)")
    except Exception:
        print("  Ollama: NOT RUNNING (start with: ollama serve)")

    if all_ok:
        print("\nAll required dependencies satisfied.")
    else:
        print("\nSome required dependencies are missing. Install them and try again.")

    return 0 if all_ok else 1


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

    # Configure logging
    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    handlers = {
        "process": _cmd_process,
        "youtube": _cmd_youtube,
        "batch": _cmd_batch,
        "check": _cmd_check,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
