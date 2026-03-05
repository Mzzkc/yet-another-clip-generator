#!/usr/bin/env python3
"""
Helper script for the Viral Clip Extractor.

Provides a menu-driven interface for processing videos, running system
checks, viewing results, and managing configuration.
"""

import os
import subprocess
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_banner() -> None:
    """Print the application banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
+===========================================================+
|                                                           |
|      VIRAL CLIP EXTRACTOR                                 |
|      AI-Powered Viral Clip Detection & Extraction         |
|      ASMR-Optimized for Instagram Reels                   |
|                                                           |
+===========================================================+
{Colors.ENDC}"""
    print(banner)


def print_menu() -> None:
    """Print the main menu."""
    menu = f"""
{Colors.BOLD}Main Menu:{Colors.ENDC}

{Colors.GREEN}1.{Colors.ENDC} Process a local video file
{Colors.GREEN}2.{Colors.ENDC} Process a YouTube video
{Colors.GREEN}3.{Colors.ENDC} Batch process video directory
{Colors.GREEN}4.{Colors.ENDC} Run system verification
{Colors.GREEN}5.{Colors.ENDC} View/edit configuration
{Colors.GREEN}6.{Colors.ENDC} View recent results
{Colors.GREEN}7.{Colors.ENDC} Quick test (first 5 minutes)
{Colors.RED}8.{Colors.ENDC} Exit
"""
    print(menu)


def run_command(cmd: str, description: str) -> bool:
    """Run a shell command and display output."""
    print(f"\n{Colors.CYAN}-> {description}...{Colors.ENDC}\n")
    try:
        result = subprocess.run(cmd, shell=True)
        return result.returncode == 0
    except Exception as exc:
        print(f"{Colors.RED}Error: {exc}{Colors.ENDC}")
        return False


def get_input(prompt: str, default: str = "") -> str:
    """Get user input with optional default value."""
    if default:
        display = f"{Colors.BOLD}{prompt} [{default}]: {Colors.ENDC}"
    else:
        display = f"{Colors.BOLD}{prompt}: {Colors.ENDC}"
    value = input(display).strip()
    return value if value else default


def validate_file_path(path: str, extensions: set[str] | None = None) -> bool:
    """Validate that a file path exists and has a valid extension."""
    p = Path(path)
    if not p.exists():
        print(f"{Colors.RED}File not found: {path}{Colors.ENDC}")
        return False
    if not p.is_file():
        print(f"{Colors.RED}Not a file: {path}{Colors.ENDC}")
        return False
    if extensions and p.suffix.lower() not in extensions:
        print(f"{Colors.RED}Invalid file type: {p.suffix} (expected: {', '.join(extensions)}){Colors.ENDC}")
        return False
    return True


def validate_directory(path: str) -> bool:
    """Validate that a directory exists."""
    p = Path(path)
    if not p.exists():
        print(f"{Colors.RED}Directory not found: {path}{Colors.ENDC}")
        return False
    if not p.is_dir():
        print(f"{Colors.RED}Not a directory: {path}{Colors.ENDC}")
        return False
    return True


# -- Menu handlers -----------------------------------------------------------


def handle_process_local() -> None:
    """Process a local video file."""
    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    path = get_input("Video file path")
    if not path or not validate_file_path(path, video_extensions):
        return

    title = get_input("Video title (optional)", Path(path).stem)
    output_dir = get_input("Output directory", "./clip_output")
    top_n = get_input("Number of top clips", "10")
    min_score = get_input("Minimum virality score (0-100)", "70")

    cmd = (
        f"PYTHONPATH={os.getcwd()} python -m viral_clip_extractor process "
        f'--video "{path}" --title "{title}" '
        f'--output-dir "{output_dir}" --top-n {top_n} --min-score {min_score}'
    )

    print(f"\n{Colors.CYAN}Command: {cmd}{Colors.ENDC}")
    run_command(cmd, f"Processing {Path(path).name}")


def handle_process_youtube() -> None:
    """Process a YouTube video."""
    url = get_input("YouTube URL")
    if not url:
        print(f"{Colors.RED}No URL provided{Colors.ENDC}")
        return

    if "youtube.com" not in url and "youtu.be" not in url:
        print(f"{Colors.YELLOW}Warning: URL doesn't look like a YouTube link{Colors.ENDC}")
        confirm = get_input("Continue anyway? (y/n)", "n")
        if confirm.lower() != "y":
            return

    output_dir = get_input("Output directory", "./clip_output")
    top_n = get_input("Number of top clips", "10")
    min_score = get_input("Minimum virality score (0-100)", "70")

    cmd = (
        f"PYTHONPATH={os.getcwd()} python -m viral_clip_extractor youtube "
        f'--url "{url}" --output-dir "{output_dir}" --top-n {top_n} --min-score {min_score}'
    )

    print(f"\n{Colors.CYAN}Command: {cmd}{Colors.ENDC}")
    run_command(cmd, "Processing YouTube video")


def handle_batch_process() -> None:
    """Batch process a directory of videos."""
    videos_dir = get_input("Videos directory path")
    if not videos_dir or not validate_directory(videos_dir):
        return

    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    video_files = [
        f for f in Path(videos_dir).iterdir()
        if f.suffix.lower() in video_extensions
    ]

    if not video_files:
        print(f"{Colors.YELLOW}No video files found in {videos_dir}{Colors.ENDC}")
        return

    print(f"\n{Colors.GREEN}Found {len(video_files)} video file(s):{Colors.ENDC}")
    for vf in sorted(video_files)[:10]:
        size_mb = vf.stat().st_size / (1024 * 1024)
        print(f"  {vf.name} ({size_mb:.1f} MB)")
    if len(video_files) > 10:
        print(f"  ... and {len(video_files) - 10} more")

    output_dir = get_input("Output directory", "./clip_output")
    top_n = get_input("Number of top clips per video", "10")
    min_score = get_input("Minimum virality score (0-100)", "70")

    confirm = get_input(f"Process {len(video_files)} videos? (y/n)", "y")
    if confirm.lower() != "y":
        return

    cmd = (
        f"PYTHONPATH={os.getcwd()} python -m viral_clip_extractor batch "
        f'--videos-dir "{videos_dir}" --output-dir "{output_dir}" '
        f"--top-n {top_n} --min-score {min_score}"
    )

    run_command(cmd, f"Batch processing {len(video_files)} videos")


def handle_system_check() -> None:
    """Run the system verification script."""
    script_path = Path(__file__).parent / "test_clip_system.py"
    if script_path.exists():
        run_command(f"python {script_path}", "Running system verification")
    else:
        run_command(
            f"PYTHONPATH={os.getcwd()} python -m viral_clip_extractor check",
            "Running dependency check",
        )


def handle_view_config() -> None:
    """View or create configuration file."""
    config_path = Path("config.ini")

    if config_path.exists():
        print(f"\n{Colors.BOLD}Current Configuration ({config_path}):{Colors.ENDC}\n")
        print(config_path.read_text(encoding="utf-8"))
    else:
        print(f"\n{Colors.YELLOW}No config.ini found — using defaults.{Colors.ENDC}")
        create = get_input("Create a default config.ini? (y/n)", "y")
        if create.lower() == "y":
            try:
                from viral_clip_extractor.utils.config import save_default_config

                save_default_config("config.ini")
                print(f"{Colors.GREEN}Created config.ini with defaults{Colors.ENDC}")
                print(f"\n{config_path.read_text(encoding='utf-8')}")
            except Exception as exc:
                print(f"{Colors.RED}Error creating config: {exc}{Colors.ENDC}")
        return

    edit = get_input("Edit config with nano/vim? (y/n)", "n")
    if edit.lower() == "y":
        editor = os.environ.get("EDITOR", "nano")
        subprocess.run([editor, str(config_path)])


def handle_view_results() -> None:
    """View recent extraction results."""
    output_dir = Path("./clip_output")

    if not output_dir.exists():
        print(f"\n{Colors.YELLOW}No output directory found. Run a video first.{Colors.ENDC}")
        return

    # List clip files
    clips = sorted(output_dir.glob("*.mp4"))
    csv_files = sorted(output_dir.glob("*.csv"))

    print(f"\n{Colors.BOLD}Output Directory: {output_dir.resolve()}{Colors.ENDC}\n")

    if clips:
        print(f"{Colors.GREEN}Extracted Clips ({len(clips)}):{Colors.ENDC}")
        for clip in clips[:20]:
            size_mb = clip.stat().st_size / (1024 * 1024)
            print(f"  {clip.name} ({size_mb:.1f} MB)")
        if len(clips) > 20:
            print(f"  ... and {len(clips) - 20} more")
    else:
        print(f"{Colors.YELLOW}No clips extracted yet.{Colors.ENDC}")

    if csv_files:
        print(f"\n{Colors.GREEN}CSV Reports:{Colors.ENDC}")
        for csv_file in csv_files:
            print(f"  {csv_file.name}")
            try:
                lines = csv_file.read_text(encoding="utf-8").strip().split("\n")
                print(f"    Header: {lines[0][:80]}...")
                print(f"    Entries: {len(lines) - 1}")
            except Exception:
                pass
    else:
        print(f"\n{Colors.YELLOW}No CSV reports yet.{Colors.ENDC}")


def handle_quick_test() -> None:
    """Process the first 5 minutes of a video as a quick test."""
    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    path = get_input("Video file path for quick test")
    if not path or not validate_file_path(path, video_extensions):
        return

    title = get_input("Video title (optional)", Path(path).stem)
    output_dir = get_input("Output directory", "./clip_output/quick_test")

    # Use ffmpeg to extract first 5 minutes, then process that
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, prefix="quicktest_") as tmp:
        tmp_path = tmp.name

    print(f"\n{Colors.CYAN}Extracting first 5 minutes...{Colors.ENDC}")
    extract_cmd = (
        f'ffmpeg -y -i "{path}" -t 300 -c copy "{tmp_path}" -loglevel error'
    )
    result = subprocess.run(extract_cmd, shell=True)

    if result.returncode != 0:
        print(f"{Colors.RED}FFmpeg extraction failed. Processing full video instead.{Colors.ENDC}")
        tmp_path = path

    cmd = (
        f"PYTHONPATH={os.getcwd()} python -m viral_clip_extractor process "
        f'--video "{tmp_path}" --title "{title} (Quick Test)" '
        f'--output-dir "{output_dir}" --top-n 3 --min-score 50'
    )

    run_command(cmd, "Quick test processing")

    # Cleanup temp file if we created one
    if tmp_path != path:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def main() -> None:
    """Main menu loop."""
    while True:
        print_banner()
        print_menu()

        try:
            choice = input(f"{Colors.BOLD}Select option (1-8): {Colors.ENDC}").strip()

            if choice == "1":
                handle_process_local()
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")

            elif choice == "2":
                handle_process_youtube()
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")

            elif choice == "3":
                handle_batch_process()
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")

            elif choice == "4":
                handle_system_check()
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")

            elif choice == "5":
                handle_view_config()
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")

            elif choice == "6":
                handle_view_results()
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")

            elif choice == "7":
                handle_quick_test()
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")

            elif choice == "8":
                print(f"\n{Colors.GREEN}Goodbye!{Colors.ENDC}\n")
                break

            else:
                print(f"{Colors.RED}Invalid option. Please select 1-8.{Colors.ENDC}")
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")

        except KeyboardInterrupt:
            print(f"\n\n{Colors.GREEN}Goodbye!{Colors.ENDC}\n")
            break
        except Exception as exc:
            print(f"{Colors.RED}Error: {exc}{Colors.ENDC}")
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")


if __name__ == "__main__":
    main()
