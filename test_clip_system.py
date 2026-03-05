#!/usr/bin/env python3
"""
System verification script for the Viral Clip Extractor.

Checks that all required and optional dependencies are installed,
services are running, and the environment is ready for clip extraction.
"""

import shutil
import subprocess
import sys
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str) -> None:
    """Print a styled section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.ENDC}\n")


def print_test(name: str, passed: bool, details: str = "", fix: str = "") -> None:
    """Print a single test result with optional fix suggestion."""
    status = f"{Colors.GREEN}PASS{Colors.ENDC}" if passed else f"{Colors.RED}FAIL{Colors.ENDC}"
    print(f"  [{status}] {name}")
    if details:
        print(f"         {details}")
    if not passed and fix:
        print(f"         {Colors.YELLOW}Fix: {fix}{Colors.ENDC}")


def check_python_version() -> bool:
    """Check Python version >= 3.10."""
    version = sys.version_info
    passed = version >= (3, 10)
    details = f"Python {version.major}.{version.minor}.{version.micro}"
    fix = "Install Python 3.10+: https://www.python.org/downloads/"
    print_test("Python Version (>= 3.10)", passed, details, fix)
    return passed


def check_ffmpeg() -> bool:
    """Check that FFmpeg is available."""
    if not shutil.which("ffmpeg"):
        print_test("FFmpeg Available", False, "Not found in PATH", "sudo apt install ffmpeg")
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=5
        )
        version_line = result.stdout.split("\n")[0] if result.stdout else "unknown"
        print_test("FFmpeg Available", True, version_line)
        return True
    except Exception as exc:
        print_test("FFmpeg Available", False, str(exc), "sudo apt install ffmpeg")
        return False


def check_ffprobe() -> bool:
    """Check that FFprobe is available."""
    if not shutil.which("ffprobe"):
        print_test("FFprobe Available", False, "Not found in PATH", "sudo apt install ffmpeg")
        return False
    print_test("FFprobe Available", True)
    return True


def check_package(import_name: str, display_name: str, pip_name: str) -> bool:
    """Check if a Python package is importable."""
    try:
        __import__(import_name)
        print_test(f"{display_name} Installed", True)
        return True
    except ImportError:
        print_test(
            f"{display_name} Installed", False,
            "Not installed",
            f"pip install {pip_name}",
        )
        return False


def check_optional_package(import_name: str, display_name: str, pip_name: str) -> bool:
    """Check an optional Python package — PASS with warning if missing."""
    try:
        __import__(import_name)
        print_test(f"{display_name} (optional)", True)
        return True
    except ImportError:
        print(
            f"  [{Colors.YELLOW}SKIP{Colors.ENDC}] {display_name} (optional)"
        )
        print(f"         Not installed — pip install {pip_name}")
        return True  # Optional, so doesn't count as failure


def check_ollama_running() -> bool:
    """Check if the Ollama service is running."""
    try:
        import requests

        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            print_test(
                "Ollama Service Running", True,
                f"{len(models)} model(s) available",
            )
            return True
        print_test(
            "Ollama Service Running", False,
            f"HTTP {resp.status_code}",
            "Start with: ollama serve",
        )
        return False
    except ImportError:
        print_test(
            "Ollama Service Running", False,
            "requests module not installed",
            "pip install requests",
        )
        return False
    except Exception:
        print_test(
            "Ollama Service Running", False,
            "Cannot connect to localhost:11434",
            "Start with: ollama serve",
        )
        return False


def check_qwen_model() -> bool:
    """Check if Qwen2.5-VL model is available in Ollama."""
    try:
        import requests

        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code != 200:
            print_test(
                "Qwen2.5-VL Model Available", False,
                "Cannot query Ollama",
                "ollama pull qwen2.5-vl:7b",
            )
            return False

        models = [m["name"] for m in resp.json().get("models", [])]
        qwen_models = [n for n in models if "qwen" in n.lower() and "vl" in n.lower()]

        if qwen_models:
            print_test("Qwen2.5-VL Model Available", True, f"Found: {', '.join(qwen_models)}")
            return True

        print_test(
            "Qwen2.5-VL Model Available", False,
            f"Available models: {', '.join(models) or 'none'}",
            "ollama pull qwen2.5-vl:7b",
        )
        return False
    except Exception:
        print_test(
            "Qwen2.5-VL Model Available", False,
            "Ollama not reachable",
            "ollama serve && ollama pull qwen2.5-vl:7b",
        )
        return False


def check_output_dir_writable() -> bool:
    """Check that the output directory is writable."""
    output_dir = Path("./clip_output")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        print_test("Output Directory Writable", True, str(output_dir.resolve()))
        return True
    except Exception as exc:
        print_test(
            "Output Directory Writable", False,
            str(exc),
            f"mkdir -p {output_dir} && chmod 755 {output_dir}",
        )
        return False


def check_disk_space() -> bool:
    """Check that at least 2 GB of disk space is free."""
    try:
        stat = shutil.disk_usage(".")
        free_gb = stat.free / (1024 ** 3)
        passed = free_gb > 2.0
        details = f"{free_gb:.1f} GB free"
        fix = "Free up disk space (need > 2 GB)"
        print_test("Disk Space (> 2 GB)", passed, details, fix)
        return passed
    except Exception as exc:
        print_test("Disk Space (> 2 GB)", False, str(exc))
        return False


def check_yt_transcriber() -> bool:
    """Check if yt-transcriber is available (optional)."""
    yt_path = Path("/home/emzi/Projects/yt-transcriber")
    script_path = yt_path / "transcribe.py"
    if script_path.is_file():
        print_test("yt-transcriber (optional)", True, str(yt_path))
        return True
    print(
        f"  [{Colors.YELLOW}SKIP{Colors.ENDC}] yt-transcriber (optional)"
    )
    print(f"         Not found at {yt_path}")
    return True  # Optional


def check_faster_whisper() -> bool:
    """Check if faster-whisper is available (optional)."""
    return check_optional_package("faster_whisper", "faster-whisper", "faster-whisper")


def run_all_checks() -> bool:
    """Run all system checks and print summary."""
    print_header("Viral Clip Extractor - System Verification")

    results: dict[str, bool] = {}

    # Required checks
    print(f"{Colors.BOLD}Required Dependencies:{Colors.ENDC}")
    results["python"] = check_python_version()
    results["ffmpeg"] = check_ffmpeg()
    results["ffprobe"] = check_ffprobe()
    results["scenedetect"] = check_package("scenedetect", "scenedetect", "scenedetect[opencv]")
    results["librosa"] = check_package("librosa", "librosa", "librosa")
    results["cv2"] = check_package("cv2", "OpenCV (cv2)", "opencv-python-headless")
    results["numpy"] = check_package("numpy", "numpy", "numpy")
    results["requests"] = check_package("requests", "requests", "requests")
    results["yt_dlp"] = check_package("yt_dlp", "yt-dlp", "yt-dlp")

    print(f"\n{Colors.BOLD}Services:{Colors.ENDC}")
    results["ollama"] = check_ollama_running()
    results["qwen"] = check_qwen_model()

    print(f"\n{Colors.BOLD}Environment:{Colors.ENDC}")
    results["output_dir"] = check_output_dir_writable()
    results["disk_space"] = check_disk_space()

    print(f"\n{Colors.BOLD}Optional Dependencies:{Colors.ENDC}")
    check_yt_transcriber()
    check_faster_whisper()

    # Summary
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{'SUMMARY':^60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    if passed == total:
        print(
            f"{Colors.GREEN}{Colors.BOLD}ALL CHECKS PASSED ({passed}/{total}){Colors.ENDC}"
        )
        print(f"\n{Colors.GREEN}System is ready! Quick start:{Colors.ENDC}")
        print("  python -m viral_clip_extractor process --video YOUR_VIDEO.mp4 --title \"Title\"")
        print("  python -m viral_clip_extractor youtube --url https://youtube.com/watch?v=XXXXX")
        return True

    print(
        f"{Colors.YELLOW}{Colors.BOLD}SOME CHECKS FAILED ({passed}/{total} passed){Colors.ENDC}\n"
    )
    print("Fix the failing checks above and re-run this script.")
    return False


if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
