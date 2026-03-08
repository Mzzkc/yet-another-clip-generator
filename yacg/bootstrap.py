"""
Self-bootstrapping dependency management for the YACG.

Checks for all required dependencies at startup and installs anything
missing, following the pattern from yt-transcriber. Call ensure_ready()
before importing any heavy modules.
"""

import importlib
import platform
import shutil
import site
import subprocess
import sys


# Map: import name -> pip package name
PYTHON_DEPS = {
    "numpy": "numpy>=1.24",
    "cv2": "opencv-python-headless>=4.8",
    "librosa": "librosa>=0.10",
    "scenedetect": "scenedetect[opencv]>=0.6",
    "requests": "requests>=2.28",
    "yt_dlp": "yt-dlp",
    "faster_whisper": "faster-whisper",
}

_bootstrapped = False


def get_os_info() -> dict:
    """Detect OS and package manager."""
    system = platform.system().lower()
    info = {"system": system, "distro": None, "pkg_manager": None, "wsl": False}

    if system == "linux":
        try:
            with open("/etc/os-release") as f:
                release = f.read().lower()
            if "ubuntu" in release or "debian" in release:
                info["distro"] = "debian"
                info["pkg_manager"] = "apt"
            elif "fedora" in release or "rhel" in release or "centos" in release:
                info["distro"] = "redhat"
                info["pkg_manager"] = "dnf"
            elif "arch" in release:
                info["distro"] = "arch"
                info["pkg_manager"] = "pacman"
            elif "opensuse" in release or "suse" in release:
                info["distro"] = "suse"
                info["pkg_manager"] = "zypper"
        except FileNotFoundError:
            pass

        try:
            with open("/proc/version") as f:
                if "microsoft" in f.read().lower():
                    info["wsl"] = True
        except FileNotFoundError:
            pass
    elif system == "darwin":
        info["distro"] = "macos"
        info["pkg_manager"] = "brew"

    return info


def check_ffmpeg() -> bool:
    """Check if ffmpeg and ffprobe are available."""
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def print_ffmpeg_instructions(os_info: dict) -> None:
    """Print OS-specific ffmpeg install instructions."""
    print("\n" + "=" * 60)
    print("  ffmpeg is required but not found!")
    print("=" * 60)

    instructions = {
        "debian": "sudo apt update && sudo apt install -y ffmpeg",
        "redhat": "sudo dnf install -y ffmpeg",
        "arch": "sudo pacman -S ffmpeg",
        "suse": "sudo zypper install ffmpeg",
        "macos": "brew install ffmpeg",
    }

    distro = os_info.get("distro")
    if distro and distro in instructions:
        print(f"\n  Detected: {distro}")
        print(f"  Run:  {instructions[distro]}")
    else:
        print("\n  Install ffmpeg via your package manager:")
        for name, cmd in instructions.items():
            print(f"    {name:>10}: {cmd}")

    if os_info.get("system") == "windows":
        print("  Windows: scoop install ffmpeg  OR  choco install ffmpeg")

    print("=" * 60 + "\n")


def _in_virtualenv() -> bool:
    """Check if running inside a virtualenv, venv, or conda environment."""
    import os
    return (
        hasattr(sys, "real_prefix")  # virtualenv
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)  # venv
        or os.environ.get("CONDA_DEFAULT_ENV") is not None  # conda
    )


def _pip_install(packages: list[str]) -> bool:
    """Install packages via pip with multiple fallback strategies.

    Tries safe strategies first (plain pip, --user). Only falls back to
    --break-system-packages as a last resort with a warning, since it can
    corrupt system Python on PEP 668 systems (Debian 12+, Ubuntu 23.04+,
    Fedora 38+).
    """
    # Safe strategies first; --break-system-packages only as last resort
    strategies: list[tuple[list[str], bool]] = [
        ([sys.executable, "-m", "pip", "install", *packages], False),
        ([sys.executable, "-m", "pip", "install", "--user", *packages], False),
    ]

    # Only add --break-system-packages if NOT in a virtualenv
    if not _in_virtualenv():
        strategies.append(
            ([sys.executable, "-m", "pip", "install", "--user", "--break-system-packages", *packages], True),
        )

    for cmd, is_break_system in strategies:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                if is_break_system:
                    print(
                        "  Warning: installed with --break-system-packages. "
                        "Consider using a virtualenv to avoid modifying system Python."
                    )
                # Refresh import paths
                importlib.invalidate_caches()
                user_site = site.getusersitepackages()
                if user_site not in sys.path:
                    sys.path.insert(0, user_site)
                return True
        except (subprocess.TimeoutExpired, Exception):
            continue

    return False


def ensure_python_deps() -> bool:
    """Check and install missing Python dependencies."""
    missing = []
    for import_name, pip_name in PYTHON_DEPS.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)

    if not missing:
        return True

    print(f"  Installing missing packages: {', '.join(missing)}")
    if _pip_install(missing):
        # Verify they actually import now
        still_missing = []
        for import_name, pip_name in PYTHON_DEPS.items():
            try:
                __import__(import_name)
            except ImportError:
                still_missing.append(pip_name)

        if still_missing:
            print(f"  Failed to install: {', '.join(still_missing)}")
            return False

        print("  All dependencies installed successfully")
        return True

    print("  Failed to install dependencies via pip")
    return False


def check_ollama(host: str = "http://localhost:11434") -> dict:
    """Check Ollama availability and loaded models.

    Returns a dict with 'available' (bool), 'models' (list), and
    optionally 'error' (str) describing why the check failed.
    """
    try:
        import requests
        resp = requests.get(f"{host}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return {"available": True, "models": models}
        return {
            "available": False,
            "models": [],
            "error": f"Ollama returned HTTP {resp.status_code}",
        }
    except ImportError:
        return {"available": False, "models": [], "error": "requests package not installed"}
    except Exception as exc:
        return {"available": False, "models": [], "error": str(exc)}


def ensure_ready(verbose: bool = True) -> bool:
    """Run all bootstrap checks. Call this before using the pipeline.

    Returns True if all required dependencies are satisfied.
    """
    global _bootstrapped
    if _bootstrapped:
        return True

    ok = True

    # 1. Check ffmpeg
    if not check_ffmpeg():
        if verbose:
            print_ffmpeg_instructions(get_os_info())
        ok = False

    # 2. Check/install Python deps
    if not ensure_python_deps():
        if verbose:
            print("  Some required Python packages could not be installed.")
            print("  Try manually: pip install -r requirements-clip-extractor.txt")
        ok = False

    if ok:
        _bootstrapped = True

    return ok
