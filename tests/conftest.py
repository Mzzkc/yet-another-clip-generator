"""
Shared test fixtures for the Viral Clip Extractor test suite.

Provides session-scoped fixtures for synthetic and real video test files.
Synthetic fixtures are auto-created via FFmpeg; the rickroll fixture
requires the preprocessing stage to have downloaded it.
"""

import subprocess
from pathlib import Path

import pytest

FIXTURE_DIR = Path("/tmp/vce_test_fixtures")


@pytest.fixture(scope="session")
def fixture_dir():
    """Return the shared fixture directory, creating synthetic fixtures if missing."""
    FIXTURE_DIR.mkdir(exist_ok=True)
    _ensure_synthetic_fixtures(FIXTURE_DIR)
    return FIXTURE_DIR


@pytest.fixture(scope="session")
def synthetic_1s(fixture_dir):
    """1-second black+silence video, 320x240, h264+aac."""
    return fixture_dir / "synthetic_1s.mp4"


@pytest.fixture(scope="session")
def synthetic_noaudio(fixture_dir):
    """5-second blue video with no audio track, 320x240."""
    return fixture_dir / "synthetic_noaudio.mp4"


@pytest.fixture(scope="session")
def synthetic_singleframe(fixture_dir):
    """Single-frame (~0.04s) red video, 320x240."""
    return fixture_dir / "synthetic_singleframe.mp4"


@pytest.fixture(scope="session")
def rickroll_30s(fixture_dir):
    """30-second clip of the rickroll video. Skips if not available."""
    path = fixture_dir / "rickroll_30s.mp4"
    if not path.exists():
        pytest.skip("rickroll_30s.mp4 not available (run preprocessing to create)")
    return path


def _ensure_synthetic_fixtures(d: Path):
    """Create synthetic test fixtures via FFmpeg if they don't exist."""
    # synthetic_1s.mp4: 1s black+silence, 320x240
    if not (d / "synthetic_1s.mp4").exists():
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "color=black:s=320x240:d=1",
                "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                "-t", "1", "-c:v", "libx264", "-c:a", "aac",
                str(d / "synthetic_1s.mp4"),
            ],
            capture_output=True,
        )

    # synthetic_noaudio.mp4: 5s blue, no audio
    if not (d / "synthetic_noaudio.mp4").exists():
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "color=blue:s=320x240:d=5",
                "-c:v", "libx264", "-an",
                str(d / "synthetic_noaudio.mp4"),
            ],
            capture_output=True,
        )

    # synthetic_singleframe.mp4: ~0.04s red
    if not (d / "synthetic_singleframe.mp4").exists():
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "color=red:s=320x240:d=0.04",
                "-c:v", "libx264",
                str(d / "synthetic_singleframe.mp4"),
            ],
            capture_output=True,
        )
