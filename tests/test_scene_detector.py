"""Tests for the SceneDetector module.

All external dependencies (scenedetect, cv2) are mocked so these tests run
without hardware or video files.
"""

from unittest.mock import MagicMock, patch

import pytest

from yacg.models import SceneSegment


# ---------------------------------------------------------------------------
# Helpers — mock FrameTimecode objects returned by scenedetect.detect()
# ---------------------------------------------------------------------------


def _make_timecode(seconds: float) -> MagicMock:
    """Create a mock FrameTimecode that returns *seconds* from get_seconds()."""
    tc = MagicMock()
    tc.get_seconds.return_value = seconds
    return tc


def _scene_pair(start: float, end: float) -> tuple:
    """Return a (start_tc, end_tc) tuple mimicking scenedetect output."""
    return (_make_timecode(start), _make_timecode(end))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def detector():
    """Return a SceneDetector with default settings."""
    from yacg.core.scene_detector import SceneDetector

    return SceneDetector(threshold=3.0, min_scene_len=7.0, max_scene_len=60.0)


@pytest.fixture()
def detector_tight():
    """Return a SceneDetector with tight min/max for easier split/merge testing."""
    from yacg.core.scene_detector import SceneDetector

    return SceneDetector(threshold=3.0, min_scene_len=5.0, max_scene_len=20.0)


# ---------------------------------------------------------------------------
# Test: detect_scenes — happy path
# ---------------------------------------------------------------------------


@patch("yacg.core.scene_detector._HAS_SCENEDETECT", True)
@patch("yacg.core.scene_detector.detect")
@patch("yacg.core.scene_detector.AdaptiveDetector")
def test_detect_scenes_basic(mock_adaptive, mock_detect, detector, tmp_path):
    """Basic scene detection returns correctly typed SceneSegment objects."""
    # Create a dummy video file so the path check passes
    video_file = tmp_path / "test.mp4"
    video_file.write_bytes(b"\x00" * 100)

    mock_detect.return_value = [
        _scene_pair(0.0, 10.0),
        _scene_pair(10.0, 25.0),
        _scene_pair(25.0, 40.0),
    ]

    result = detector.detect_scenes(str(video_file))

    assert len(result) == 3
    for seg in result:
        assert isinstance(seg, SceneSegment)
    assert result[0].start_time == 0.0
    assert result[0].end_time == 10.0
    assert result[1].start_time == 10.0
    assert result[2].end_time == 40.0
    # Scene indices are sequential after re-indexing
    assert [s.scene_index for s in result] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Test: detect_scenes — single-scene / no boundaries
# ---------------------------------------------------------------------------


@patch("yacg.core.scene_detector._HAS_SCENEDETECT", True)
@patch("yacg.core.scene_detector.open_video")
@patch("yacg.core.scene_detector.detect")
@patch("yacg.core.scene_detector.AdaptiveDetector")
def test_detect_scenes_no_boundaries(
    mock_adaptive, mock_detect, mock_open_video, detector, tmp_path
):
    """When scenedetect finds no boundaries, returns the whole video as one scene."""
    video_file = tmp_path / "single.mp4"
    video_file.write_bytes(b"\x00" * 100)

    mock_detect.return_value = []

    # Mock open_video to return a video with known duration
    mock_video = MagicMock()
    mock_video.duration.get_seconds.return_value = 30.0
    mock_open_video.return_value = mock_video

    result = detector.detect_scenes(str(video_file))

    assert len(result) == 1
    assert result[0].start_time == 0.0
    assert result[0].end_time == 30.0
    assert result[0].scene_index == 0


# ---------------------------------------------------------------------------
# Test: detect_scenes — very short video
# ---------------------------------------------------------------------------


@patch("yacg.core.scene_detector._HAS_SCENEDETECT", True)
@patch("yacg.core.scene_detector.open_video")
@patch("yacg.core.scene_detector.detect")
@patch("yacg.core.scene_detector.AdaptiveDetector")
def test_detect_scenes_very_short_video(
    mock_adaptive, mock_detect, mock_open_video, detector, tmp_path
):
    """A video shorter than 0.1s returns an empty list."""
    video_file = tmp_path / "tiny.mp4"
    video_file.write_bytes(b"\x00" * 10)

    mock_detect.return_value = []

    mock_video = MagicMock()
    mock_video.duration.get_seconds.return_value = 0.05
    mock_open_video.return_value = mock_video

    result = detector.detect_scenes(str(video_file))

    assert result == []


# ---------------------------------------------------------------------------
# Test: file not found
# ---------------------------------------------------------------------------


@patch("yacg.core.scene_detector._HAS_SCENEDETECT", True)
def test_detect_scenes_file_not_found(detector):
    """Raises FileNotFoundError for a non-existent path."""
    with pytest.raises(FileNotFoundError, match="Video not found"):
        detector.detect_scenes("/nonexistent/path/video.mp4")


# ---------------------------------------------------------------------------
# Test: corrupt file / detection failure
# ---------------------------------------------------------------------------


@patch("yacg.core.scene_detector._HAS_SCENEDETECT", True)
@patch("yacg.core.scene_detector.detect")
@patch("yacg.core.scene_detector.AdaptiveDetector")
def test_detect_scenes_corrupt_file(mock_adaptive, mock_detect, detector, tmp_path):
    """Raises RuntimeError when scenedetect fails on a corrupt file."""
    video_file = tmp_path / "corrupt.mp4"
    video_file.write_bytes(b"\x00" * 100)

    mock_detect.side_effect = Exception("Failed to decode video")

    with pytest.raises(RuntimeError, match="Scene detection failed"):
        detector.detect_scenes(str(video_file))


# ---------------------------------------------------------------------------
# Test: missing scenedetect dependency
# ---------------------------------------------------------------------------


@patch("yacg.core.scene_detector._HAS_SCENEDETECT", False)
def test_detect_scenes_missing_dependency(detector):
    """Raises ImportError with install instructions when scenedetect is absent."""
    with pytest.raises(ImportError, match="pip install scenedetect"):
        detector.detect_scenes("any_path.mp4")


# ---------------------------------------------------------------------------
# Test: merge_short_scenes
# ---------------------------------------------------------------------------


def test_merge_short_scenes(detector):
    """Short scenes are merged into their neighbors."""
    scenes = [
        SceneSegment(start_time=0.0, end_time=3.0, scene_index=0),  # 3s — short
        SceneSegment(start_time=3.0, end_time=15.0, scene_index=1),  # 12s — ok
        SceneSegment(start_time=15.0, end_time=18.0, scene_index=2),  # 3s — short
        SceneSegment(start_time=18.0, end_time=30.0, scene_index=3),  # 12s — ok
    ]

    merged = detector.merge_short_scenes(scenes, min_duration=7.0)

    # First short scene (0-3) merged forward into (3-15) → (0-15)
    # Third short scene (15-18) merged into previous (0-15) → (0-18)
    # Actually: first short + second normal → merged to (0-15)
    # Third short merged into (0-15) → (0-18)
    # Fourth normal → added as-is (18-30)
    assert len(merged) == 2
    assert merged[0].start_time == 0.0
    assert merged[0].end_time == 18.0
    assert merged[1].start_time == 18.0
    assert merged[1].end_time == 30.0


def test_merge_short_scenes_single(detector):
    """A single scene is returned unchanged regardless of length."""
    scenes = [SceneSegment(start_time=0.0, end_time=2.0, scene_index=0)]
    merged = detector.merge_short_scenes(scenes, min_duration=7.0)

    assert len(merged) == 1
    assert merged[0].start_time == 0.0
    assert merged[0].end_time == 2.0


def test_merge_short_scenes_empty(detector):
    """Empty input returns empty output."""
    assert detector.merge_short_scenes([], min_duration=7.0) == []


# ---------------------------------------------------------------------------
# Test: split_long_scenes
# ---------------------------------------------------------------------------


def test_split_long_scenes(detector):
    """A 120s scene is split into two 60s halves at the default max."""
    scenes = [
        SceneSegment(start_time=0.0, end_time=120.0, scene_index=0),
    ]

    split = detector.split_long_scenes(scenes, max_duration=60.0)

    assert len(split) == 2
    assert split[0].start_time == 0.0
    assert split[0].end_time == 60.0
    assert split[1].start_time == 60.0
    assert split[1].end_time == 120.0


def test_split_long_scenes_recursive(detector):
    """A 200s scene is recursively split until all segments fit max=60."""
    scenes = [
        SceneSegment(start_time=0.0, end_time=200.0, scene_index=0),
    ]

    split = detector.split_long_scenes(scenes, max_duration=60.0)

    # 200 → 0-100, 100-200
    # 0-100 → 0-50, 50-100  (both <=60)
    # 100-200 → 100-150, 150-200  (both <=60)
    assert len(split) == 4
    for seg in split:
        assert seg.duration <= 60.0
    # Verify continuity
    assert split[0].start_time == 0.0
    assert split[-1].end_time == 200.0


def test_split_short_scene_unchanged(detector):
    """Scenes within max_duration are not split."""
    scenes = [
        SceneSegment(start_time=0.0, end_time=30.0, scene_index=0),
        SceneSegment(start_time=30.0, end_time=55.0, scene_index=1),
    ]

    split = detector.split_long_scenes(scenes, max_duration=60.0)

    assert len(split) == 2
    assert split[0].duration == 30.0
    assert split[1].duration == 25.0


# ---------------------------------------------------------------------------
# Test: full pipeline with merge + split
# ---------------------------------------------------------------------------


@patch("yacg.core.scene_detector._HAS_SCENEDETECT", True)
@patch("yacg.core.scene_detector.detect")
@patch("yacg.core.scene_detector.AdaptiveDetector")
def test_detect_scenes_merge_and_split(mock_adaptive, mock_detect, tmp_path):
    """End-to-end: short scenes merge, long scenes split, indices are sequential."""
    from yacg.core.scene_detector import SceneDetector

    video_file = tmp_path / "full.mp4"
    video_file.write_bytes(b"\x00" * 100)

    # 3s (short), 20s (ok), 100s (too long for max=60)
    mock_detect.return_value = [
        _scene_pair(0.0, 3.0),
        _scene_pair(3.0, 23.0),
        _scene_pair(23.0, 123.0),
    ]

    det = SceneDetector(threshold=3.0, min_scene_len=7.0, max_scene_len=60.0)
    result = det.detect_scenes(str(video_file))

    # First short scene (0-3) merges forward into (3-23) → (0-23)
    # Third long scene (23-123, 100s) splits: 23-73, 73-123 (both 50s <=60)
    # Total: 3 segments
    assert len(result) == 3
    assert result[0].start_time == 0.0
    assert result[0].end_time == 23.0
    # Indices are sequential
    assert [s.scene_index for s in result] == [0, 1, 2]
    # All return SceneSegment
    for seg in result:
        assert isinstance(seg, SceneSegment)


# ---------------------------------------------------------------------------
# Test: PipelineConfig integration
# ---------------------------------------------------------------------------


def test_init_with_pipeline_config():
    """SceneDetector reads parameters from PipelineConfig."""
    from yacg.core.scene_detector import SceneDetector
    from yacg.models import PipelineConfig

    config = PipelineConfig(
        scene_threshold=2.5,
        min_scene_len=10.0,
        max_scene_len=45.0,
    )
    det = SceneDetector(config=config)

    assert det.threshold == 2.5
    assert det.min_scene_len == 10.0
    assert det.max_scene_len == 45.0


def test_init_with_direct_params():
    """SceneDetector uses direct parameters when no config is given."""
    from yacg.core.scene_detector import SceneDetector

    det = SceneDetector(threshold=4.0, min_scene_len=8.0, max_scene_len=50.0)

    assert det.threshold == 4.0
    assert det.min_scene_len == 8.0
    assert det.max_scene_len == 50.0


# ---------------------------------------------------------------------------
# Test: _get_video_duration fallback
# ---------------------------------------------------------------------------


@patch("yacg.core.scene_detector._HAS_SCENEDETECT", True)
@patch("yacg.core.scene_detector.open_video")
@patch("yacg.core.scene_detector.detect")
@patch("yacg.core.scene_detector.AdaptiveDetector")
def test_duration_fallback_failure(
    mock_adaptive, mock_detect, mock_open_video, detector, tmp_path
):
    """When open_video fails to get duration, returns empty list."""
    video_file = tmp_path / "broken.mp4"
    video_file.write_bytes(b"\x00" * 100)

    mock_detect.return_value = []
    mock_open_video.side_effect = Exception("Cannot open video")

    result = detector.detect_scenes(str(video_file))

    assert result == []
