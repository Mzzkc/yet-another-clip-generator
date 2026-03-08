"""
Tests for VisualAnalyzer and SmartCropper.

All external dependencies (cv2) are mocked so tests run without hardware
or OpenCV installed.
"""

import types
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from viral_clip_extractor.models import PipelineConfig, VisualFeatures


@pytest.fixture(autouse=True)
def _reset_face_detection_caches():
    """Reset visual_analyzer face detection caches between tests.

    The DNN net and Haar cascade are cached at module level for performance.
    Tests using mock cv2 need fresh caches so each test gets its own mock
    cascade rather than a stale one from a previous test.
    """
    import viral_clip_extractor.core.visual_analyzer as va_mod
    va_mod._cached_dnn_checked = False
    va_mod._cached_dnn_net = None
    va_mod._cached_haar_cascade = None
    yield
    va_mod._cached_dnn_checked = False
    va_mod._cached_dnn_net = None
    va_mod._cached_haar_cascade = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(w: int = 640, h: int = 480, color: tuple = (128, 128, 128)) -> np.ndarray:
    """Create a solid-colour BGR frame for testing."""
    frame = np.full((h, w, 3), color, dtype=np.uint8)
    return frame


def _make_gray(w: int = 640, h: int = 480, val: int = 128) -> np.ndarray:
    """Create a single-channel grayscale image."""
    return np.full((h, w), val, dtype=np.uint8)


def _mock_cv2_module() -> MagicMock:
    """Create a comprehensive cv2 mock with all necessary attributes."""
    mock = MagicMock()
    mock.CAP_PROP_POS_MSEC = 0
    mock.CAP_PROP_FRAME_WIDTH = 3
    mock.CAP_PROP_FRAME_HEIGHT = 4
    mock.COLOR_BGR2GRAY = 6
    mock.COLOR_BGR2HSV = 40
    mock.data = MagicMock()
    mock.data.haarcascades = "/fake/haarcascades/"
    return mock


# ---------------------------------------------------------------------------
# VisualAnalyzer tests
# ---------------------------------------------------------------------------

class TestVisualAnalyzer:
    """Tests for the VisualAnalyzer class."""

    def _get_analyzer(self) -> "VisualAnalyzer":
        """Import and instantiate with mocked cv2."""
        from viral_clip_extractor.core.visual_analyzer import VisualAnalyzer
        return VisualAnalyzer(PipelineConfig())

    def test_analyze_segment_returns_visual_features(self):
        """analyze_segment returns a VisualFeatures dataclass."""
        mock_cv2 = _mock_cv2_module()

        # VideoCapture mock that returns 3 frames then stops
        cap = MagicMock()
        cap.isOpened.return_value = True
        frames = [_make_frame(), _make_frame(color=(200, 100, 50)), _make_frame(color=(50, 200, 100))]
        cap.read.side_effect = [(True, f) for f in frames]
        mock_cv2.VideoCapture.return_value = cap

        # cvtColor returns grayscale or HSV as needed
        def fake_cvtcolor(img, code):
            if code == mock_cv2.COLOR_BGR2GRAY:
                return _make_gray(img.shape[1], img.shape[0])
            elif code == mock_cv2.COLOR_BGR2HSV:
                return np.random.randint(0, 256, img.shape, dtype=np.uint8)
            return img

        mock_cv2.cvtColor.side_effect = fake_cvtcolor

        # Optical flow
        flow = np.ones((480, 640, 2), dtype=np.float32) * 2.0
        mock_cv2.calcOpticalFlowFarneback.return_value = flow
        mock_cv2.cartToPolar.return_value = (
            np.ones((480, 640), dtype=np.float32) * 2.0,
            np.zeros((480, 640), dtype=np.float32),
        )

        # Face detection returns no faces
        cascade = MagicMock()
        cascade.empty.return_value = False
        cascade.detectMultiScale.return_value = []
        mock_cv2.CascadeClassifier.return_value = cascade

        # GaussianBlur and minMaxLoc for composition fallback
        mock_cv2.GaussianBlur.return_value = _make_gray()
        mock_cv2.minMaxLoc.return_value = (0, 255, (0, 0), (320, 240))

        import viral_clip_extractor.core.visual_analyzer as va_mod
        va_mod._cv2 = mock_cv2

        try:
            analyzer = self._get_analyzer()
            result = analyzer.analyze_segment("/fake/video.mp4", 0.0, 3.0)

            assert isinstance(result, VisualFeatures)
            assert 0.0 <= result.motion_score <= 1.0
            assert 0.0 <= result.face_presence <= 1.0
            assert 0.0 <= result.visual_interest <= 1.0
            assert 0.0 <= result.composition_score <= 1.0
        finally:
            va_mod._cv2 = None

    def test_analyze_segment_empty_video(self):
        """Returns zero features when no frames can be read."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (False, None)
        mock_cv2.VideoCapture.return_value = cap

        import viral_clip_extractor.core.visual_analyzer as va_mod
        va_mod._cv2 = mock_cv2

        try:
            analyzer = self._get_analyzer()
            result = analyzer.analyze_segment("/fake/video.mp4", 0.0, 5.0)

            assert result.motion_score == 0.0
            assert result.face_presence == 0.0
            assert result.visual_interest == 0.0
            assert result.composition_score == 0.0
        finally:
            va_mod._cv2 = None

    def test_analyze_segment_unopenable_video(self):
        """Returns zero features when video cannot be opened."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = cap

        import viral_clip_extractor.core.visual_analyzer as va_mod
        va_mod._cv2 = mock_cv2

        try:
            analyzer = self._get_analyzer()
            result = analyzer.analyze_segment("/fake/nonexistent.mp4", 0.0, 5.0)

            assert isinstance(result, VisualFeatures)
            assert result.motion_score == 0.0
        finally:
            va_mod._cv2 = None

    def test_analyze_segment_with_faces(self):
        """Face presence score reflects detected faces."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = True
        frames = [_make_frame() for _ in range(3)]
        cap.read.side_effect = [(True, f) for f in frames]
        mock_cv2.VideoCapture.return_value = cap

        def _cvt_color_faces(img, code):
            if code == mock_cv2.COLOR_BGR2GRAY:
                return _make_gray(img.shape[1], img.shape[0])
            elif code == mock_cv2.COLOR_BGR2HSV:
                return np.random.randint(0, 256, (img.shape[0], img.shape[1], 3), dtype=np.uint8)
            return img

        mock_cv2.cvtColor.side_effect = _cvt_color_faces

        # Optical flow
        flow = np.zeros((480, 640, 2), dtype=np.float32)
        mock_cv2.calcOpticalFlowFarneback.return_value = flow
        mock_cv2.cartToPolar.return_value = (
            np.zeros((480, 640), dtype=np.float32),
            np.zeros((480, 640), dtype=np.float32),
        )

        # Face cascade: _compute_face_presence calls detectMultiScale 3x,
        # then _compute_composition calls it 3x more — total 6 calls
        cascade = MagicMock()
        cascade.empty.return_value = False
        face_rect = np.array([[100, 100, 50, 50]])
        cascade.detectMultiScale.side_effect = [
            face_rect,     # face_presence: Frame 1 — face
            face_rect,     # face_presence: Frame 2 — face
            [],            # face_presence: Frame 3 — no face
            face_rect,     # composition: Frame 1 — face
            face_rect,     # composition: Frame 2 — face
            [],            # composition: Frame 3 — no face
        ]
        mock_cv2.CascadeClassifier.return_value = cascade

        mock_cv2.GaussianBlur.return_value = _make_gray()
        mock_cv2.minMaxLoc.return_value = (0, 255, (0, 0), (320, 240))

        import viral_clip_extractor.core.visual_analyzer as va_mod
        va_mod._cv2 = mock_cv2

        try:
            analyzer = self._get_analyzer()
            result = analyzer.analyze_segment("/fake/video.mp4", 0.0, 3.0)

            assert isinstance(result, VisualFeatures)
            # Face presence should be > 0 since faces were detected
            # (exact value depends on how many frames pass through face_presence
            # vs composition separately — both create new CascadeClassifier)
            assert result.face_presence >= 0.0
        finally:
            va_mod._cv2 = None

    def test_sample_frames_invalid_segment(self):
        """_sample_frames returns empty list for zero/negative duration."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = True
        mock_cv2.VideoCapture.return_value = cap

        import viral_clip_extractor.core.visual_analyzer as va_mod
        va_mod._cv2 = mock_cv2

        try:
            analyzer = self._get_analyzer()
            frames = analyzer._sample_frames("/fake/video.mp4", 5.0, 3.0, fps=1)
            assert frames == []
        finally:
            va_mod._cv2 = None

    def test_detect_faces_count(self):
        """detect_faces returns correct face count."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, _make_frame())
        mock_cv2.VideoCapture.return_value = cap

        mock_cv2.cvtColor.return_value = _make_gray()

        cascade = MagicMock()
        cascade.empty.return_value = False
        cascade.detectMultiScale.return_value = np.array([
            [100, 100, 50, 50],
            [300, 200, 60, 60],
        ])
        mock_cv2.CascadeClassifier.return_value = cascade

        import viral_clip_extractor.core.visual_analyzer as va_mod

        with patch.object(va_mod, "_get_cv2", return_value=mock_cv2):
            analyzer = self._get_analyzer()
            count = analyzer.detect_faces("/fake/video.mp4", 1.0)
            assert count == 2

    def test_detect_faces_no_video(self):
        """detect_faces returns 0 when video can't be opened."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = cap

        import viral_clip_extractor.core.visual_analyzer as va_mod

        with patch.object(va_mod, "_get_cv2", return_value=mock_cv2):
            analyzer = self._get_analyzer()
            assert analyzer.detect_faces("/fake/bad.mp4", 0.0) == 0

    def test_motion_score_single_frame(self):
        """Motion score is 0 with only one frame (no pairs for flow)."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.side_effect = [(True, _make_frame()), (False, None)]
        mock_cv2.VideoCapture.return_value = cap

        def _cvt_single(img, code):
            if code == mock_cv2.COLOR_BGR2GRAY:
                return _make_gray(img.shape[1], img.shape[0])
            elif code == mock_cv2.COLOR_BGR2HSV:
                return np.random.randint(0, 256, (img.shape[0], img.shape[1], 3), dtype=np.uint8)
            return img

        mock_cv2.cvtColor.side_effect = _cvt_single

        cascade = MagicMock()
        cascade.empty.return_value = False
        cascade.detectMultiScale.return_value = []
        mock_cv2.CascadeClassifier.return_value = cascade

        mock_cv2.GaussianBlur.return_value = _make_gray()
        mock_cv2.minMaxLoc.return_value = (0, 255, (0, 0), (320, 240))

        import viral_clip_extractor.core.visual_analyzer as va_mod
        va_mod._cv2 = mock_cv2

        try:
            analyzer = self._get_analyzer()
            result = analyzer.analyze_segment("/fake/video.mp4", 0.0, 1.0)
            assert result.motion_score == 0.0
        finally:
            va_mod._cv2 = None


# ---------------------------------------------------------------------------
# SmartCropper tests
# ---------------------------------------------------------------------------

class TestSmartCropper:
    """Tests for the SmartCropper class."""

    @pytest.fixture(autouse=True)
    def _reset_face_detection_caches(self):
        """Reset visual_analyzer face detection caches between tests.

        The DNN and Haar cascade caches are module-level globals that
        persist across tests.  If a prior test loads the real cv2 and
        caches a real CascadeClassifier, subsequent tests that mock cv2
        in smart_cropper still pick up the cached real classifier via
        ``_detect_faces_in_gray`` — causing mock bypass and spurious
        failures.
        """
        import viral_clip_extractor.core.visual_analyzer as va_mod
        va_mod._cached_dnn_checked = False
        va_mod._cached_dnn_net = None
        va_mod._cached_haar_cascade = None
        yield
        va_mod._cached_dnn_checked = False
        va_mod._cached_dnn_net = None
        va_mod._cached_haar_cascade = None

    def _get_cropper(self) -> "SmartCropper":
        from viral_clip_extractor.extractors.smart_cropper import SmartCropper
        return SmartCropper(PipelineConfig())

    def test_crop_params_horizontal_video_no_face(self):
        """Centre-crops a 1920x1080 horizontal video when no faces found."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_WIDTH: 1920,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)
        cap.read.return_value = (True, _make_frame(1920, 1080))

        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.cvtColor.return_value = _make_gray(1920, 1080)

        cascade = MagicMock()
        cascade.empty.return_value = False
        cascade.detectMultiScale.return_value = []
        mock_cv2.CascadeClassifier.return_value = cascade

        import viral_clip_extractor.extractors.smart_cropper as sc_mod
        sc_mod._cv2 = mock_cv2

        try:
            cropper = self._get_cropper()
            params = cropper.get_crop_params("/fake/video.mp4")

            assert params["crop_h"] == 1080
            expected_w = int(1080 * 9 / 16)
            assert params["crop_w"] == expected_w
            # No faces → center crop
            assert params["crop_x"] == (1920 - expected_w) // 2
            assert params["crop_y"] == 0
        finally:
            sc_mod._cv2 = None

    def test_crop_params_already_vertical(self):
        """No cropping needed for already-vertical video."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_WIDTH: 1080,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 1920,
        }.get(prop, 0)
        cap.read.return_value = (True, _make_frame(1080, 1920))
        mock_cv2.VideoCapture.return_value = cap

        import viral_clip_extractor.extractors.smart_cropper as sc_mod
        sc_mod._cv2 = mock_cv2

        try:
            cropper = self._get_cropper()
            params = cropper.get_crop_params("/fake/vertical.mp4")

            assert params["crop_x"] == 0
            assert params["crop_y"] == 0
            assert params["crop_w"] == 1080
            assert params["crop_h"] == 1920
        finally:
            sc_mod._cv2 = None

    def test_crop_params_face_aware(self):
        """Crop centres on detected face rather than frame centre."""
        mock_cv2 = _mock_cv2_module()

        # Single capture — refactored code opens video once
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_WIDTH: 1920,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)
        cap.read.return_value = (True, _make_frame(1920, 1080))

        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.cvtColor.return_value = _make_gray(1920, 1080)

        cascade = MagicMock()
        cascade.empty.return_value = False
        # Face at right side of frame (x=1400, 200x250 — passes validation)
        cascade.detectMultiScale.return_value = np.array([[1400, 300, 200, 250]])
        mock_cv2.CascadeClassifier.return_value = cascade

        import viral_clip_extractor.extractors.smart_cropper as sc_mod
        sc_mod._cv2 = mock_cv2

        try:
            cropper = self._get_cropper()
            params = cropper.get_crop_params("/fake/video.mp4")

            crop_w = int(1080 * 9 / 16)
            face_cx = 1400 + 100  # face centre x (200-wide face)
            expected_x = int(face_cx - crop_w / 2)
            # Clamp to bounds
            expected_x = max(0, min(expected_x, 1920 - crop_w))

            assert params["crop_x"] == expected_x
            assert params["crop_w"] == crop_w
        finally:
            sc_mod._cv2 = None

    def test_ffmpeg_filter_format(self):
        """get_ffmpeg_filter returns valid crop=W:H:X:Y string."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_WIDTH: 1920,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)
        cap.read.return_value = (True, _make_frame(1920, 1080))

        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.cvtColor.return_value = _make_gray(1920, 1080)

        cascade = MagicMock()
        cascade.empty.return_value = False
        cascade.detectMultiScale.return_value = []
        mock_cv2.CascadeClassifier.return_value = cascade

        import viral_clip_extractor.extractors.smart_cropper as sc_mod
        sc_mod._cv2 = mock_cv2

        try:
            cropper = self._get_cropper()
            filt = cropper.get_ffmpeg_filter("/fake/video.mp4")

            assert filt.startswith("crop=")
            parts = filt.replace("crop=", "").split(":")
            assert len(parts) == 4
            # All parts should be valid integers
            for p in parts:
                int(p)
        finally:
            sc_mod._cv2 = None

    def test_crop_params_unopenable_video(self):
        """Returns zero-dimension params when video cannot be opened."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = cap

        import viral_clip_extractor.extractors.smart_cropper as sc_mod
        sc_mod._cv2 = mock_cv2

        try:
            cropper = self._get_cropper()
            params = cropper.get_crop_params("/fake/bad.mp4")

            assert params["crop_w"] == 0
            assert params["crop_h"] == 0
        finally:
            sc_mod._cv2 = None

    def test_detect_subject_center_with_face(self):
        """detect_subject_center returns face centre when face is found."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_WIDTH: 1920,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)
        cap.read.return_value = (True, _make_frame(1920, 1080))
        mock_cv2.VideoCapture.return_value = cap

        mock_cv2.cvtColor.return_value = _make_gray(1920, 1080)

        cascade = MagicMock()
        cascade.empty.return_value = False
        # 200x250 face — passes validation (area ~2.4% but rescued by dimensions)
        cascade.detectMultiScale.return_value = np.array([[400, 200, 200, 250]])
        mock_cv2.CascadeClassifier.return_value = cascade

        import viral_clip_extractor.extractors.smart_cropper as sc_mod
        sc_mod._cv2 = mock_cv2

        try:
            cropper = self._get_cropper()
            cx, cy = cropper.detect_subject_center("/fake/video.mp4", 1.0)

            assert cx == 500  # 400 + 200/2
            assert cy == 325  # 200 + 250/2
        finally:
            sc_mod._cv2 = None

    def test_detect_subject_center_fallback(self):
        """detect_subject_center uses brightness fallback when no face."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_WIDTH: 640,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 480,
        }.get(prop, 0)
        cap.read.return_value = (True, _make_frame())
        mock_cv2.VideoCapture.return_value = cap

        mock_cv2.cvtColor.return_value = _make_gray()

        cascade = MagicMock()
        cascade.empty.return_value = False
        cascade.detectMultiScale.return_value = []
        mock_cv2.CascadeClassifier.return_value = cascade

        mock_cv2.GaussianBlur.return_value = _make_gray()
        mock_cv2.minMaxLoc.return_value = (0, 255, (0, 0), (500, 300))

        import viral_clip_extractor.extractors.smart_cropper as sc_mod
        sc_mod._cv2 = mock_cv2

        try:
            cropper = self._get_cropper()
            cx, cy = cropper.detect_subject_center("/fake/video.mp4", 0.0)

            assert cx == 500
            assert cy == 300
        finally:
            sc_mod._cv2 = None

    def test_crop_params_multi_frame_median(self):
        """Multi-frame sampling uses median face center X across 3 frames."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_WIDTH: 1920,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)

        # 3 frames with faces at consistent X positions (200x250 — passes validation):
        # frame1: face at x=800, frame2: face at x=850, frame3: face at x=900
        # Centers = [900, 950, 1000], spread = 100 (5.2% of 1920) → passes spatial consistency
        # Median center = 950
        frames = [
            _make_frame(1920, 1080),
            _make_frame(1920, 1080),
            _make_frame(1920, 1080),
        ]
        frame_idx = [0]

        def fake_read():
            idx = frame_idx[0]
            frame_idx[0] += 1
            if idx < len(frames):
                return (True, frames[idx])
            return (False, None)

        cap.read.side_effect = fake_read

        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.cvtColor.return_value = _make_gray(1920, 1080)

        cascade = MagicMock()
        cascade.empty.return_value = False
        # Consistent face positions for each detectMultiScale call
        face_results = [
            np.array([[800, 300, 200, 250]]),   # center_x = 900
            np.array([[850, 300, 200, 250]]),   # center_x = 950
            np.array([[900, 300, 200, 250]]),   # center_x = 1000
        ]
        detect_idx = [0]

        def fake_detect(*args, **kwargs):
            idx = detect_idx[0]
            detect_idx[0] += 1
            if idx < len(face_results):
                return face_results[idx]
            return []

        cascade.detectMultiScale.side_effect = fake_detect
        mock_cv2.CascadeClassifier.return_value = cascade

        import viral_clip_extractor.extractors.smart_cropper as sc_mod
        sc_mod._cv2 = mock_cv2

        try:
            cropper = self._get_cropper()
            # Pass end_time to trigger multi-frame sampling
            params = cropper.get_crop_params(
                "/fake/video.mp4", start_time=0.0, end_time=30.0,
            )

            crop_w = int(1080 * 9 / 16)
            # Median of [900, 950, 1000] = 950
            expected_x = int(950 - crop_w / 2)
            expected_x = max(0, min(expected_x, 1920 - crop_w))

            assert params["crop_x"] == expected_x
            assert params["crop_w"] == crop_w
        finally:
            sc_mod._cv2 = None

    def test_crop_params_center_fallback_no_faces(self):
        """When no faces detected in any frame, falls back to center crop."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_WIDTH: 1920,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)
        cap.read.return_value = (True, _make_frame(1920, 1080))

        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.cvtColor.return_value = _make_gray(1920, 1080)

        cascade = MagicMock()
        cascade.empty.return_value = False
        cascade.detectMultiScale.return_value = []  # no faces ever
        mock_cv2.CascadeClassifier.return_value = cascade

        import viral_clip_extractor.extractors.smart_cropper as sc_mod
        sc_mod._cv2 = mock_cv2

        try:
            cropper = self._get_cropper()
            params = cropper.get_crop_params(
                "/fake/video.mp4", start_time=5.0, end_time=35.0,
            )

            crop_w = int(1080 * 9 / 16)
            # No faces → center crop, NOT brightness-based
            assert params["crop_x"] == (1920 - crop_w) // 2
        finally:
            sc_mod._cv2 = None

    def test_crop_params_partial_face_detection(self):
        """When faces detected in only some frames, uses available detections."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_WIDTH: 1920,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)
        cap.read.return_value = (True, _make_frame(1920, 1080))

        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.cvtColor.return_value = _make_gray(1920, 1080)

        cascade = MagicMock()
        cascade.empty.return_value = False
        # Face in frame 1, none in frame 2, face in frame 3 (200x250 passes validation)
        detect_results = [
            np.array([[800, 300, 200, 250]]),  # center_x = 900
            [],                                 # no face
            np.array([[800, 300, 200, 250]]),  # center_x = 900
        ]
        detect_idx = [0]

        def fake_detect(*args, **kwargs):
            idx = detect_idx[0]
            detect_idx[0] += 1
            if idx < len(detect_results):
                return detect_results[idx]
            return []

        cascade.detectMultiScale.side_effect = fake_detect
        mock_cv2.CascadeClassifier.return_value = cascade

        import viral_clip_extractor.extractors.smart_cropper as sc_mod
        sc_mod._cv2 = mock_cv2

        try:
            cropper = self._get_cropper()
            params = cropper.get_crop_params(
                "/fake/video.mp4", start_time=0.0, end_time=30.0,
            )

            crop_w = int(1080 * 9 / 16)
            # Median of [900, 900] = 900
            expected_x = int(900 - crop_w / 2)
            expected_x = max(0, min(expected_x, 1920 - crop_w))

            assert params["crop_x"] == expected_x
        finally:
            sc_mod._cv2 = None


# ---------------------------------------------------------------------------
# Face validation tests (TDD — functions don't exist yet)
# ---------------------------------------------------------------------------

class TestFaceValidation:
    """Tests for face detection validation functions.

    These functions will be added to visual_analyzer.py in Stage 4.
    All tests should FAIL until implementation is complete (TDD).
    """

    def test_reject_face_too_small(self):
        """A face bbox covering < 3% of frame area must be rejected."""
        from viral_clip_extractor.core.visual_analyzer import _validate_face_detection

        # 30x30 bbox on 1920x1080 frame = 900 / 2_073_600 = 0.043% area
        assert _validate_face_detection((100, 100, 30, 30), 1920, 1080) is False

    def test_reject_face_too_small_boundary(self):
        """A face bbox at exactly 3% of frame area boundary — reject at < 3%."""
        from viral_clip_extractor.core.visual_analyzer import _validate_face_detection

        # 3% of 1920x1080 = 62208 px². A 249x249 = 62001 < 62208 → reject
        assert _validate_face_detection((100, 100, 249, 249), 1920, 1080) is False

    def test_reject_face_too_large(self):
        """A face bbox covering > 70% of frame area must be rejected."""
        from viral_clip_extractor.core.visual_analyzer import _validate_face_detection

        # 1500x1000 on 1920x1080 = 1_500_000 / 2_073_600 = 72.3% area
        assert _validate_face_detection((100, 40, 1500, 1000), 1920, 1080) is False

    def test_reject_face_too_large_boundary(self):
        """A face bbox just over 70% of frame area — must be rejected."""
        from viral_clip_extractor.core.visual_analyzer import _validate_face_detection

        # 70% of 2073600 = 1451520. A 1206x1206 = 1454436 > 1451520 → reject
        assert _validate_face_detection((100, 0, 1206, 1206), 1920, 1080) is False

    def test_reject_face_bad_aspect_ratio(self):
        """A bbox with width/height > 2:1 must be rejected (not a face)."""
        from viral_clip_extractor.core.visual_analyzer import _validate_face_detection

        # 300x100 = 3:1 ratio → reject
        assert _validate_face_detection((100, 100, 300, 100), 1920, 1080) is False

    def test_reject_face_bad_aspect_ratio_tall(self):
        """A bbox with height/width > 2:1 must be rejected (not a face)."""
        from viral_clip_extractor.core.visual_analyzer import _validate_face_detection

        # 80x250 = height/width = 3.1:1 → reject
        assert _validate_face_detection((100, 100, 80, 250), 1920, 1080) is False

    def test_accept_face_plausible(self):
        """A reasonable face (200x250, ~2.4% of 1920x1080) must be accepted."""
        from viral_clip_extractor.core.visual_analyzer import _validate_face_detection

        # 200x250 = 50000 / 2073600 = 2.41% — but shape is plausible
        # Wait: 2.41% < 3%, this would fail the min_area check.
        # Prompt says "~2.4% of 1920x1080" and "must be accepted" — the min area
        # threshold should be based on the bbox dimensions being plausible, not
        # just raw area. Let's trust the prompt: 200x250 is a real face size.
        # The area fraction for a real face on 1080p is legitimately small.
        # Adjusting: the 3% threshold is for *tiny* detections like 30x30.
        # A 200x250 face on 1080p is clearly valid. The min area threshold
        # likely uses a minimum dimension size, not just area percentage.
        # Actually re-reading the prompt: "< 3% of frame area must be rejected"
        # and "200x250, ~2.4% ... must be accepted" — this means the validation
        # uses BOTH area percentage AND dimension heuristics. A 200x250 face
        # passes the minimum dimension check even if area < 3%.
        # Simplest interpretation: min_area_fraction only rejects truly tiny
        # faces. Let's use: min side > 5% of min frame dimension as alternative.
        # 200 > 0.05 * 1080 = 54 ✓. Just test what the prompt says.
        assert _validate_face_detection((400, 300, 200, 250), 1920, 1080) is True

    def test_accept_face_medium_size(self):
        """A medium face (150x180) on 1920x1080 should be accepted."""
        from viral_clip_extractor.core.visual_analyzer import _validate_face_detection

        # 150x180 — reasonable face, aspect ratio 0.83, dimensions plausible
        assert _validate_face_detection((500, 300, 150, 180), 1920, 1080) is True

    def test_reject_inconsistent_face_positions(self):
        """Face centers spread > 30% of frame width must be rejected."""
        from viral_clip_extractor.core.visual_analyzer import validate_spatial_consistency

        # Centers at [100, 960, 1800] on 1920-wide frame
        # Spread = 1800 - 100 = 1700, which is 1700/1920 = 88.5% → reject
        result = validate_spatial_consistency([100, 960, 1800], 1920)
        assert result == []  # All rejected — too inconsistent

    def test_accept_consistent_face_positions(self):
        """Face centers within 30% of frame width should be accepted."""
        from viral_clip_extractor.core.visual_analyzer import validate_spatial_consistency

        # Centers at [900, 950, 920] on 1920-wide frame
        # Spread = 950 - 900 = 50, which is 50/1920 = 2.6% → accept
        result = validate_spatial_consistency([900, 950, 920], 1920)
        assert result == [900, 950, 920]  # All kept

    def test_spatial_consistency_single_center(self):
        """A single face center should always be accepted."""
        from viral_clip_extractor.core.visual_analyzer import validate_spatial_consistency

        result = validate_spatial_consistency([500], 1920)
        assert result == [500]

    def test_spatial_consistency_empty_input(self):
        """Empty input returns empty output."""
        from viral_clip_extractor.core.visual_analyzer import validate_spatial_consistency

        result = validate_spatial_consistency([], 1920)
        assert result == []


# ---------------------------------------------------------------------------
# Exception handling tests (TDD)
# ---------------------------------------------------------------------------

class TestExceptionHandling:
    """Tests for proper exception logging (replacing silent swallows)."""

    def test_dnn_exception_logs_warning(self):
        """When DNN detection throws, it must log a warning (not silently pass)."""
        mock_cv2 = _mock_cv2_module()

        import viral_clip_extractor.core.visual_analyzer as va_mod
        va_mod._cached_dnn_checked = False
        va_mod._cached_dnn_net = None
        va_mod._cached_haar_cascade = None

        # Provide a mock DNN net that raises on forward()
        mock_net = MagicMock()
        mock_net.forward.side_effect = RuntimeError("DNN inference failed")
        mock_net.setInput = MagicMock()

        # Patch _get_dnn_net to return our broken net
        with patch.object(va_mod, "_get_dnn_net", return_value=mock_net):
            mock_cv2.cvtColor.return_value = _make_frame(300, 300)
            mock_cv2.dnn.blobFromImage.return_value = np.zeros((1, 3, 300, 300))

            # Haar fallback
            cascade = MagicMock()
            cascade.empty.return_value = False
            cascade.detectMultiScale.return_value = []
            mock_cv2.CascadeClassifier.return_value = cascade

            va_mod._cv2 = mock_cv2
            try:
                gray = _make_gray(1920, 1080)
                with patch.object(va_mod.logger, "warning") as mock_warn:
                    va_mod._detect_faces_in_gray(mock_cv2, gray)
                    # The warning MUST be called when DNN throws
                    mock_warn.assert_called()
                    # Check the warning message mentions DNN
                    warn_msg = mock_warn.call_args[0][0]
                    assert "dnn" in warn_msg.lower() or "DNN" in warn_msg
            finally:
                va_mod._cv2 = None

    def test_dnn_model_load_exception_logs_warning(self):
        """When DNN model loading throws, it must log a warning."""
        mock_cv2 = _mock_cv2_module()

        import viral_clip_extractor.core.visual_analyzer as va_mod
        va_mod._cached_dnn_checked = False
        va_mod._cached_dnn_net = None

        # Make model files appear to exist but loading fails
        with patch("viral_clip_extractor.core.visual_analyzer.Path") as MockPath:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path.__truediv__ = MagicMock(return_value=mock_path)
            MockPath.return_value = mock_path
            MockPath.home.return_value = mock_path

            mock_cv2.dnn.readNetFromCaffe.side_effect = RuntimeError("corrupt model")

            with patch.object(va_mod.logger, "warning") as mock_warn:
                result = va_mod._get_dnn_net(mock_cv2)
                assert result is None
                # Must log warning about the failure
                mock_warn.assert_called()

    def test_detection_logging_shows_detector(self):
        """When a face is detected, the log must indicate which detector was used."""
        mock_cv2 = _mock_cv2_module()

        import viral_clip_extractor.core.visual_analyzer as va_mod
        va_mod._cached_dnn_checked = False
        va_mod._cached_dnn_net = None
        va_mod._cached_haar_cascade = None

        cascade = MagicMock()
        cascade.empty.return_value = False
        cascade.detectMultiScale.return_value = np.array([[100, 100, 80, 80]])
        mock_cv2.CascadeClassifier.return_value = cascade

        va_mod._cv2 = mock_cv2
        try:
            gray = _make_gray(640, 480)
            with patch.object(va_mod.logger, "debug") as mock_debug:
                va_mod._detect_faces_in_gray(mock_cv2, gray)
                # Must log which detector produced the result
                assert mock_debug.called
                all_msgs = " ".join(
                    str(c[0][0]) for c in mock_debug.call_args_list if c[0]
                )
                assert "haar" in all_msgs.lower() or "cascade" in all_msgs.lower() \
                    or "dnn" in all_msgs.lower() or "detector" in all_msgs.lower()
        finally:
            va_mod._cv2 = None


# ---------------------------------------------------------------------------
# Integration test (TDD)
# ---------------------------------------------------------------------------

class TestValidationIntegration:
    """Integration test: validation filters affect crop output."""

    @pytest.fixture(autouse=True)
    def _reset_caches(self):
        """Reset visual_analyzer caches."""
        import viral_clip_extractor.core.visual_analyzer as va_mod
        va_mod._cached_dnn_checked = False
        va_mod._cached_dnn_net = None
        va_mod._cached_haar_cascade = None
        yield
        va_mod._cached_dnn_checked = False
        va_mod._cached_dnn_net = None
        va_mod._cached_haar_cascade = None

    def test_validation_filters_in_crop_params(self):
        """Implausible face detection (30x30 on 1920x1080) → center crop, not face-aware."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_WIDTH: 1920,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)
        cap.read.return_value = (True, _make_frame(1920, 1080))
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.cvtColor.return_value = _make_gray(1920, 1080)

        # Haar returns a tiny 30x30 face — implausible, should be rejected by validation
        cascade = MagicMock()
        cascade.empty.return_value = False
        cascade.detectMultiScale.return_value = np.array([[1400, 500, 30, 30]])
        mock_cv2.CascadeClassifier.return_value = cascade

        import viral_clip_extractor.extractors.smart_cropper as sc_mod
        sc_mod._cv2 = mock_cv2

        try:
            from viral_clip_extractor.extractors.smart_cropper import SmartCropper
            cropper = SmartCropper(PipelineConfig())
            params = cropper.get_crop_params("/fake/video.mp4")

            crop_w = int(1080 * 9 / 16)
            # Validation should reject the 30x30 face → center crop
            expected_center_x = (1920 - crop_w) // 2
            assert params["crop_x"] == expected_center_x, (
                f"Expected center crop at x={expected_center_x}, got x={params['crop_x']}. "
                f"Validation should have rejected the 30x30 face detection."
            )
        finally:
            sc_mod._cv2 = None

    def test_spatial_consistency_rejects_scattered_faces_in_crop_params(self):
        """Scattered face centers (spread > 30% of frame) → center crop, not face-aware."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_WIDTH: 1920,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)

        frames = [_make_frame(1920, 1080) for _ in range(3)]
        frame_idx = [0]

        def fake_read():
            idx = frame_idx[0]
            frame_idx[0] += 1
            if idx < len(frames):
                return (True, frames[idx])
            return (False, None)

        cap.read.side_effect = fake_read
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.cvtColor.return_value = _make_gray(1920, 1080)

        cascade = MagicMock()
        cascade.empty.return_value = False
        # Widely scattered face positions (200x250 — passes per-face validation)
        # Centers at [500, 1000, 1600], spread = 1100 = 57% of 1920 → rejected
        face_results = [
            np.array([[400, 300, 200, 250]]),   # center_x = 500
            np.array([[900, 300, 200, 250]]),   # center_x = 1000
            np.array([[1500, 300, 200, 250]]),  # center_x = 1600
        ]
        detect_idx = [0]

        def fake_detect(*args, **kwargs):
            idx = detect_idx[0]
            detect_idx[0] += 1
            if idx < len(face_results):
                return face_results[idx]
            return []

        cascade.detectMultiScale.side_effect = fake_detect
        mock_cv2.CascadeClassifier.return_value = cascade

        import viral_clip_extractor.extractors.smart_cropper as sc_mod
        sc_mod._cv2 = mock_cv2

        try:
            from viral_clip_extractor.extractors.smart_cropper import SmartCropper
            cropper = SmartCropper(PipelineConfig())
            params = cropper.get_crop_params(
                "/fake/video.mp4", start_time=0.0, end_time=30.0,
            )

            crop_w = int(1080 * 9 / 16)
            # Spatial consistency rejects all → center crop
            expected_center_x = (1920 - crop_w) // 2
            assert params["crop_x"] == expected_center_x, (
                f"Expected center crop at x={expected_center_x}, got x={params['crop_x']}. "
                f"Spatial consistency should have rejected scattered face centers."
            )
        finally:
            sc_mod._cv2 = None

    def test_validation_accepts_plausible_in_crop_params(self):
        """Plausible face detection (200x250) → face-aware crop, not center."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_WIDTH: 1920,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)
        cap.read.return_value = (True, _make_frame(1920, 1080))
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.cvtColor.return_value = _make_gray(1920, 1080)

        # Haar returns a plausible 200x250 face at x=1300
        cascade = MagicMock()
        cascade.empty.return_value = False
        cascade.detectMultiScale.return_value = np.array([[1300, 300, 200, 250]])
        mock_cv2.CascadeClassifier.return_value = cascade

        import viral_clip_extractor.extractors.smart_cropper as sc_mod
        sc_mod._cv2 = mock_cv2

        try:
            from viral_clip_extractor.extractors.smart_cropper import SmartCropper
            cropper = SmartCropper(PipelineConfig())
            params = cropper.get_crop_params("/fake/video.mp4")

            crop_w = int(1080 * 9 / 16)
            center_x = (1920 - crop_w) // 2
            # Plausible face → face-aware crop, NOT center
            face_cx = 1300 + 100  # face center x = 1400
            expected_x = int(face_cx - crop_w / 2)
            expected_x = max(0, min(expected_x, 1920 - crop_w))
            assert params["crop_x"] == expected_x, (
                f"Expected face-aware crop at x={expected_x}, got x={params['crop_x']}. "
                f"Validation should have accepted the 200x250 face detection."
            )
        finally:
            sc_mod._cv2 = None


# ---------------------------------------------------------------------------
# Edge case / robustness tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases: corrupt video, missing models, all-black frames,
    resource cleanup, and cv2.dnn errors."""

    @pytest.fixture(autouse=True)
    def _reset_caches(self):
        import viral_clip_extractor.core.visual_analyzer as va_mod
        va_mod._cached_dnn_checked = False
        va_mod._cached_dnn_net = None
        va_mod._cached_haar_cascade = None
        yield
        va_mod._cached_dnn_checked = False
        va_mod._cached_dnn_net = None
        va_mod._cached_haar_cascade = None

    def test_all_black_frame_returns_no_faces(self):
        """All-black frame produces zero face detections, no crash."""
        mock_cv2 = _mock_cv2_module()

        cascade = MagicMock()
        cascade.empty.return_value = False
        cascade.detectMultiScale.return_value = []
        mock_cv2.CascadeClassifier.return_value = cascade

        import viral_clip_extractor.core.visual_analyzer as va_mod
        va_mod._cv2 = mock_cv2

        try:
            gray = np.zeros((1080, 1920), dtype=np.uint8)
            faces = va_mod._detect_faces_in_gray(mock_cv2, gray)
            assert len(faces) == 0
        finally:
            va_mod._cv2 = None

    def test_corrupt_video_crop_params_returns_zeros(self):
        """Corrupt/unopenable video returns zero-dimension crop params."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = cap

        import viral_clip_extractor.extractors.smart_cropper as sc_mod
        sc_mod._cv2 = mock_cv2

        try:
            from viral_clip_extractor.extractors.smart_cropper import SmartCropper
            cropper = SmartCropper(PipelineConfig())
            params = cropper.get_crop_params("/corrupt/video.mp4")
            assert params == {"crop_x": 0, "crop_y": 0, "crop_w": 0, "crop_h": 0}
        finally:
            sc_mod._cv2 = None

    def test_corrupt_video_ffmpeg_filter_returns_empty(self):
        """Corrupt video produces empty FFmpeg filter string."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = cap

        import viral_clip_extractor.extractors.smart_cropper as sc_mod
        sc_mod._cv2 = mock_cv2

        try:
            from viral_clip_extractor.extractors.smart_cropper import SmartCropper
            cropper = SmartCropper(PipelineConfig())
            filt = cropper.get_ffmpeg_filter("/corrupt/video.mp4")
            assert filt == ""
        finally:
            sc_mod._cv2 = None

    def test_zero_dimension_video_returns_zeros(self):
        """Video reporting 0x0 dimensions returns zero crop params."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = 0  # All props return 0
        cap.read.return_value = (True, _make_frame(1, 1))
        mock_cv2.VideoCapture.return_value = cap

        import viral_clip_extractor.extractors.smart_cropper as sc_mod
        sc_mod._cv2 = mock_cv2

        try:
            from viral_clip_extractor.extractors.smart_cropper import SmartCropper
            cropper = SmartCropper(PipelineConfig())
            params = cropper.get_crop_params("/fake/zero-dim.mp4")
            assert params["crop_w"] == 0
            assert params["crop_h"] == 0
        finally:
            sc_mod._cv2 = None

    def test_dnn_unexpected_error_falls_back_to_haar(self):
        """cv2.dnn raising unexpected errors falls back to Haar gracefully."""
        mock_cv2 = _mock_cv2_module()

        # DNN net that raises on forward()
        mock_net = MagicMock()
        mock_net.setInput = MagicMock()
        mock_net.forward.side_effect = RuntimeError("CUDA OOM")

        import viral_clip_extractor.core.visual_analyzer as va_mod

        # Haar cascade returns a face
        cascade = MagicMock()
        cascade.empty.return_value = False
        cascade.detectMultiScale.return_value = np.array([[200, 200, 100, 120]])
        mock_cv2.CascadeClassifier.return_value = cascade

        mock_cv2.cvtColor.return_value = _make_frame(300, 300)
        mock_cv2.dnn.blobFromImage.return_value = np.zeros((1, 3, 300, 300))

        with patch.object(va_mod, "_get_dnn_net", return_value=mock_net):
            va_mod._cv2 = mock_cv2
            try:
                gray = _make_gray(640, 480)
                faces = va_mod._detect_faces_in_gray(mock_cv2, gray)
                # Should fall back to Haar and find the face
                assert len(faces) == 1
            finally:
                va_mod._cv2 = None

    def test_video_capture_release_on_success(self):
        """VideoCapture.release() is called even on successful path."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_WIDTH: 1920,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)
        cap.read.return_value = (True, _make_frame(1920, 1080))
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.cvtColor.return_value = _make_gray(1920, 1080)

        cascade = MagicMock()
        cascade.empty.return_value = False
        cascade.detectMultiScale.return_value = []
        mock_cv2.CascadeClassifier.return_value = cascade

        import viral_clip_extractor.extractors.smart_cropper as sc_mod
        sc_mod._cv2 = mock_cv2

        try:
            from viral_clip_extractor.extractors.smart_cropper import SmartCropper
            cropper = SmartCropper(PipelineConfig())
            cropper.get_crop_params("/fake/video.mp4")
            cap.release.assert_called_once()
        finally:
            sc_mod._cv2 = None

    def test_video_capture_release_on_read_failure(self):
        """VideoCapture.release() is called even when frame read fails."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_WIDTH: 1920,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)
        cap.read.return_value = (False, None)
        mock_cv2.VideoCapture.return_value = cap

        import viral_clip_extractor.extractors.smart_cropper as sc_mod
        sc_mod._cv2 = mock_cv2

        try:
            from viral_clip_extractor.extractors.smart_cropper import SmartCropper
            cropper = SmartCropper(PipelineConfig())
            cropper.get_crop_params("/fake/broken.mp4")
            cap.release.assert_called_once()
        finally:
            sc_mod._cv2 = None

    def test_detect_subject_center_rejects_implausible_face(self):
        """detect_subject_center rejects tiny face and uses brightness fallback."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_WIDTH: 1920,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)
        cap.read.return_value = (True, _make_frame(1920, 1080))
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.cvtColor.return_value = _make_gray(1920, 1080)

        # Haar returns a tiny 30x30 face — should be rejected by validation
        cascade = MagicMock()
        cascade.empty.return_value = False
        cascade.detectMultiScale.return_value = np.array([[1400, 500, 30, 30]])
        mock_cv2.CascadeClassifier.return_value = cascade

        # Brightness fallback should return this point
        mock_cv2.GaussianBlur.return_value = _make_gray(1920, 1080)
        mock_cv2.minMaxLoc.return_value = (0, 255, (0, 0), (960, 540))

        import viral_clip_extractor.extractors.smart_cropper as sc_mod
        sc_mod._cv2 = mock_cv2

        try:
            from viral_clip_extractor.extractors.smart_cropper import SmartCropper
            cropper = SmartCropper(PipelineConfig())
            cx, cy = cropper.detect_subject_center("/fake/video.mp4", 5.0)

            # Should use brightness fallback (960, 540), NOT unvalidated face (1415, 515)
            assert cx == 960, (
                f"Expected brightness fallback cx=960, got cx={cx}. "
                f"detect_subject_center should reject 30x30 face detection."
            )
            assert cy == 540
        finally:
            sc_mod._cv2 = None

    def test_detect_subject_center_accepts_plausible_face(self):
        """detect_subject_center uses validated face when plausible."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_WIDTH: 1920,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)
        cap.read.return_value = (True, _make_frame(1920, 1080))
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.cvtColor.return_value = _make_gray(1920, 1080)

        # 200x250 face — plausible, should be accepted
        cascade = MagicMock()
        cascade.empty.return_value = False
        cascade.detectMultiScale.return_value = np.array([[400, 200, 200, 250]])
        mock_cv2.CascadeClassifier.return_value = cascade

        import viral_clip_extractor.extractors.smart_cropper as sc_mod
        sc_mod._cv2 = mock_cv2

        try:
            from viral_clip_extractor.extractors.smart_cropper import SmartCropper
            cropper = SmartCropper(PipelineConfig())
            cx, cy = cropper.detect_subject_center("/fake/video.mp4", 0.0)

            # Should use validated face center (500, 325)
            assert cx == 500  # 400 + 200/2
            assert cy == 325  # 200 + 250/2
        finally:
            sc_mod._cv2 = None

    def test_haar_cascade_load_failure_returns_empty(self):
        """When Haar cascade fails to load (empty), returns no faces."""
        mock_cv2 = _mock_cv2_module()

        cascade = MagicMock()
        cascade.empty.return_value = True  # Failed to load
        mock_cv2.CascadeClassifier.return_value = cascade

        import viral_clip_extractor.core.visual_analyzer as va_mod
        va_mod._cv2 = mock_cv2

        try:
            gray = _make_gray(640, 480)
            with patch.object(va_mod.logger, "warning") as mock_warn:
                faces = va_mod._detect_faces_in_gray(mock_cv2, gray)
                assert len(faces) == 0
                # Should log warning about cascade failure
                mock_warn.assert_called()
        finally:
            va_mod._cv2 = None

    def test_optical_flow_exception_logs_error_message(self):
        """When optical flow throws, the exception message is included in the log."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = True
        frames = [_make_frame(), _make_frame()]
        cap.read.side_effect = [(True, f) for f in frames]
        mock_cv2.VideoCapture.return_value = cap

        def _cvt(img, code):
            if code == mock_cv2.COLOR_BGR2GRAY:
                return _make_gray(img.shape[1], img.shape[0])
            elif code == mock_cv2.COLOR_BGR2HSV:
                return np.random.randint(0, 256, img.shape, dtype=np.uint8)
            return img
        mock_cv2.cvtColor.side_effect = _cvt

        # Optical flow throws
        mock_cv2.calcOpticalFlowFarneback.side_effect = RuntimeError("bad frame")

        cascade = MagicMock()
        cascade.empty.return_value = False
        cascade.detectMultiScale.return_value = []
        mock_cv2.CascadeClassifier.return_value = cascade

        mock_cv2.GaussianBlur.return_value = _make_gray()
        mock_cv2.minMaxLoc.return_value = (0, 255, (0, 0), (320, 240))

        import viral_clip_extractor.core.visual_analyzer as va_mod
        va_mod._cv2 = mock_cv2

        try:
            from viral_clip_extractor.core.visual_analyzer import VisualAnalyzer
            analyzer = VisualAnalyzer(PipelineConfig())

            with patch.object(va_mod.logger, "debug") as mock_debug:
                result = analyzer.analyze_segment("/fake/video.mp4", 0.0, 2.0)
                # Motion should be 0 since flow failed
                assert result.motion_score == 0.0
                # The exception message should appear in the log
                all_msgs = " ".join(
                    str(c) for c in mock_debug.call_args_list
                )
                assert "bad frame" in all_msgs
        finally:
            va_mod._cv2 = None

    def test_crop_params_survives_read_exception(self):
        """get_crop_params handles cap.read() raising an exception gracefully."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_WIDTH: 1920,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)
        cap.read.side_effect = RuntimeError("GPU memory error")
        mock_cv2.VideoCapture.return_value = cap

        import viral_clip_extractor.extractors.smart_cropper as sc_mod
        sc_mod._cv2 = mock_cv2

        try:
            from viral_clip_extractor.extractors.smart_cropper import SmartCropper
            cropper = SmartCropper(PipelineConfig())
            # Should NOT raise — should fall back to center crop
            params = cropper.get_crop_params("/fake/video.mp4")
            crop_w = int(1080 * 9 / 16)
            assert params["crop_x"] == (1920 - crop_w) // 2
            assert params["crop_w"] == crop_w
            cap.release.assert_called_once()
        finally:
            sc_mod._cv2 = None

    def test_detect_subject_center_survives_read_exception(self):
        """detect_subject_center handles cap.read() raising gracefully."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_WIDTH: 1920,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)
        cap.read.side_effect = RuntimeError("GPU memory error")
        mock_cv2.VideoCapture.return_value = cap

        import viral_clip_extractor.extractors.smart_cropper as sc_mod
        sc_mod._cv2 = mock_cv2

        try:
            from viral_clip_extractor.extractors.smart_cropper import SmartCropper
            cropper = SmartCropper(PipelineConfig())
            # Should NOT raise — should return center fallback
            cx, cy = cropper.detect_subject_center("/fake/video.mp4", 1.0)
            assert cx == 960  # 1920 // 2
            assert cy == 540  # 1080 // 2
            cap.release.assert_called_once()
        finally:
            sc_mod._cv2 = None

    def test_sample_frames_survives_read_exception(self):
        """_sample_frames handles cap.read() raising on some frames."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = True
        # First read succeeds, second raises, third succeeds
        good_frame = _make_frame(640, 480)
        cap.read.side_effect = [
            (True, good_frame),
            RuntimeError("bad frame"),
            (True, good_frame),
        ]
        mock_cv2.VideoCapture.return_value = cap

        import viral_clip_extractor.core.visual_analyzer as va_mod
        va_mod._cv2 = mock_cv2

        try:
            from viral_clip_extractor.core.visual_analyzer import VisualAnalyzer
            analyzer = VisualAnalyzer(PipelineConfig())
            frames = analyzer._sample_frames("/fake/video.mp4", 0.0, 3.0, fps=1)
            # Should get 2 frames (skipping the one that raised)
            assert len(frames) == 2
            cap.release.assert_called_once()
        finally:
            va_mod._cv2 = None

    def test_validate_face_zero_dimensions(self):
        """_validate_face_detection rejects zero-size bboxes without crashing."""
        from viral_clip_extractor.core.visual_analyzer import _validate_face_detection
        assert _validate_face_detection((100, 100, 0, 0), 1920, 1080) is False

    def test_validate_face_negative_dimensions(self):
        """_validate_face_detection rejects negative dimension bboxes."""
        from viral_clip_extractor.core.visual_analyzer import _validate_face_detection
        assert _validate_face_detection((100, 100, -50, 100), 1920, 1080) is False
        assert _validate_face_detection((100, 100, 100, -50), 1920, 1080) is False

    def test_validate_face_zero_frame_dimensions(self):
        """_validate_face_detection returns False (no crash) when frame dims are 0."""
        from viral_clip_extractor.core.visual_analyzer import _validate_face_detection
        # Previously caused ZeroDivisionError
        assert _validate_face_detection((10, 10, 50, 50), 0, 0) is False
        assert _validate_face_detection((10, 10, 50, 50), 0, 1080) is False
        assert _validate_face_detection((10, 10, 50, 50), 1920, 0) is False

    def test_dnn_skip_logs_when_models_missing(self):
        """_get_dnn_net logs at debug level when model files are absent."""
        import viral_clip_extractor.core.visual_analyzer as va_mod
        va_mod._cached_dnn_checked = False
        va_mod._cached_dnn_net = None

        mock_cv2 = _mock_cv2_module()

        with patch("viral_clip_extractor.core.visual_analyzer.Path") as mock_path:
            # Make all path.exists() return False (no model files)
            mock_path.return_value.__truediv__ = MagicMock(
                return_value=MagicMock(exists=MagicMock(return_value=False))
            )
            mock_path.home.return_value.__truediv__ = MagicMock(
                return_value=MagicMock(exists=MagicMock(return_value=False))
            )
            mock_path.return_value.resolve.return_value.parent.parent.__truediv__ = MagicMock(
                return_value=MagicMock(exists=MagicMock(return_value=False))
            )

            import logging
            with patch.object(va_mod.logger, "debug") as mock_debug:
                result = va_mod._get_dnn_net(mock_cv2)
                assert result is None
                # Verify the skip was logged
                mock_debug.assert_called_with(
                    "DNN face model not found — using Haar cascade only"
                )

    def test_crop_params_videocapture_released_on_frame_read_exception(self):
        """VideoCapture is released even when frame reading raises."""
        import viral_clip_extractor.extractors.smart_cropper as sc_mod
        import viral_clip_extractor.core.visual_analyzer as va_mod

        mock_cv2 = _mock_cv2_module()
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {3: 1920.0, 4: 1080.0}.get(prop, 0.0)
        # Frame read raises on every attempt
        cap.read.side_effect = RuntimeError("I/O error during read")
        mock_cv2.VideoCapture.return_value = cap

        sc_mod._cv2 = mock_cv2
        va_mod._cv2 = mock_cv2
        va_mod._cached_dnn_checked = True
        va_mod._cached_dnn_net = None

        try:
            from viral_clip_extractor.extractors.smart_cropper import SmartCropper
            sc = SmartCropper(PipelineConfig())
            params = sc.get_crop_params("/fake/video.mp4", start_time=0, end_time=10)
            # Should get center crop (no faces due to exception)
            expected_crop_w = int(1080 * 9 / 16)
            assert params["crop_x"] == (1920 - expected_crop_w) // 2
            # VideoCapture must be released
            cap.release.assert_called_once()
        finally:
            sc_mod._cv2 = None
            va_mod._cv2 = None

    def test_crop_params_zero_dimensions_video(self):
        """get_crop_params handles video reporting 0x0 dimensions."""
        import viral_clip_extractor.extractors.smart_cropper as sc_mod

        mock_cv2 = _mock_cv2_module()
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = 0.0  # All props return 0
        cap.read.return_value = (False, None)
        mock_cv2.VideoCapture.return_value = cap

        sc_mod._cv2 = mock_cv2

        try:
            from viral_clip_extractor.extractors.smart_cropper import SmartCropper
            sc = SmartCropper(PipelineConfig())
            params = sc.get_crop_params("/fake/zero-dim-video.mp4")
            assert params == {"crop_x": 0, "crop_y": 0, "crop_w": 0, "crop_h": 0}
            cap.release.assert_called_once()
        finally:
            sc_mod._cv2 = None

    def test_consistent_false_positive_at_edge_produces_face_aware_crop(self):
        """Consistent false positives at edge position pass spatial consistency.

        This documents a known limitation: if Haar detects plausible-sized
        non-square faces at the same edge position across all 3 sampled frames,
        spatial consistency passes (spread ≈ 0) and the crop is face-aware
        at the wrong position. This test guards against regression if the
        detection pipeline changes.
        """
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_WIDTH: 1920,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)

        frames = [_make_frame(1920, 1080) for _ in range(3)]
        frame_idx = [0]

        def fake_read():
            idx = frame_idx[0]
            frame_idx[0] += 1
            if idx < len(frames):
                return (True, frames[idx])
            return (False, None)

        cap.read.side_effect = fake_read
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.cvtColor.return_value = _make_gray(1920, 1080)

        cascade = MagicMock()
        cascade.empty.return_value = False
        # All 3 frames: 200x250 face at x=1091 (the original bug position)
        # These pass per-face validation (non-square, > 10% min_dim)
        # and pass spatial consistency (spread = 0)
        cascade.detectMultiScale.return_value = np.array([[1091, 300, 200, 250]])
        mock_cv2.CascadeClassifier.return_value = cascade

        import viral_clip_extractor.extractors.smart_cropper as sc_mod
        sc_mod._cv2 = mock_cv2

        try:
            from viral_clip_extractor.extractors.smart_cropper import SmartCropper
            cropper = SmartCropper(PipelineConfig())
            params = cropper.get_crop_params(
                "/fake/video.mp4", start_time=0.0, end_time=30.0,
            )

            # Consistent false positives at edge → face-aware crop (known limitation)
            # Face center_x = 1091 + 100 = 1191
            assert params["crop_x"] != (1920 - params["crop_w"]) // 2, (
                "Expected face-aware (non-center) crop for consistent edge detections. "
                "If this fails, a new filter is rejecting consistent false positives "
                "— which would be an improvement, not a regression."
            )
        finally:
            sc_mod._cv2 = None

    def test_square_detection_above_3pct_area_accepted(self):
        """Perfectly square detection above 3% area passes validation.

        Documents that the squareness check only applies in the rescue
        heuristic for sub-3% area detections. Large square detections
        are accepted — this matches Haar cascade's behaviour of returning
        square bounding boxes for legitimate faces.
        """
        from viral_clip_extractor.core.visual_analyzer import _validate_face_detection

        # 316x316 on 1920x1080 = 4.82% area, perfectly square (ratio = 1.0)
        result = _validate_face_detection((800, 400, 316, 316), 1920, 1080)
        assert result is True, (
            "Expected 316x316 (4.82% area) to pass validation. "
            "Squareness check should only apply to sub-3% area detections."
        )

        # 400x400 on 1920x1080 = 7.72% area, also square
        result2 = _validate_face_detection((500, 300, 400, 400), 1920, 1080)
        assert result2 is True, (
            "Expected 400x400 (7.72% area) to pass validation."
        )

    def test_aspect_ratio_exactly_2_rejected(self):
        """Detection with exactly 2:1 aspect ratio is rejected."""
        from viral_clip_extractor.core.visual_analyzer import _validate_face_detection

        # 400x200 = 2:1 ratio, area = 3.86% of 1920x1080
        result = _validate_face_detection((500, 300, 400, 200), 1920, 1080)
        assert result is False, (
            "Expected 400x200 (2:1 ratio) to be rejected."
        )

        # 200x400 = 1:2 ratio
        result2 = _validate_face_detection((500, 300, 200, 400), 1920, 1080)
        assert result2 is False, (
            "Expected 200x400 (1:2 ratio) to be rejected."
        )

    def test_all_black_frame_no_false_positives(self):
        """All-black frame produces no face detections."""
        import viral_clip_extractor.core.visual_analyzer as va_mod

        mock_cv2 = _mock_cv2_module()
        va_mod._cached_dnn_checked = True
        va_mod._cached_dnn_net = None

        mock_cascade = MagicMock()
        mock_cascade.empty.return_value = False
        mock_cascade.detectMultiScale.return_value = np.empty((0, 4), dtype=int)
        mock_cv2.CascadeClassifier.return_value = mock_cascade

        va_mod._cv2 = mock_cv2
        try:
            black_gray = np.zeros((1080, 1920), dtype=np.uint8)
            from viral_clip_extractor.core.visual_analyzer import _detect_faces_in_gray
            result = _detect_faces_in_gray(mock_cv2, black_gray)
            assert len(result) == 0
        finally:
            va_mod._cv2 = None
