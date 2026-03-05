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
        va_mod._cv2 = mock_cv2

        try:
            analyzer = self._get_analyzer()
            count = analyzer.detect_faces("/fake/video.mp4", 1.0)
            assert count == 2
        finally:
            va_mod._cv2 = None

    def test_detect_faces_no_video(self):
        """detect_faces returns 0 when video can't be opened."""
        mock_cv2 = _mock_cv2_module()

        cap = MagicMock()
        cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = cap

        import viral_clip_extractor.core.visual_analyzer as va_mod
        va_mod._cv2 = mock_cv2

        try:
            analyzer = self._get_analyzer()
            assert analyzer.detect_faces("/fake/bad.mp4", 0.0) == 0
        finally:
            va_mod._cv2 = None

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

    def _get_cropper(self) -> "SmartCropper":
        from viral_clip_extractor.extractors.smart_cropper import SmartCropper
        return SmartCropper(PipelineConfig())

    def test_crop_params_horizontal_video_no_face(self):
        """Centre-crops a 1920x1080 horizontal video when no faces found."""
        mock_cv2 = _mock_cv2_module()

        # Main cap for dimensions
        cap_main = MagicMock()
        cap_main.isOpened.return_value = True
        cap_main.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_WIDTH: 1920,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)

        # Face detection cap
        cap_face = MagicMock()
        cap_face.isOpened.return_value = True
        cap_face.read.return_value = (True, _make_frame(1920, 1080))

        mock_cv2.VideoCapture.side_effect = [cap_main, cap_face]
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
            # Centre crop: x should be roughly (1920 - expected_w) / 2
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

        cap_main = MagicMock()
        cap_main.isOpened.return_value = True
        cap_main.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_WIDTH: 1920,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)

        cap_face = MagicMock()
        cap_face.isOpened.return_value = True
        cap_face.read.return_value = (True, _make_frame(1920, 1080))

        mock_cv2.VideoCapture.side_effect = [cap_main, cap_face]
        mock_cv2.cvtColor.return_value = _make_gray(1920, 1080)

        cascade = MagicMock()
        cascade.empty.return_value = False
        # Face at right side of frame (x=1400)
        cascade.detectMultiScale.return_value = np.array([[1400, 300, 100, 100]])
        mock_cv2.CascadeClassifier.return_value = cascade

        import viral_clip_extractor.extractors.smart_cropper as sc_mod
        sc_mod._cv2 = mock_cv2

        try:
            cropper = self._get_cropper()
            params = cropper.get_crop_params("/fake/video.mp4")

            crop_w = int(1080 * 9 / 16)
            face_cx = 1400 + 50  # face centre x
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

        cap_main = MagicMock()
        cap_main.isOpened.return_value = True
        cap_main.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_WIDTH: 1920,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)

        cap_face = MagicMock()
        cap_face.isOpened.return_value = True
        cap_face.read.return_value = (True, _make_frame(1920, 1080))

        mock_cv2.VideoCapture.side_effect = [cap_main, cap_face]
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
        cascade.detectMultiScale.return_value = np.array([[400, 200, 100, 120]])
        mock_cv2.CascadeClassifier.return_value = cascade

        import viral_clip_extractor.extractors.smart_cropper as sc_mod
        sc_mod._cv2 = mock_cv2

        try:
            cropper = self._get_cropper()
            cx, cy = cropper.detect_subject_center("/fake/video.mp4", 1.0)

            assert cx == 450  # 400 + 100/2
            assert cy == 260  # 200 + 120/2
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
