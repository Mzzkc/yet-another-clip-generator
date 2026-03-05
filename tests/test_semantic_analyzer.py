"""Tests for the SemanticAnalyzer module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from viral_clip_extractor.core.semantic_analyzer import SemanticAnalyzer, _DEFAULT_SCORE
from viral_clip_extractor.models import SemanticFeatures


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def analyzer() -> SemanticAnalyzer:
    return SemanticAnalyzer(model="qwen2.5-vl:7b", ollama_host="http://localhost:11434")


@pytest.fixture
def valid_llm_json() -> str:
    """A well-formed JSON response like the LLM would produce."""
    return json.dumps({
        "emotional_intensity": 7.5,
        "narrative_interest": 6.0,
        "hook_potential": 8.0,
        "asmr_quality": 9.0,
        "visual_appeal": 7.0,
        "uniqueness": 5.5,
        "brief_description": "A calming tapping sequence on glass.",
    })


@pytest.fixture
def ollama_tags_response() -> dict:
    """Mock response for /api/tags with the model present."""
    return {
        "models": [
            {"name": "qwen2.5-vl:7b", "size": 4_000_000_000},
            {"name": "llama3:8b", "size": 5_000_000_000},
        ]
    }


# ---------------------------------------------------------------------------
# check_availability
# ---------------------------------------------------------------------------


class TestCheckAvailability:
    def test_returns_true_when_model_present(self, analyzer: SemanticAnalyzer, ollama_tags_response: dict) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = ollama_tags_response

        with patch("viral_clip_extractor.core.semantic_analyzer.requests.get", return_value=mock_resp):
            assert analyzer.check_availability() is True

    def test_returns_false_when_model_missing(self, analyzer: SemanticAnalyzer) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "other:latest"}]}

        with patch("viral_clip_extractor.core.semantic_analyzer.requests.get", return_value=mock_resp):
            assert analyzer.check_availability() is False

    def test_returns_false_on_connection_error(self, analyzer: SemanticAnalyzer) -> None:
        import requests as req
        with patch(
            "viral_clip_extractor.core.semantic_analyzer.requests.get",
            side_effect=req.ConnectionError("refused"),
        ):
            assert analyzer.check_availability() is False

    def test_returns_false_on_timeout(self, analyzer: SemanticAnalyzer) -> None:
        import requests as req
        with patch(
            "viral_clip_extractor.core.semantic_analyzer.requests.get",
            side_effect=req.Timeout("timed out"),
        ):
            assert analyzer.check_availability() is False

    def test_returns_false_on_non_200_status(self, analyzer: SemanticAnalyzer) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch("viral_clip_extractor.core.semantic_analyzer.requests.get", return_value=mock_resp):
            assert analyzer.check_availability() is False


# ---------------------------------------------------------------------------
# _parse_llm_response
# ---------------------------------------------------------------------------


class TestParseLlmResponse:
    def test_valid_json(self, analyzer: SemanticAnalyzer, valid_llm_json: str) -> None:
        result = analyzer._parse_llm_response(valid_llm_json)
        assert result is not None
        assert isinstance(result, SemanticFeatures)
        assert result.emotional_intensity == 7.5
        assert result.hook_potential == 8.0
        assert result.asmr_quality == 9.0
        assert result.description == "A calming tapping sequence on glass."

    def test_json_embedded_in_text(self, analyzer: SemanticAnalyzer, valid_llm_json: str) -> None:
        response = f"Here is the analysis:\n{valid_llm_json}\nDone."
        result = analyzer._parse_llm_response(response)
        assert result is not None
        assert result.narrative_interest == 6.0

    def test_missing_fields_returns_none(self, analyzer: SemanticAnalyzer) -> None:
        incomplete = json.dumps({"emotional_intensity": 7.0})
        result = analyzer._parse_llm_response(incomplete)
        assert result is None

    def test_no_json_returns_none(self, analyzer: SemanticAnalyzer) -> None:
        result = analyzer._parse_llm_response("I cannot analyze this video.")
        assert result is None

    def test_malformed_json_returns_none(self, analyzer: SemanticAnalyzer) -> None:
        result = analyzer._parse_llm_response("{bad json content")
        assert result is None

    def test_values_clamped_to_range(self, analyzer: SemanticAnalyzer) -> None:
        data = json.dumps({
            "emotional_intensity": 15.0,
            "narrative_interest": -3.0,
            "hook_potential": 8.0,
            "asmr_quality": 9.0,
            "visual_appeal": 7.0,
            "uniqueness": 5.5,
        })
        result = analyzer._parse_llm_response(data)
        assert result is not None
        assert result.emotional_intensity == 10.0
        assert result.narrative_interest == 0.0


# ---------------------------------------------------------------------------
# _create_virality_prompt
# ---------------------------------------------------------------------------


class TestCreateViralityPrompt:
    def test_includes_duration(self, analyzer: SemanticAnalyzer) -> None:
        prompt = analyzer._create_virality_prompt(15.0, "Test Video")
        assert "15.0s" in prompt

    def test_includes_title(self, analyzer: SemanticAnalyzer) -> None:
        prompt = analyzer._create_virality_prompt(10.0, "Dragon ASMR")
        assert "Dragon ASMR" in prompt

    def test_empty_title(self, analyzer: SemanticAnalyzer) -> None:
        prompt = analyzer._create_virality_prompt(10.0, "")
        assert "EMOTIONAL_INTENSITY" in prompt
        assert '"' not in prompt.split("\n")[0] or "from" not in prompt.split("\n")[0]


# ---------------------------------------------------------------------------
# analyze_segment (integration with mocks)
# ---------------------------------------------------------------------------


class TestAnalyzeSegment:
    @patch("viral_clip_extractor.core.semantic_analyzer.extract_segment")
    @patch("viral_clip_extractor.core.semantic_analyzer.requests.post")
    def test_successful_analysis(
        self,
        mock_post: MagicMock,
        mock_extract: MagicMock,
        analyzer: SemanticAnalyzer,
        valid_llm_json: str,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        # Create a fake segment file that _encode_video can read
        fake_segment = tmp_path / "seg.mp4"
        fake_segment.write_bytes(b"fake video data")

        # Make extract_segment write to the temp path (we patch mkstemp to use it)
        def fake_extract(vp: str, start: float, end: float, out: str) -> str:
            with open(out, "wb") as f:
                f.write(b"fake video data")
            return out

        mock_extract.side_effect = fake_extract

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": valid_llm_json}
        mock_post.return_value = mock_resp

        result = analyzer.analyze_segment("/fake/video.mp4", 0.0, 15.0, "Test")

        assert isinstance(result, SemanticFeatures)
        assert result.emotional_intensity == 7.5
        assert result.hook_potential == 8.0
        mock_extract.assert_called_once()
        mock_post.assert_called_once()

    @patch("viral_clip_extractor.core.semantic_analyzer.extract_segment")
    def test_extract_failure_returns_defaults(
        self, mock_extract: MagicMock, analyzer: SemanticAnalyzer
    ) -> None:
        mock_extract.side_effect = RuntimeError("ffmpeg not found")

        result = analyzer.analyze_segment("/fake/video.mp4", 0.0, 10.0)
        assert isinstance(result, SemanticFeatures)
        assert result.emotional_intensity == _DEFAULT_SCORE
        assert result.description == "Analysis unavailable — using default scores"

    def test_invalid_duration_returns_defaults(self, analyzer: SemanticAnalyzer) -> None:
        result = analyzer.analyze_segment("/fake/video.mp4", 10.0, 5.0)
        assert isinstance(result, SemanticFeatures)
        assert result.emotional_intensity == _DEFAULT_SCORE

    @patch("viral_clip_extractor.core.semantic_analyzer.time.sleep")
    @patch("viral_clip_extractor.core.semantic_analyzer.extract_segment")
    @patch("viral_clip_extractor.core.semantic_analyzer.requests.post")
    def test_retries_on_failure_then_returns_defaults(
        self,
        mock_post: MagicMock,
        mock_extract: MagicMock,
        mock_sleep: MagicMock,
        analyzer: SemanticAnalyzer,
    ) -> None:
        def fake_extract(vp: str, start: float, end: float, out: str) -> str:
            with open(out, "wb") as f:
                f.write(b"fake")
            return out

        mock_extract.side_effect = fake_extract

        # All attempts return unparseable response
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "not json at all"}
        mock_post.return_value = mock_resp

        result = analyzer.analyze_segment("/fake/video.mp4", 0.0, 10.0)
        assert isinstance(result, SemanticFeatures)
        assert result.emotional_intensity == _DEFAULT_SCORE
        assert mock_post.call_count == 3  # _MAX_RETRIES

    @patch("viral_clip_extractor.core.semantic_analyzer.time.sleep")
    @patch("viral_clip_extractor.core.semantic_analyzer.extract_segment")
    @patch("viral_clip_extractor.core.semantic_analyzer.requests.post")
    def test_connection_error_retries(
        self,
        mock_post: MagicMock,
        mock_extract: MagicMock,
        mock_sleep: MagicMock,
        analyzer: SemanticAnalyzer,
    ) -> None:
        import requests as req

        def fake_extract(vp: str, start: float, end: float, out: str) -> str:
            with open(out, "wb") as f:
                f.write(b"fake")
            return out

        mock_extract.side_effect = fake_extract
        mock_post.side_effect = req.ConnectionError("refused")

        result = analyzer.analyze_segment("/fake/video.mp4", 0.0, 10.0)
        assert isinstance(result, SemanticFeatures)
        assert result.emotional_intensity == _DEFAULT_SCORE
        assert mock_post.call_count == 3


# ---------------------------------------------------------------------------
# _default_features
# ---------------------------------------------------------------------------


class TestDefaultFeatures:
    def test_all_defaults_at_midpoint(self, analyzer: SemanticAnalyzer) -> None:
        features = analyzer._default_features()
        assert features.emotional_intensity == _DEFAULT_SCORE
        assert features.narrative_interest == _DEFAULT_SCORE
        assert features.hook_potential == _DEFAULT_SCORE
        assert features.asmr_quality == _DEFAULT_SCORE
        assert features.visual_appeal == _DEFAULT_SCORE
        assert features.uniqueness == _DEFAULT_SCORE

    def test_default_has_description(self, analyzer: SemanticAnalyzer) -> None:
        features = analyzer._default_features()
        assert "default" in features.description.lower() or "unavailable" in features.description.lower()


# ---------------------------------------------------------------------------
# _clamp
# ---------------------------------------------------------------------------


class TestClamp:
    def test_within_range(self) -> None:
        assert SemanticAnalyzer._clamp(5.0) == 5.0

    def test_above_max(self) -> None:
        assert SemanticAnalyzer._clamp(15.0) == 10.0

    def test_below_min(self) -> None:
        assert SemanticAnalyzer._clamp(-3.0) == 0.0

    def test_at_boundaries(self) -> None:
        assert SemanticAnalyzer._clamp(0.0) == 0.0
        assert SemanticAnalyzer._clamp(10.0) == 10.0
