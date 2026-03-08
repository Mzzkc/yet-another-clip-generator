"""Tests for content-profile-aware prompt rendering across all 3 prompt sites.

Verifies that segmentation, virality, and caption prompts reflect content type,
audience, tone, custom instructions, and transcript text when provided.
"""

from unittest.mock import MagicMock, patch

import pytest

from yacg.core.semantic_analyzer import SemanticAnalyzer
from yacg.transcript_segmenter import TranscriptSegmenter


# ---------------------------------------------------------------------------
# Helpers — construct components with content profile fields
# ---------------------------------------------------------------------------


def _make_segmenter(**kwargs) -> TranscriptSegmenter:
    """Create a TranscriptSegmenter with content profile params.

    Avoids loading the Whisper model (not needed for prompt tests).
    """
    defaults = dict(
        whisper_model="small",
        ollama_host="http://localhost:11434",
        model_name="qwen2.5:7b",
    )
    defaults.update(kwargs)
    seg = TranscriptSegmenter.__new__(TranscriptSegmenter)
    seg.whisper_model = defaults["whisper_model"]
    seg.ollama_host = defaults["ollama_host"]
    seg.model_name = defaults["model_name"]
    seg.whisper_device = "auto"
    seg.whisper_compute_type = "auto"
    seg.pause_threshold = 0.3
    seg.min_segment_duration = 15.0
    seg.max_segment_duration = 45.0
    seg.vad_filter = True
    seg.content_type = defaults.get("content_type", "")
    seg.channel_description = defaults.get("channel_description", "")
    seg.target_audience = defaults.get("target_audience", "")
    seg.custom_instructions = defaults.get("custom_instructions", "")
    seg._session = MagicMock()
    seg._cached_model = None
    seg._cached_model_key = None
    return seg


def _make_semantic_analyzer(**kwargs) -> SemanticAnalyzer:
    """Create a SemanticAnalyzer with content profile params."""
    return SemanticAnalyzer(
        model=kwargs.get("model", "qwen2.5-vl:7b"),
        ollama_host=kwargs.get("ollama_host", "http://localhost:11434"),
        num_frames=kwargs.get("num_frames", 1),
        content_type=kwargs.get("content_type", "general"),
        channel_description=kwargs.get("channel_description", ""),
        target_audience=kwargs.get("target_audience", ""),
        tone=kwargs.get("tone", ""),
        custom_instructions=kwargs.get("custom_instructions", ""),
    )


# ---------------------------------------------------------------------------
# Segmentation prompt tests
# ---------------------------------------------------------------------------


class TestSegmentationPrompt:
    """TranscriptSegmenter._create_segmentation_prompt content awareness."""

    def test_gaming_prompt_has_gaming_guidance(self):
        seg = _make_segmenter(content_type="gaming")
        prompt = seg._create_segmentation_prompt("Hello world", "Test", 10)
        assert "HIGHLIGHT" in prompt or "highlight" in prompt.lower()

    def test_cooking_prompt_has_cooking_guidance(self):
        seg = _make_segmenter(content_type="cooking")
        prompt = seg._create_segmentation_prompt("Hello world", "Test", 10)
        # Cooking guidance mentions techniques or satisfying
        assert "cooking" in prompt.lower() or "technique" in prompt.lower() or "satisfying" in prompt.lower()

    def test_general_prompt_has_general_guidance(self):
        seg = _make_segmenter(content_type="general")
        prompt = seg._create_segmentation_prompt("Hello world", "Test", 10)
        # General guidance has hooks and complete thoughts
        assert "hook" in prompt.lower() or "HOOK" in prompt

    def test_gaming_vs_cooking_different(self):
        gaming_seg = _make_segmenter(content_type="gaming")
        cooking_seg = _make_segmenter(content_type="cooking")
        gaming_prompt = gaming_seg._create_segmentation_prompt("Hello", "T", 10)
        cooking_prompt = cooking_seg._create_segmentation_prompt("Hello", "T", 10)
        assert gaming_prompt != cooking_prompt

    def test_target_audience_injected(self):
        seg = _make_segmenter(
            content_type="general",
            target_audience="beginner Python developers",
        )
        prompt = seg._create_segmentation_prompt("Hello world", "Test", 10)
        assert "beginner Python developers" in prompt

    def test_custom_instructions_injected(self):
        seg = _make_segmenter(
            content_type="general",
            custom_instructions="Focus on funny moments only",
        )
        prompt = seg._create_segmentation_prompt("Hello world", "Test", 10)
        assert "Focus on funny moments only" in prompt

    def test_prompt_requests_json_output(self):
        seg = _make_segmenter(content_type="general")
        prompt = seg._create_segmentation_prompt("Hello world", "Test", 10)
        assert "JSON" in prompt

    def test_empty_audience_not_injected(self):
        seg = _make_segmenter(content_type="general", target_audience="")
        prompt = seg._create_segmentation_prompt("Hello world", "Test", 10)
        assert "TARGET AUDIENCE" not in prompt

    def test_empty_instructions_not_injected(self):
        seg = _make_segmenter(content_type="general", custom_instructions="")
        prompt = seg._create_segmentation_prompt("Hello world", "Test", 10)
        assert "ADDITIONAL INSTRUCTIONS" not in prompt

    def test_educational_prompt_has_educational_guidance(self):
        seg = _make_segmenter(content_type="educational")
        prompt = seg._create_segmentation_prompt("Hello world", "Test", 10)
        assert "insight" in prompt.lower() or "aha" in prompt.lower() or "explanation" in prompt.lower()

    def test_fitness_prompt_has_fitness_guidance(self):
        seg = _make_segmenter(content_type="fitness")
        prompt = seg._create_segmentation_prompt("Hello world", "Test", 10)
        assert "workout" in prompt.lower() or "exercise" in prompt.lower() or "fitness" in prompt.lower()

    def test_comedy_prompt_has_comedy_guidance(self):
        seg = _make_segmenter(content_type="comedy")
        prompt = seg._create_segmentation_prompt("Hello world", "Test", 10)
        assert "punch" in prompt.lower() or "laugh" in prompt.lower() or "comedy" in prompt.lower()

    def test_asmr_prompt_has_asmr_guidance(self):
        seg = _make_segmenter(content_type="asmr")
        prompt = seg._create_segmentation_prompt("Hello world", "Test", 10)
        assert "asmr" in prompt.lower() or "tingle" in prompt.lower() or "trigger" in prompt.lower()

    def test_music_prompt_has_music_guidance(self):
        seg = _make_segmenter(content_type="music")
        prompt = seg._create_segmentation_prompt("Hello world", "Test", 10)
        assert "music" in prompt.lower() or "drop" in prompt.lower() or "chorus" in prompt.lower()

    def test_all_specialized_types_differ_from_general(self):
        """All content types with specialized guidance produce different prompts from general."""
        general_seg = _make_segmenter(content_type="general")
        general_prompt = general_seg._create_segmentation_prompt("Hello", "T", 10)
        specialized_types = ["gaming", "cooking", "asmr", "educational", "fitness", "comedy", "music"]
        for ct in specialized_types:
            seg = _make_segmenter(content_type=ct)
            prompt = seg._create_segmentation_prompt("Hello", "T", 10)
            assert prompt != general_prompt, f"{ct} prompt should differ from general"

    def test_channel_description_injected_in_segmentation(self):
        seg = _make_segmenter(
            content_type="general",
            channel_description="A daily vlog about urban gardening",
        )
        prompt = seg._create_segmentation_prompt("Hello world", "Test", 10)
        assert "urban gardening" in prompt


# ---------------------------------------------------------------------------
# Semantic / virality prompt tests
# ---------------------------------------------------------------------------


class TestViralityPrompt:
    """SemanticAnalyzer._create_virality_prompt content awareness."""

    def test_gaming_prompt_differs_from_general(self):
        gaming = _make_semantic_analyzer(content_type="gaming")
        general = _make_semantic_analyzer(content_type="general")
        gaming_prompt = gaming._create_virality_prompt(15.0, "Test")
        general_prompt = general._create_virality_prompt(15.0, "Test")
        assert gaming_prompt != general_prompt

    def test_asmr_prompt_mentions_asmr(self):
        analyzer = _make_semantic_analyzer(content_type="asmr")
        prompt = analyzer._create_virality_prompt(15.0, "Test")
        # ASMR content should have ASMR-specific guidance
        assert "asmr" in prompt.lower() or "tingle" in prompt.lower() or "relaxation" in prompt.lower()

    def test_channel_description_injected(self):
        analyzer = _make_semantic_analyzer(
            content_type="general",
            channel_description="A daily vlog about urban gardening",
        )
        prompt = analyzer._create_virality_prompt(15.0, "Test")
        assert "urban gardening" in prompt

    def test_target_audience_injected(self):
        analyzer = _make_semantic_analyzer(
            content_type="general",
            target_audience="senior citizens",
        )
        prompt = analyzer._create_virality_prompt(15.0, "Test")
        assert "senior citizens" in prompt

    def test_tone_injected(self):
        analyzer = _make_semantic_analyzer(
            content_type="general",
            tone="professional",
        )
        prompt = analyzer._create_virality_prompt(15.0, "Test")
        assert "professional" in prompt

    def test_custom_instructions_injected(self):
        analyzer = _make_semantic_analyzer(
            content_type="general",
            custom_instructions="Rate humor above all else",
        )
        prompt = analyzer._create_virality_prompt(15.0, "Test")
        assert "Rate humor above all else" in prompt

    def test_prompt_requests_json_output(self):
        analyzer = _make_semantic_analyzer(content_type="general")
        prompt = analyzer._create_virality_prompt(15.0, "Test")
        assert "JSON" in prompt

    def test_prompt_includes_duration(self):
        analyzer = _make_semantic_analyzer(content_type="general")
        prompt = analyzer._create_virality_prompt(28.5, "Test")
        assert "28.5" in prompt

    def test_prompt_includes_title(self):
        analyzer = _make_semantic_analyzer(content_type="general")
        prompt = analyzer._create_virality_prompt(15.0, "Dragon ASMR")
        assert "Dragon ASMR" in prompt

    def test_empty_channel_not_injected(self):
        analyzer = _make_semantic_analyzer(content_type="general", channel_description="")
        prompt = analyzer._create_virality_prompt(15.0, "Test")
        assert "Channel context" not in prompt

    def test_empty_audience_not_injected(self):
        analyzer = _make_semantic_analyzer(content_type="general", target_audience="")
        prompt = analyzer._create_virality_prompt(15.0, "Test")
        assert "Target audience" not in prompt


# ---------------------------------------------------------------------------
# Caption prompt tests
# ---------------------------------------------------------------------------


class TestCaptionPrompt:
    """OllamaVideoAnalyzer._create_caption_prompt content awareness."""

    def test_asmr_vs_general_different(self):
        from yacg.caption_generator import OllamaVideoAnalyzer

        asmr = OllamaVideoAnalyzer(content_type="asmr")
        general = OllamaVideoAnalyzer(content_type="general")
        asmr_prompt = asmr._create_caption_prompt("Test")
        general_prompt = general._create_caption_prompt("Test")
        assert asmr_prompt != general_prompt

    def test_asmr_prompt_mentions_asmr(self):
        from yacg.caption_generator import OllamaVideoAnalyzer

        analyzer = OllamaVideoAnalyzer(content_type="asmr")
        prompt = analyzer._create_caption_prompt("Test")
        assert "ASMR" in prompt or "asmr" in prompt.lower()

    def test_general_prompt_mentions_strategist(self):
        from yacg.caption_generator import OllamaVideoAnalyzer

        analyzer = OllamaVideoAnalyzer(content_type="general")
        prompt = analyzer._create_caption_prompt("Test")
        assert "strategist" in prompt.lower() or "content" in prompt.lower()

    def test_prompt_requests_json_output(self):
        from yacg.caption_generator import OllamaVideoAnalyzer

        analyzer = OllamaVideoAnalyzer(content_type="general")
        prompt = analyzer._create_caption_prompt("Test")
        assert "JSON" in prompt

    def test_prompt_includes_title(self):
        from yacg.caption_generator import OllamaVideoAnalyzer

        analyzer = OllamaVideoAnalyzer(content_type="general")
        prompt = analyzer._create_caption_prompt("My Amazing Video")
        assert "My Amazing Video" in prompt

    def test_prompt_requests_hook(self):
        from yacg.caption_generator import OllamaVideoAnalyzer

        analyzer = OllamaVideoAnalyzer(content_type="general")
        prompt = analyzer._create_caption_prompt("Test")
        assert "HOOK" in prompt or "hook" in prompt

    def test_transcript_text_injected(self):
        from yacg.caption_generator import OllamaVideoAnalyzer

        analyzer = OllamaVideoAnalyzer(content_type="general")
        prompt = analyzer._create_caption_prompt("Test", transcript_text="Hello world this is a test")
        assert "Hello world this is a test" in prompt
        assert "TRANSCRIPT" in prompt

    def test_transcript_text_absent_when_empty(self):
        from yacg.caption_generator import OllamaVideoAnalyzer

        analyzer = OllamaVideoAnalyzer(content_type="general")
        prompt = analyzer._create_caption_prompt("Test", transcript_text="")
        assert "TRANSCRIPT" not in prompt

    def test_tone_injected_in_caption(self):
        from yacg.caption_generator import OllamaVideoAnalyzer

        analyzer = OllamaVideoAnalyzer(content_type="general", tone="energetic")
        prompt = analyzer._create_caption_prompt("Test")
        assert "energetic" in prompt

    def test_platform_tiktok_injected(self):
        from yacg.caption_generator import OllamaVideoAnalyzer

        analyzer = OllamaVideoAnalyzer(content_type="general", platform="tiktok")
        prompt = analyzer._create_caption_prompt("Test")
        assert "TikTok" in prompt

    def test_platform_reels_injected(self):
        from yacg.caption_generator import OllamaVideoAnalyzer

        analyzer = OllamaVideoAnalyzer(content_type="cooking", platform="reels")
        prompt = analyzer._create_caption_prompt("Test")
        assert "Reels" in prompt

    def test_platform_shorts_injected(self):
        from yacg.caption_generator import OllamaVideoAnalyzer

        analyzer = OllamaVideoAnalyzer(content_type="educational", platform="shorts")
        prompt = analyzer._create_caption_prompt("Test")
        assert "Shorts" in prompt

    def test_caption_length_affects_prompt(self):
        from yacg.caption_generator import OllamaVideoAnalyzer

        short_analyzer = OllamaVideoAnalyzer(content_type="general", caption_length="short")
        long_analyzer = OllamaVideoAnalyzer(content_type="general", caption_length="long")
        short_prompt = short_analyzer._create_caption_prompt("Test")
        long_prompt = long_analyzer._create_caption_prompt("Test")
        assert "(short)" in short_prompt
        assert "(long)" in long_prompt

    def test_channel_description_injected_in_caption(self):
        from yacg.caption_generator import OllamaVideoAnalyzer

        analyzer = OllamaVideoAnalyzer(
            content_type="general",
            channel_description="A daily vlog about urban gardening",
        )
        prompt = analyzer._create_caption_prompt("Test")
        assert "urban gardening" in prompt

    def test_empty_channel_not_injected_in_caption(self):
        from yacg.caption_generator import OllamaVideoAnalyzer

        analyzer = OllamaVideoAnalyzer(content_type="general", channel_description="")
        prompt = analyzer._create_caption_prompt("Test")
        assert "Channel:" not in prompt


# ---------------------------------------------------------------------------
# Cross-cutting: all prompts request valid JSON
# ---------------------------------------------------------------------------


class TestAllPromptsRequestJSON:
    """Every prompt must instruct the LLM to output valid JSON."""

    def test_segmentation_json(self):
        seg = _make_segmenter(content_type="general")
        prompt = seg._create_segmentation_prompt("transcript text", "Title", 10)
        assert "JSON" in prompt

    def test_virality_json(self):
        analyzer = _make_semantic_analyzer(content_type="general")
        prompt = analyzer._create_virality_prompt(15.0, "Title")
        assert "JSON" in prompt

    def test_caption_json(self):
        from yacg.caption_generator import OllamaVideoAnalyzer
        analyzer = OllamaVideoAnalyzer(content_type="general")
        prompt = analyzer._create_caption_prompt("Title")
        assert "JSON" in prompt
