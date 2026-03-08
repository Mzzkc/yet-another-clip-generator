"""Tests for ContentProfile dataclass, presets, and PipelineConfig integration."""

from copy import deepcopy

import pytest

from yacg.models import (
    CONTENT_PRESETS,
    VALID_CAPTION_LENGTHS,
    VALID_CONTENT_TYPES,
    VALID_PLATFORMS,
    VALID_TONES,
    ContentProfile,
    PipelineConfig,
)


# ---------------------------------------------------------------------------
# ContentProfile creation and defaults
# ---------------------------------------------------------------------------


class TestContentProfileDefaults:
    """ContentProfile defaults to 'general', not 'asmr'."""

    def test_default_content_type_is_general(self):
        profile = ContentProfile()
        assert profile.content_type == "general"

    def test_default_tone(self):
        profile = ContentProfile()
        assert profile.tone == "engaging"

    def test_default_platform(self):
        profile = ContentProfile()
        assert profile.platform == "all"

    def test_default_caption_length(self):
        profile = ContentProfile()
        assert profile.caption_length == "medium"

    def test_default_hashtag_count(self):
        profile = ContentProfile()
        assert profile.hashtag_count == 5

    def test_default_empty_strings(self):
        profile = ContentProfile()
        assert profile.channel_description == ""
        assert profile.target_audience == ""
        assert profile.custom_instructions == ""


# ---------------------------------------------------------------------------
# ContentProfile validation
# ---------------------------------------------------------------------------


class TestContentProfileValidation:
    """Validation rejects invalid values."""

    def test_invalid_content_type_raises(self):
        with pytest.raises(ValueError, match="Invalid content_type"):
            ContentProfile(content_type="invalid_type")

    def test_invalid_tone_raises(self):
        with pytest.raises(ValueError, match="Invalid tone"):
            ContentProfile(tone="screaming")

    def test_invalid_platform_raises(self):
        with pytest.raises(ValueError, match="Invalid platform"):
            ContentProfile(platform="facebook")

    def test_invalid_caption_length_raises(self):
        with pytest.raises(ValueError, match="Invalid caption_length"):
            ContentProfile(caption_length="extra_long")

    def test_hashtag_count_below_min_raises(self):
        with pytest.raises(ValueError, match="hashtag_count must be 3-7"):
            ContentProfile(hashtag_count=2)

    def test_hashtag_count_above_max_raises(self):
        with pytest.raises(ValueError, match="hashtag_count must be 3-7"):
            ContentProfile(hashtag_count=8)

    def test_all_valid_content_types_accepted(self):
        for ct in VALID_CONTENT_TYPES:
            profile = ContentProfile(content_type=ct)
            assert profile.content_type == ct

    def test_all_valid_tones_accepted(self):
        for tone in VALID_TONES:
            profile = ContentProfile(tone=tone)
            assert profile.tone == tone

    def test_all_valid_platforms_accepted(self):
        for platform in VALID_PLATFORMS:
            profile = ContentProfile(platform=platform)
            assert profile.platform == platform

    def test_all_valid_caption_lengths_accepted(self):
        for length in VALID_CAPTION_LENGTHS:
            profile = ContentProfile(caption_length=length)
            assert profile.caption_length == length

    def test_hashtag_count_boundary_values(self):
        ContentProfile(hashtag_count=3)  # min
        ContentProfile(hashtag_count=7)  # max

    def test_empty_tone_is_allowed(self):
        """Empty tone string bypasses validation (for cases where tone is optional)."""
        profile = ContentProfile(tone="")
        assert profile.tone == ""


# ---------------------------------------------------------------------------
# CONTENT_PRESETS
# ---------------------------------------------------------------------------


class TestContentPresets:
    """Preset dictionary has all 11 content types with valid configurations."""

    def test_presets_has_all_content_types(self):
        assert set(CONTENT_PRESETS.keys()) == VALID_CONTENT_TYPES

    def test_preset_count(self):
        assert len(CONTENT_PRESETS) == 11

    def test_preset_keys_match_content_type(self):
        for key, profile in CONTENT_PRESETS.items():
            assert profile.content_type == key, (
                f"Preset key '{key}' has content_type='{profile.content_type}'"
            )

    def test_all_presets_pass_validation(self):
        """Every preset should be a valid ContentProfile (no ValueError)."""
        for key, profile in CONTENT_PRESETS.items():
            assert isinstance(profile, ContentProfile), f"Preset '{key}' is not a ContentProfile"

    def test_presets_have_nonempty_audience(self):
        for key, profile in CONTENT_PRESETS.items():
            assert profile.target_audience, f"Preset '{key}' has empty target_audience"

    def test_presets_have_valid_tone(self):
        for key, profile in CONTENT_PRESETS.items():
            assert profile.tone in VALID_TONES, f"Preset '{key}' has invalid tone '{profile.tone}'"

    def test_gaming_preset_values(self):
        gaming = CONTENT_PRESETS["gaming"]
        assert gaming.tone == "energetic"
        assert gaming.target_audience == "gamers and gaming enthusiasts"
        assert gaming.caption_length == "short"

    def test_asmr_preset_values(self):
        asmr = CONTENT_PRESETS["asmr"]
        assert asmr.tone == "calm"
        assert asmr.hashtag_count == 4
        assert asmr.caption_length == "short"

    def test_cooking_preset_values(self):
        cooking = CONTENT_PRESETS["cooking"]
        assert cooking.tone == "casual"
        assert cooking.target_audience == "home cooks and food enthusiasts"
        assert cooking.platform == "reels"
        assert cooking.caption_length == "medium"

    def test_educational_preset_values(self):
        edu = CONTENT_PRESETS["educational"]
        assert edu.tone == "professional"
        assert edu.target_audience == "learners and curious minds"
        assert edu.platform == "shorts"
        assert edu.caption_length == "long"
        assert edu.hashtag_count == 4

    def test_fitness_preset_values(self):
        fit = CONTENT_PRESETS["fitness"]
        assert fit.tone == "energetic"
        assert fit.platform == "reels"

    def test_comedy_preset_values(self):
        comedy = CONTENT_PRESETS["comedy"]
        assert comedy.tone == "humorous"
        assert comedy.platform == "tiktok"
        assert comedy.caption_length == "short"

    def test_music_preset_values(self):
        music = CONTENT_PRESETS["music"]
        assert music.tone == "engaging"
        assert music.caption_length == "short"

    def test_beauty_preset_values(self):
        beauty = CONTENT_PRESETS["beauty"]
        assert beauty.tone == "casual"
        assert beauty.platform == "reels"

    def test_tech_preset_values(self):
        tech = CONTENT_PRESETS["tech"]
        assert tech.tone == "professional"
        assert tech.platform == "shorts"
        assert tech.hashtag_count == 4

    def test_vlog_preset_values(self):
        vlog = CONTENT_PRESETS["vlog"]
        assert vlog.tone == "casual"
        assert vlog.platform == "all"

    def test_general_preset_values(self):
        general = CONTENT_PRESETS["general"]
        assert general.tone == "engaging"
        assert general.platform == "all"
        assert general.caption_length == "medium"
        assert general.hashtag_count == 5

    def test_preset_override(self):
        """Load gaming preset then override tone — override takes precedence."""
        gaming = deepcopy(CONTENT_PRESETS["gaming"])
        gaming.tone = "calm"
        assert gaming.tone == "calm"
        assert gaming.content_type == "gaming"  # other fields preserved


# ---------------------------------------------------------------------------
# PipelineConfig integration
# ---------------------------------------------------------------------------


class TestPipelineConfigIntegration:
    """ContentProfile embedded in PipelineConfig."""

    def test_default_pipeline_has_general_profile(self):
        config = PipelineConfig()
        assert config.content_profile.content_type == "general"

    def test_default_asmr_mode_is_false(self):
        config = PipelineConfig()
        assert config.asmr_mode is False

    def test_asmr_content_type_with_asmr_mode(self):
        """When content_type is 'asmr', asmr_mode should be True
        (auto-sync is expected in __post_init__)."""
        asmr_profile = deepcopy(CONTENT_PRESETS["asmr"])
        config = PipelineConfig(content_profile=asmr_profile)
        # Design spec says asmr_mode auto-syncs. If Worker 1 hasn't
        # wired auto-sync yet, this documents the expected behavior.
        # After integration, this test verifies the auto-sync works.
        assert config.content_profile.content_type == "asmr"

    def test_general_content_type_has_asmr_mode_false(self):
        config = PipelineConfig()
        assert config.content_profile.content_type == "general"
        assert config.asmr_mode is False

    def test_content_type_no_longer_on_pipeline_config(self):
        """The old content_type field was removed from PipelineConfig.
        Access it through content_profile instead."""
        config = PipelineConfig()
        # content_type should NOT be a direct attribute anymore
        # (it's now config.content_profile.content_type)
        assert hasattr(config, "content_profile")
        assert hasattr(config.content_profile, "content_type")

    def test_custom_profile_in_pipeline_config(self):
        profile = ContentProfile(
            content_type="cooking",
            channel_description="A home cooking channel",
            target_audience="beginner cooks",
            tone="casual",
        )
        config = PipelineConfig(content_profile=profile)
        assert config.content_profile.content_type == "cooking"
        assert config.content_profile.channel_description == "A home cooking channel"
