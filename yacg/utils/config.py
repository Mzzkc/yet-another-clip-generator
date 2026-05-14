"""
Configuration loader for Yet Another Clip Generator (YACG).

Reads pipeline settings from an INI file (compatible with the existing
config.ini format) and falls back to PipelineConfig defaults for any
missing values.
"""

import configparser
import json
import logging
from pathlib import Path
from typing import Optional

from yacg.models import (
    CONTENT_PRESETS,
    ContentProfile,
    PipelineConfig,
    SubtitleStyle,
)

logger = logging.getLogger(__name__)

# Single source of truth — derive defaults from PipelineConfig
_DEFAULT_WEIGHTS: dict[str, float] = dict(PipelineConfig().scoring_weights)


def load_config(path: Optional[str] = None) -> PipelineConfig:
    """Load pipeline configuration from an INI file.

    If *path* is ``None`` or the file does not exist, returns a
    ``PipelineConfig`` with all defaults.  Missing keys in the file are
    silently filled with defaults.

    Args:
        path: Path to an INI-style config file.

    Returns:
        A fully-populated ``PipelineConfig`` instance.
    """
    defaults = PipelineConfig()

    if path is None:
        logger.info("No config path provided — using defaults")
        return defaults

    config_path = Path(path)
    if not config_path.exists():
        logger.warning("Config file %s not found — using defaults", path)
        return defaults

    logger.info("Loading config from %s", path)
    parser = configparser.ConfigParser()
    parser.read(config_path, encoding="utf-8")

    def _get(section: str, key: str, fallback: str) -> str:
        return parser.get(section, key, fallback=fallback)

    def _getfloat(section: str, key: str, fallback: float) -> float:
        return parser.getfloat(section, key, fallback=fallback)

    def _getint(section: str, key: str, fallback: int) -> int:
        return parser.getint(section, key, fallback=fallback)

    def _getbool(section: str, key: str, fallback: bool) -> bool:
        return parser.getboolean(section, key, fallback=fallback)

    # Parse scoring weights — accept JSON dict or fall back to defaults
    raw_weights = _get("Scoring", "weights", fallback="")
    if raw_weights:
        try:
            scoring_weights = json.loads(raw_weights)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in [Scoring] weights — using defaults")
            scoring_weights = dict(_DEFAULT_WEIGHTS)
    else:
        scoring_weights = dict(_DEFAULT_WEIGHTS)

    # Parse SubtitleStyle from [Subtitle] section
    default_style = SubtitleStyle()
    subtitle_style = SubtitleStyle(
        font_name=_get("Subtitle", "font_name", default_style.font_name),
        font_size_pct=_getfloat("Subtitle", "font_size_pct", default_style.font_size_pct),
        primary_color=_get("Subtitle", "primary_color", default_style.primary_color),
        outline_color=_get("Subtitle", "outline_color", default_style.outline_color),
        outline_width=_getfloat("Subtitle", "outline_width", default_style.outline_width),
        shadow=_getfloat("Subtitle", "shadow", default_style.shadow),
        margin_v_pct=_getfloat("Subtitle", "margin_v_pct", default_style.margin_v_pct),
        margin_h_pct=_getfloat("Subtitle", "margin_h_pct", default_style.margin_h_pct),
    )

    # Parse ContentProfile — prefer [ContentProfile] section, fall back to
    # legacy [ASMR Optimization] for backward compatibility.
    default_profile = defaults.content_profile
    if parser.has_section("ContentProfile"):
        cp_section = "ContentProfile"
    elif parser.has_section("ASMR Optimization"):
        cp_section = "ASMR Optimization"
        logger.info("Using legacy [ASMR Optimization] section — consider renaming to [ContentProfile]")
    else:
        cp_section = None

    if cp_section is not None:
        # Read content_type first — if it maps to a preset, use preset as base
        ini_content_type = _get(cp_section, "content_type", default_profile.content_type)
        from copy import deepcopy
        if ini_content_type in CONTENT_PRESETS:
            content_profile = deepcopy(CONTENT_PRESETS[ini_content_type])
        else:
            try:
                content_profile = ContentProfile(content_type=ini_content_type)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid content_type '{ini_content_type}' in config file "
                    f"[{cp_section}] section. {exc}"
                ) from exc

        # Individual field overrides from INI take precedence over preset
        if parser.has_option(cp_section, "channel_description"):
            content_profile.channel_description = _get(cp_section, "channel_description", "")
        if parser.has_option(cp_section, "target_audience"):
            content_profile.target_audience = _get(cp_section, "target_audience", content_profile.target_audience)
        if parser.has_option(cp_section, "tone"):
            content_profile.tone = _get(cp_section, "tone", content_profile.tone)
        if parser.has_option(cp_section, "platform"):
            content_profile.platform = _get(cp_section, "platform", content_profile.platform)
        if parser.has_option(cp_section, "caption_length"):
            content_profile.caption_length = _get(cp_section, "caption_length", content_profile.caption_length)
        if parser.has_option(cp_section, "hashtag_count"):
            content_profile.hashtag_count = _getint(cp_section, "hashtag_count", content_profile.hashtag_count)
        if parser.has_option(cp_section, "custom_instructions"):
            content_profile.custom_instructions = _get(cp_section, "custom_instructions", "")
        # Re-validate after all field overrides — direct attribute assignment
        # above bypasses __post_init__ validation, so invalid INI values
        # (e.g. tone=screaming, platform=facebook) would slip through.
        content_profile.__post_init__()
    else:
        content_profile = default_profile

    return PipelineConfig(
        model_name=_get("Model", "model_name", defaults.model_name),
        ollama_host=_get("Model", "ollama_host", defaults.ollama_host),
        scene_threshold=_getfloat("SceneDetection", "threshold", defaults.scene_threshold),
        min_scene_len=_getfloat("SceneDetection", "min_scene_len", defaults.min_scene_len),
        max_scene_len=_getfloat("SceneDetection", "max_scene_len", defaults.max_scene_len),
        top_n_clips=_getint("ClipSelection", "top_n_clips", defaults.top_n_clips),
        min_virality_score=_getfloat("ClipSelection", "min_virality_score", defaults.min_virality_score),
        whisper_model=_get("Model", "whisper_model", defaults.whisper_model),
        segmentation_model=_get("Model", "segmentation_model", defaults.segmentation_model),
        whisper_device=_get("Model", "whisper_device", defaults.whisper_device),
        whisper_compute_type=_get("Model", "whisper_compute_type", defaults.whisper_compute_type),
        num_frames=_getint("Model", "num_frames", defaults.num_frames),
        asmr_mode=_getbool("ASMR Optimization", "asmr_mode", defaults.asmr_mode),
        content_profile=content_profile,
        output_dir=_get("Output", "output_dir", defaults.output_dir),
        subtitle_style=subtitle_style,
        context_padding=_getfloat("Temporal", "context_padding", defaults.context_padding),
        dry_run=_getbool("Output", "dry_run", defaults.dry_run),
        pause_threshold=_getfloat("Segmentation", "pause_threshold", defaults.pause_threshold),
        min_segment_duration=_getfloat("Segmentation", "min_segment_duration", defaults.min_segment_duration),
        max_segment_duration=_getfloat("Segmentation", "max_segment_duration", defaults.max_segment_duration),
        vad_filter=_getbool("Segmentation", "vad_filter", defaults.vad_filter),
        captions=_getbool("Output", "captions", defaults.captions),
        scoring_weights=scoring_weights,
    )


def save_default_config(path: str) -> None:
    """Write a default configuration file for first-time setup.

    Args:
        path: Destination file path.
    """
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    parser = configparser.ConfigParser()

    defaults = PipelineConfig()

    parser["Model"] = {
        "model_name": defaults.model_name,
        "ollama_host": defaults.ollama_host,
        "whisper_model": defaults.whisper_model,
        "segmentation_model": defaults.segmentation_model,
        "whisper_device": defaults.whisper_device,
        "whisper_compute_type": defaults.whisper_compute_type,
        "num_frames": str(defaults.num_frames),
    }

    parser["SceneDetection"] = {
        "threshold": str(defaults.scene_threshold),
        "min_scene_len": str(defaults.min_scene_len),
        "max_scene_len": str(defaults.max_scene_len),
    }

    parser["ClipSelection"] = {
        "top_n_clips": str(defaults.top_n_clips),
        "min_virality_score": str(defaults.min_virality_score),
    }

    parser["ASMR Optimization"] = {
        "asmr_mode": str(defaults.asmr_mode).lower(),
    }

    default_cp = defaults.content_profile
    parser["ContentProfile"] = {
        "content_type": default_cp.content_type,
        "channel_description": default_cp.channel_description,
        "target_audience": default_cp.target_audience,
        "tone": default_cp.tone,
        "platform": default_cp.platform,
        "caption_length": default_cp.caption_length,
        "hashtag_count": str(default_cp.hashtag_count),
        "custom_instructions": default_cp.custom_instructions,
    }

    parser["Output"] = {
        "output_dir": defaults.output_dir,
        "dry_run": str(defaults.dry_run).lower(),
        "captions": str(defaults.captions).lower(),
    }

    parser["Temporal"] = {
        "context_padding": str(defaults.context_padding),
    }

    parser["Segmentation"] = {
        "pause_threshold": str(defaults.pause_threshold),
        "min_segment_duration": str(defaults.min_segment_duration),
        "max_segment_duration": str(defaults.max_segment_duration),
        "vad_filter": str(defaults.vad_filter).lower(),
    }

    default_style = defaults.subtitle_style
    parser["Subtitle"] = {
        "font_name": default_style.font_name,
        "font_size_pct": str(default_style.font_size_pct),
        "primary_color": default_style.primary_color,
        "outline_color": default_style.outline_color,
        "outline_width": str(default_style.outline_width),
        "shadow": str(default_style.shadow),
        "margin_v_pct": str(default_style.margin_v_pct),
        "margin_h_pct": str(default_style.margin_h_pct),
    }

    parser["Scoring"] = {
        "weights": json.dumps(defaults.scoring_weights, indent=2),
    }

    with open(config_path, "w", encoding="utf-8") as f:
        f.write("# YACG Configuration\n")
        f.write("# Edit these settings to customize pipeline behavior\n\n")
        parser.write(f)

    logger.info("Default config written to %s", path)
