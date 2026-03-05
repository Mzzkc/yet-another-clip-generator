"""
Configuration loader for the Viral Clip Extractor.

Reads pipeline settings from an INI file (compatible with the existing
config.ini format) and falls back to PipelineConfig defaults for any
missing values.
"""

import configparser
import json
import logging
from pathlib import Path
from typing import Optional

from viral_clip_extractor.models import PipelineConfig

logger = logging.getLogger(__name__)

# Default ASMR-optimized scoring weights
_DEFAULT_WEIGHTS: dict[str, float] = {
    "hook": 0.20,
    "emotional": 0.15,
    "audio_peaks": 0.15,
    "asmr": 0.12,
    "motion": 0.12,
    "narrative": 0.10,
    "high_freq": 0.10,
    "uniqueness": 0.08,
    "visual": 0.07,
    "duration": 0.05,
}


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

    return PipelineConfig(
        model_name=_get("Model", "model_name", defaults.model_name),
        ollama_host=_get("Model", "ollama_host", defaults.ollama_host),
        scene_threshold=_getfloat("SceneDetection", "threshold", defaults.scene_threshold),
        min_scene_len=_getfloat("SceneDetection", "min_scene_len", defaults.min_scene_len),
        max_scene_len=_getfloat("SceneDetection", "max_scene_len", defaults.max_scene_len),
        top_n_clips=_getint("ClipSelection", "top_n_clips", defaults.top_n_clips),
        min_virality_score=_getfloat("ClipSelection", "min_virality_score", defaults.min_virality_score),
        enable_semantic=_getbool("Features", "enable_semantic", defaults.enable_semantic),
        enable_captions=_getbool("Features", "enable_captions", defaults.enable_captions),
        vertical_crop=_getbool("Features", "vertical_crop", defaults.vertical_crop),
        asmr_mode=_getbool("ASMR Optimization", "asmr_mode", defaults.asmr_mode),
        output_dir=_get("Output", "output_dir", defaults.output_dir),
        context_padding=_getfloat("Temporal", "context_padding", defaults.context_padding),
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

    parser["Features"] = {
        "enable_semantic": str(defaults.enable_semantic).lower(),
        "enable_captions": str(defaults.enable_captions).lower(),
        "vertical_crop": str(defaults.vertical_crop).lower(),
    }

    parser["ASMR Optimization"] = {
        "asmr_mode": str(defaults.asmr_mode).lower(),
    }

    parser["Output"] = {
        "output_dir": defaults.output_dir,
    }

    parser["Temporal"] = {
        "context_padding": str(defaults.context_padding),
    }

    parser["Scoring"] = {
        "weights": json.dumps(defaults.scoring_weights, indent=2),
    }

    with open(config_path, "w", encoding="utf-8") as f:
        f.write("# Viral Clip Extractor Configuration\n")
        f.write("# Edit these settings to customize pipeline behavior\n\n")
        parser.write(f)

    logger.info("Default config written to %s", path)
