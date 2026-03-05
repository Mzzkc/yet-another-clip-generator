"""
Virality scoring engine.

Combines audio, visual, semantic, and temporal signals into a unified
viral-potential score (0-100) using configurable weights tuned for
ASMR content on Instagram Reels.
"""

import logging
from typing import Optional

from viral_clip_extractor.models import (
    AudioFeatures,
    PipelineConfig,
    SemanticFeatures,
    ViralityScore,
    VisualFeatures,
)

logger = logging.getLogger(__name__)

# Default ASMR-optimized weights (from design doc)
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

# Weight categories for redistribution when semantic is unavailable
_SEMANTIC_KEYS = {"hook", "emotional", "asmr", "narrative", "uniqueness"}
_AUDIO_KEYS = {"audio_peaks", "high_freq"}
_VISUAL_KEYS = {"motion", "visual"}


class ViralityScorer:
    """Score video segments for viral potential.

    Combines audio, visual, semantic, and temporal signals into a weighted
    score (0-100). When semantic features are unavailable, their weights
    are redistributed proportionally to audio and visual components.

    Args:
        weights: Custom weight dict mapping component names to floats.
            If None, uses default ASMR-optimized weights.
        config: Pipeline configuration (weights are read from
            ``config.scoring_weights`` if *weights* is not provided).
    """

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        if weights is not None:
            self.weights = dict(weights)
        elif config is not None:
            self.weights = dict(config.scoring_weights)
        else:
            self.weights = dict(_DEFAULT_WEIGHTS)
        self.config = config or PipelineConfig()

    def calculate_score(
        self,
        audio: AudioFeatures,
        visual: VisualFeatures,
        semantic: Optional[SemanticFeatures],
        duration: float,
    ) -> ViralityScore:
        """Compute a composite virality score from multi-modal features.

        All inputs are normalized to a 0-10 scale before weighting.
        When *semantic* is ``None``, semantic weights are redistributed
        proportionally across audio and visual components.

        Args:
            audio: Audio analysis results.
            visual: Visual analysis results.
            semantic: Semantic analysis results (may be ``None``).
            duration: Clip duration in seconds.

        Returns:
            A ``ViralityScore`` with total, component breakdown, and confidence.
        """
        weights = dict(self.weights)

        # Redistribute semantic weights when unavailable
        if semantic is None:
            semantic_total = sum(weights.get(k, 0.0) for k in _SEMANTIC_KEYS)
            av_keys = _AUDIO_KEYS | _VISUAL_KEYS
            av_total = sum(weights.get(k, 0.0) for k in av_keys)

            if av_total > 0:
                for key in av_keys:
                    proportion = weights.get(key, 0.0) / av_total
                    weights[key] = weights.get(key, 0.0) + semantic_total * proportion

            for key in _SEMANTIC_KEYS:
                weights[key] = 0.0

        # Build component scores (all on 0-10 scale)
        components: dict[str, float] = {}

        # Audio components — raw values are typically 0-1 from librosa
        components["audio_peaks"] = self._normalize(audio.audio_peak_score, 0.0, 1.0)
        components["high_freq"] = self._normalize(audio.high_freq_score, 0.0, 1.0)

        # Visual components
        components["motion"] = self._normalize(visual.motion_score, 0.0, 50.0)
        components["visual"] = self._normalize(visual.visual_interest, 0.0, 100.0)

        # Semantic components (already on 0-10 scale per SemanticFeatures doc)
        if semantic is not None:
            components["hook"] = self._normalize(semantic.hook_potential, 0.0, 10.0)
            components["emotional"] = self._normalize(
                semantic.emotional_intensity, 0.0, 10.0
            )
            components["asmr"] = self._normalize(semantic.asmr_quality, 0.0, 10.0)
            components["narrative"] = self._normalize(
                semantic.narrative_interest, 0.0, 10.0
            )
            components["uniqueness"] = self._normalize(semantic.uniqueness, 0.0, 10.0)
        else:
            for key in _SEMANTIC_KEYS:
                components[key] = 0.0

        # Duration component
        components["duration"] = self._duration_score(duration)

        # Weighted sum
        weighted_sum = 0.0
        total_weight = 0.0
        for key, comp_val in components.items():
            w = weights.get(key, 0.0)
            weighted_sum += comp_val * w
            total_weight += w

        # Scale to 0-100 (components are 0-10, so multiply by 10)
        if total_weight > 0:
            total_score = (weighted_sum / total_weight) * 10.0
        else:
            total_score = 0.0

        total_score = max(0.0, min(100.0, total_score))

        # Confidence: full with semantic data, reduced without
        confidence = 1.0 if semantic is not None else 0.5

        logger.debug(
            "Virality score: %.2f (confidence: %.1f, components: %s)",
            total_score,
            confidence,
            components,
        )

        return ViralityScore(
            total_score=round(total_score, 2),
            component_scores=components,
            confidence=confidence,
        )

    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize a value to the 0-10 scale.

        Args:
            value: Raw input value.
            min_val: Expected minimum of the input range.
            max_val: Expected maximum of the input range.

        Returns:
            A float clamped and scaled to 0-10.
        """
        if max_val <= min_val:
            return 0.0
        clamped = max(min_val, min(max_val, value))
        return ((clamped - min_val) / (max_val - min_val)) * 10.0

    def _duration_score(self, duration: float) -> float:
        """Score based on clip duration optimality for Instagram Reels.

        Optimal: 7-30s = 10.  5-7s = linear 5-10.  30-60s = linear 10-5.
        <5s = linear 0-5.  >60s = decay from 5.

        Args:
            duration: Duration in seconds.

        Returns:
            A score from 0 to 10.
        """
        if duration <= 0:
            return 0.0
        if 7 <= duration <= 30:
            return 10.0
        elif 5 <= duration < 7:
            # Linear interpolation from 5 (at 5s) to 10 (at 7s)
            return 5.0 + (duration - 5.0) * 2.5
        elif 30 < duration <= 60:
            # Linear interpolation from 10 (at 30s) to 5 (at 60s)
            return 10.0 - (duration - 30.0) * (5.0 / 30.0)
        elif duration < 5:
            # Linear 0-5 over 0-5s
            return duration
        else:
            # Decay after 60s
            return max(0.0, 5.0 - (duration - 60.0) * 0.1)

    # Public alias for backward compatibility
    def duration_score(self, duration: float) -> float:
        """Score based on clip duration optimality for Instagram Reels.

        Optimal: 7-30s = 10. Acceptable: 5-60s = 5-10. Outside: decays.

        Args:
            duration: Duration in seconds.

        Returns:
            A score from 0 to 10.
        """
        return self._duration_score(duration)
