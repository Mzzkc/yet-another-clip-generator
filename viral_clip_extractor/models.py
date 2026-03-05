"""
Shared data models for the Viral Clip Extractor pipeline.

Every module in the system depends on these dataclasses. They define the
contract between pipeline stages: scene detection, audio/visual/semantic
analysis, virality scoring, and clip extraction.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SceneSegment:
    """A detected scene boundary within a video."""

    start_time: float
    end_time: float
    scene_index: int

    @property
    def duration(self) -> float:
        """Duration of the scene in seconds."""
        return self.end_time - self.start_time


@dataclass
class AudioFeatures:
    """Audio analysis results for a video segment."""

    audio_peak_score: float
    high_freq_score: float
    dynamic_range: float
    zcr_score: float
    trigger_words: list[str] = field(default_factory=list)
    overall_energy: float = 0.0


@dataclass
class VisualFeatures:
    """Visual analysis results for a video segment."""

    motion_score: float
    face_presence: float
    visual_interest: float
    composition_score: float


@dataclass
class SemanticFeatures:
    """Semantic analysis results from LLM-based video understanding."""

    emotional_intensity: float  # 0-10
    narrative_interest: float  # 0-10
    hook_potential: float  # 0-10
    asmr_quality: float  # 0-10
    visual_appeal: float  # 0-10
    uniqueness: float  # 0-10
    description: str = ""


@dataclass
class ViralityScore:
    """Composite virality score with component breakdown."""

    total_score: float  # 0-100
    component_scores: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.5  # 0-1


@dataclass
class ClipData:
    """Complete data for a single extracted clip."""

    scene: SceneSegment
    audio: AudioFeatures
    visual: VisualFeatures
    semantic: Optional[SemanticFeatures] = None
    virality: ViralityScore = field(default_factory=lambda: ViralityScore(total_score=0.0))
    output_path: Optional[str] = None
    caption: Optional[dict] = None


@dataclass
class PipelineConfig:
    """Configuration for the viral clip extraction pipeline."""

    # Model settings
    model_name: str = "qwen2.5-vl:7b"
    ollama_host: str = "http://localhost:11434"

    # Scene detection
    scene_threshold: float = 3.0
    min_scene_len: float = 7.0
    max_scene_len: float = 60.0

    # Clip selection
    top_n_clips: int = 10
    min_virality_score: float = 70.0

    # Feature toggles
    enable_semantic: bool = True
    enable_captions: bool = True
    vertical_crop: bool = True
    asmr_mode: bool = True

    # Output
    output_dir: str = "./clip_output"

    # Temporal
    context_padding: float = 2.0

    # Scoring weights (ASMR-optimized defaults from design doc)
    scoring_weights: dict = field(default_factory=lambda: {
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
    })


@dataclass
class ProcessingResult:
    """Final result of processing a video through the pipeline."""

    video_path: str
    video_title: str
    clips: list[ClipData]
    total_scenes: int
    processing_time_seconds: float
    errors: list[str] = field(default_factory=list)
