"""
Shared data models for the YACG pipeline.

Every module in the system depends on these dataclasses. They define the
contract between pipeline stages: scene detection, audio/visual/semantic
analysis, virality scoring, and clip extraction.
"""

import os
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
class WordTimestamp:
    """A single word with its timing from Whisper transcription."""

    word: str
    start: float  # seconds
    end: float  # seconds
    probability: float = 1.0


@dataclass
class SegmentBoundary:
    """LLM-identified segment boundary from transcript analysis."""

    start_time: float
    end_time: float
    hook_summary: str
    segment_type: str  # e.g. "hook", "narrative_arc", "complete_thought"


@dataclass
class CaptionData:
    """Generated caption data for short-form social media content."""

    hook: str
    description: str
    hashtags: list[str]
    category: str
    virality_score: int
    full_caption: str


@dataclass
class ClipData:
    """Complete data for a single extracted clip."""

    scene: SceneSegment
    audio: AudioFeatures
    visual: VisualFeatures
    semantic: SemanticFeatures
    virality: ViralityScore = field(default_factory=lambda: ViralityScore(total_score=0.0))
    output_path: Optional[str] = None
    caption: Optional[CaptionData] = None
    words: list[WordTimestamp] = field(default_factory=list)
    thumbnail_path: Optional[str] = None

    def __repr__(self) -> str:
        path = os.path.basename(self.output_path) if self.output_path else "N/A"
        return (
            f"ClipData(path={path!r}, "
            f"score={self.virality.total_score:.1f}, "
            f"time={self.scene.start_time:.1f}-{self.scene.end_time:.1f}s)"
        )


VALID_SCORING_KEYS: frozenset[str] = frozenset({
    "hook", "emotional", "audio_peaks", "asmr", "motion",
    "narrative", "high_freq", "uniqueness", "composition",
    "visual", "duration",
})


@dataclass
class SubtitleStyle:
    """Configurable subtitle styling for ASS subtitle generation.

    Dimensions expressed as percentages of frame size are resolution-
    independent. Colors use ASS ``&HAABBGGRR`` hex format.
    """

    font_name: str = ""  # empty = auto-detect via _find_system_font()
    font_size_pct: float = 0.055  # fraction of frame height
    primary_color: str = "&H00FFFFFF"  # white — text color when karaoke off, OR resting color when karaoke on
    outline_color: str = "&H00000000"  # black
    outline_width: float = 3.0
    shadow: float = 1.5
    # Vertical margin as fraction of frame height. 0.38 places text at
    # ~62% from top, which clears the TikTok/Reels bottom UI bar (~15-20%
    # of screen) and the top status bar (~5%). Instagram Reels safe zone
    # spans roughly 10%-85% of frame height; 62% sits safely inside.
    margin_v_pct: float = 0.38
    margin_h_pct: float = 0.15  # horizontal margin as fraction of frame width

    # ASS BorderStyle.  1 = outline + shadow (current behavior, default).
    # 3 = opaque box behind the text — much more readable against busy
    # backgrounds (VRChat scenes, particle effects, anime/cartoon
    # textures), at the cost of partial visual occlusion.  Composers
    # whose content has high visual chaos should prefer 3.
    border_style: int = 1

    # Karaoke per-word highlight.  When True, generate_ass emits ASS
    # ``\k<centiseconds>`` tags per word so the subtitle progressively
    # transitions from `primary_color` (resting) to `karaoke_active_color`
    # (active/spoken) as the dialogue plays.  Common short-form pattern
    # for ASMR / hypnosis brand content where the active word draws the
    # eye to the trigger / suggestion word being delivered.
    karaoke: bool = False
    # Color of the actively-spoken word when karaoke is enabled.  Default
    # is Ahamkara green (#22E06B in #RRGGBB) → ASS &HAABBGGRR =
    # &H006BE022.  Has no effect when karaoke=False.
    karaoke_active_color: str = "&H006BE022"

    # Maximum words per subtitle group.  Caps how many words the burner
    # may bundle into a single on-screen group.  Default 2 favors
    # single-line layouts on narrow vertical crops (1080-1215 px wide)
    # and avoids wrap when individual words are long (e.g. ASMR speakers
    # using made-up names like "andragor").  Bump to 3 for normal-rate
    # content with shorter words; lower to 1 for single-word-pop style.
    max_words_per_group: int = 2


VALID_CONTENT_TYPES: frozenset[str] = frozenset({
    "general", "gaming", "cooking", "asmr", "educational",
    "fitness", "comedy", "music", "beauty", "tech", "vlog",
})

VALID_TONES: frozenset[str] = frozenset({
    "energetic", "calm", "professional", "casual",
    "humorous", "inspirational", "dramatic", "engaging",
})

VALID_PLATFORMS: frozenset[str] = frozenset({
    "tiktok", "reels", "shorts", "all",
})

VALID_CAPTION_LENGTHS: frozenset[str] = frozenset({
    "short", "medium", "long",
})


@dataclass
class ContentProfile:
    """Content profile controlling LLM prompt behavior across the pipeline.

    Determines how segmentation, virality analysis, and caption generation
    prompts are constructed. Presets provide sensible defaults per genre;
    individual fields can be overridden via CLI or INI config.
    """

    content_type: str = "general"
    channel_description: str = ""
    target_audience: str = ""
    tone: str = "engaging"
    platform: str = "all"
    caption_length: str = "medium"
    hashtag_count: int = 5
    custom_instructions: str = ""

    def __post_init__(self) -> None:
        if self.content_type not in VALID_CONTENT_TYPES:
            raise ValueError(
                f"Invalid content_type '{self.content_type}'. "
                f"Valid: {sorted(VALID_CONTENT_TYPES)}"
            )
        if self.tone and self.tone not in VALID_TONES:
            raise ValueError(
                f"Invalid tone '{self.tone}'. Valid: {sorted(VALID_TONES)}"
            )
        if self.platform not in VALID_PLATFORMS:
            raise ValueError(
                f"Invalid platform '{self.platform}'. "
                f"Valid: {sorted(VALID_PLATFORMS)}"
            )
        if self.caption_length not in VALID_CAPTION_LENGTHS:
            raise ValueError(
                f"Invalid caption_length '{self.caption_length}'. "
                f"Valid: {sorted(VALID_CAPTION_LENGTHS)}"
            )
        if not (3 <= self.hashtag_count <= 7):
            raise ValueError(
                f"hashtag_count must be 3-7, got {self.hashtag_count}"
            )


CONTENT_PRESETS: dict[str, ContentProfile] = {
    "general": ContentProfile(
        content_type="general",
        target_audience="general social media viewers",
        tone="engaging",
        platform="all",
        caption_length="medium",
        hashtag_count=5,
    ),
    "gaming": ContentProfile(
        content_type="gaming",
        target_audience="gamers and gaming enthusiasts",
        tone="energetic",
        platform="all",
        caption_length="short",
        hashtag_count=5,
    ),
    "cooking": ContentProfile(
        content_type="cooking",
        target_audience="home cooks and food enthusiasts",
        tone="casual",
        platform="reels",
        caption_length="medium",
        hashtag_count=5,
    ),
    "asmr": ContentProfile(
        content_type="asmr",
        target_audience="ASMR listeners seeking relaxation and tingles",
        tone="calm",
        platform="all",
        caption_length="short",
        hashtag_count=4,
    ),
    "educational": ContentProfile(
        content_type="educational",
        target_audience="learners and curious minds",
        tone="professional",
        platform="shorts",
        caption_length="long",
        hashtag_count=4,
    ),
    "fitness": ContentProfile(
        content_type="fitness",
        target_audience="fitness enthusiasts and gym-goers",
        tone="energetic",
        platform="reels",
        caption_length="medium",
        hashtag_count=5,
    ),
    "comedy": ContentProfile(
        content_type="comedy",
        target_audience="people looking for entertainment and laughs",
        tone="humorous",
        platform="tiktok",
        caption_length="short",
        hashtag_count=5,
    ),
    "music": ContentProfile(
        content_type="music",
        target_audience="music lovers and artists",
        tone="engaging",
        platform="all",
        caption_length="short",
        hashtag_count=5,
    ),
    "beauty": ContentProfile(
        content_type="beauty",
        target_audience="beauty and skincare enthusiasts",
        tone="casual",
        platform="reels",
        caption_length="medium",
        hashtag_count=5,
    ),
    "tech": ContentProfile(
        content_type="tech",
        target_audience="tech enthusiasts and early adopters",
        tone="professional",
        platform="shorts",
        caption_length="medium",
        hashtag_count=4,
    ),
    "vlog": ContentProfile(
        content_type="vlog",
        target_audience="viewers interested in personal stories and daily life",
        tone="casual",
        platform="all",
        caption_length="medium",
        hashtag_count=5,
    ),
}


@dataclass
class PipelineConfig:
    """Configuration for the viral clip extraction pipeline."""

    # Model settings
    model_name: str = "qwen2.5-vl:7b"
    ollama_host: str = "http://localhost:11434"
    whisper_model: str = "small"
    # Ollama model for text-only segmentation.  Defaults to qwen2.5:7b (the
    # text-only variant) which is faster and avoids loading the unused vision
    # encoder into VRAM.  Falls back to model_name (VLM) if set to "".
    segmentation_model: str = "qwen2.5:7b"

    # Whisper hardware settings ("auto" = CUDA if available, else CPU/int8)
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"

    # Scene detection
    scene_threshold: float = 3.0
    min_scene_len: float = 7.0
    max_scene_len: float = 60.0

    # Clip selection
    top_n_clips: int = 10
    min_virality_score: float = 70.0

    asmr_mode: bool = False

    # Content profile drives all LLM prompt behavior (segmentation, virality,
    # captions). Replaces the old content_type field.
    content_profile: ContentProfile = field(default_factory=ContentProfile)

    # Number of frames to extract per segment for VLM analysis.
    # qwen2.5-vl:7b produces garbled output with multiple images,
    # so default is 1. Increase only if your VLM handles multi-image well.
    num_frames: int = 3

    # Output
    output_dir: str = "./clip_output"

    # Subtitle styling
    subtitle_style: SubtitleStyle = field(default_factory=SubtitleStyle)

    # Temporal
    context_padding: float = 2.0

    # Dry-run: run analysis only, skip extraction/subtitles/captions
    dry_run: bool = False

    # Segmentation tuning
    pause_threshold: float = 0.3
    min_segment_duration: float = 15.0
    max_segment_duration: float = 45.0
    # VAD filter removes non-speech segments before transcription.  Set to
    # False for ASMR/ambient content where non-speech audio is the primary
    # content and VAD would incorrectly discard relevant segments.
    vad_filter: bool = True
    # Whether to run the VLM caption-generation step (Step 7) per clip.
    # Default True preserves the original yacg pipeline behavior.  Set to
    # False when yacg is used mid-pipeline by a caller that generates its
    # own captions downstream — skipping the step is faster AND prevents
    # caption-failure from deleting otherwise-good clips.  See
    # `--no-captions` flag.
    captions: bool = True
    # Whether to use the VLM (configured via model_name) to localize the
    # main subject horizontally for vertical cropping, instead of relying
    # on OpenCV face detection.  Default False preserves the existing
    # SmartCropper face-detection behavior.  Set to True for content where
    # face detection misfires — non-human characters (furry/anthro/cartoon
    # avatars), heavy stylization, abstract subjects, masked subjects.
    # The VLM is asked once per crop call to return a horizontal subject
    # position as a fraction (0.0=left edge, 1.0=right edge).  Falls back
    # to face detection (and then center crop) if the VLM call fails or
    # returns an unparseable response.
    vlm_crop: bool = False

    # Minimum word-confidence (faster-whisper `probability`) for inclusion
    # in the burned subtitle stream.  Default 0.0 = include all words
    # (preserves existing yacg behavior).  Set to e.g. 0.5 to filter out
    # whisper hallucinations in long silent regions — common in
    # ASMR pre-roll/post-roll where whisper invents word-loops that would
    # otherwise burn into subtitles as "things not said."  Words with
    # ``probability is None`` are always kept (some whisper backends omit
    # the field).
    subtitle_min_word_probability: float = 0.0

    # Scoring weights (ASMR-optimized defaults from design doc).
    # Weights intentionally sum > 1.0 (currently 1.14) to emphasize key
    # features in ASMR-optimized mode. This is by design — do not normalize.
    scoring_weights: dict[str, float] = field(default_factory=lambda: {
        "hook": 0.20,
        "emotional": 0.15,
        "audio_peaks": 0.15,
        "asmr": 0.12,
        "motion": 0.12,
        "narrative": 0.10,
        "high_freq": 0.10,
        "uniqueness": 0.08,
        "composition": 0.05,
        "visual": 0.02,
        "duration": 0.05,
    })

    def __post_init__(self) -> None:
        invalid_keys = set(self.scoring_weights.keys()) - VALID_SCORING_KEYS
        if invalid_keys:
            raise ValueError(
                f"Invalid scoring weight key(s): {sorted(invalid_keys)}. "
                f"Valid keys: {sorted(VALID_SCORING_KEYS)}"
            )
        # Auto-sync asmr_mode from content profile
        if self.content_profile.content_type == "asmr":
            self.asmr_mode = True


@dataclass
class ProcessingResult:
    """Final result of processing a video through the pipeline."""

    video_path: str
    video_title: str
    clips: list[ClipData]
    total_scenes: int
    processing_time_seconds: float
    errors: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"ProcessingResult(title={self.video_title!r}, "
            f"clips={len(self.clips)}, scenes={self.total_scenes}, "
            f"time={self.processing_time_seconds:.1f}s, "
            f"errors={len(self.errors)})"
        )
