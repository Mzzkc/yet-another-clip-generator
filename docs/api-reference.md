# API Reference

This page documents the key public classes and functions in the `yacg` package.

## Public Exports

The package exports these symbols from `yacg/__init__.py`:

```python
from yacg import ClipData, PipelineConfig, ViralClipPipeline, __version__
```

## Data Models

All data models are defined in `yacg/models.py`.

### PipelineConfig

Configuration for the viral clip extraction pipeline.

```python
from yacg.models import PipelineConfig, ContentProfile, SubtitleStyle

config = PipelineConfig(
    model_name="qwen2.5-vl:7b",
    ollama_host="http://localhost:11434",
    whisper_model="small",
    segmentation_model="qwen2.5:7b",
    whisper_device="auto",
    whisper_compute_type="auto",
    scene_threshold=3.0,
    min_scene_len=7.0,
    max_scene_len=60.0,
    top_n_clips=10,
    min_virality_score=70.0,
    asmr_mode=False,
    content_profile=ContentProfile(content_type="general"),
    num_frames=3,
    output_dir="./clip_output",
    subtitle_style=SubtitleStyle(),
    context_padding=2.0,
    dry_run=False,
    pause_threshold=0.3,
    min_segment_duration=15.0,
    max_segment_duration=45.0,
    vad_filter=True,
    scoring_weights={...},
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `str` | `"qwen2.5-vl:7b"` | Ollama vision model for semantic analysis and captions |
| `ollama_host` | `str` | `"http://localhost:11434"` | Ollama API endpoint |
| `whisper_model` | `str` | `"small"` | faster-whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `segmentation_model` | `str` | `"qwen2.5:7b"` | Ollama text-only model for transcript segmentation |
| `whisper_device` | `str` | `"auto"` | Whisper device: `auto` (CUDA if available), `cpu`, `cuda` |
| `whisper_compute_type` | `str` | `"auto"` | Whisper precision: `auto`, `int8`, `float16`, `float32` |
| `scene_threshold` | `float` | `3.0` | Scene detection threshold |
| `min_scene_len` | `float` | `7.0` | Minimum scene length in seconds |
| `max_scene_len` | `float` | `60.0` | Maximum scene length in seconds |
| `top_n_clips` | `int` | `10` | Default number of clips to extract. **Note:** This field is effectively unused — `process_video()` defaults `top_n=20` via its method signature, and the CLI always passes `--top-n`. No code path reads `config.top_n_clips`. Pass `top_n` directly to `process_video()` instead |
| `min_virality_score` | `float` | `70.0` | Default minimum virality score. **Note:** This field is effectively unused — `process_video()` defaults `min_score=70.0` via its method signature, and the CLI always passes `--min-score`. No code path reads `config.min_virality_score`. Pass `min_score` directly to `process_video()` instead |
| `asmr_mode` | `bool` | `False` | Enable ASMR-optimized scoring. Auto-enabled when `content_profile.content_type == "asmr"` |
| `content_profile` | `ContentProfile` | `ContentProfile()` | Content type and caption configuration (see [ContentProfile](#contentprofile) below) |
| `num_frames` | `int` | `3` | Number of JPEG frames per segment for VLM semantic analysis (1-5). Set to `1` if your VLM produces garbled output with multiple images. Affects `SemanticAnalyzer` only — `OllamaVideoAnalyzer` (caption generation) always uses 3 frames regardless |
| `output_dir` | `str` | `"./clip_output"` | Output directory |
| `subtitle_style` | `SubtitleStyle` | `SubtitleStyle()` | Subtitle appearance configuration |
| `context_padding` | `float` | `2.0` | Seconds of padding around clip boundaries |
| `dry_run` | `bool` | `False` | Analysis only — stop after step 4 |
| `pause_threshold` | `float` | `0.3` | Speech pause duration (seconds) that triggers a segment boundary |
| `min_segment_duration` | `float` | `15.0` | Minimum segment duration in seconds |
| `max_segment_duration` | `float` | `45.0` | Maximum segment duration in seconds |
| `vad_filter` | `bool` | `True` | Voice Activity Detection filter for Whisper |
| `scoring_weights` | `dict[str, float]` | *(see below)* | Virality scoring weight dictionary (11 keys, sum=1.14) |

Scoring weights are validated on construction: keys must be from `VALID_SCORING_KEYS` (`hook`, `emotional`, `audio_peaks`, `asmr`, `motion`, `narrative`, `high_freq`, `uniqueness`, `composition`, `visual`, `duration`). Invalid keys raise `ValueError`.

### SubtitleStyle

Configuration for subtitle appearance. All values can be set via CLI flags or INI config.

```python
@dataclass
class SubtitleStyle:
    font_name: str = ""            # auto-detected if empty
    font_size_pct: float = 0.055   # fraction of video height
    primary_color: str = "&H00FFFFFF"   # white (ASS &HAABBGGRR format)
    outline_color: str = "&H00000000"   # black
    outline_width: float = 3.0
    shadow: float = 1.5
    margin_v_pct: float = 0.38     # fraction of video height
    margin_h_pct: float = 0.15     # fraction of video width
```

Colors use the ASS `&HAABBGGRR` hex format (note: blue-green-red order, not RGB).

### ContentProfile

Configuration for content-type-aware LLM prompts and caption generation. Defined in `yacg/models.py`.

```python
from yacg.models import ContentProfile

# Use a preset
profile = ContentProfile(content_type="gaming")

# Customize fully
profile = ContentProfile(
    content_type="cooking",
    channel_description="Home chef sharing quick recipes",
    target_audience="busy professionals who want to cook",
    tone="casual",
    platform="reels",
    caption_length="short",
    hashtag_count=4,
    custom_instructions="Always mention the recipe name",
)
```

| Field | Type | Default | Valid Values |
|-------|------|---------|-------------|
| `content_type` | `str` | `"general"` | `general`, `gaming`, `cooking`, `asmr`, `educational`, `fitness`, `comedy`, `music`, `beauty`, `tech`, `vlog` |
| `channel_description` | `str` | `""` | Free text |
| `target_audience` | `str` | `""` | Free text |
| `tone` | `str` | `"engaging"` | `energetic`, `calm`, `professional`, `casual`, `humorous`, `inspirational`, `dramatic`, `engaging` |
| `platform` | `str` | `"all"` | `tiktok`, `reels`, `shorts`, `all` |
| `caption_length` | `str` | `"medium"` | `short`, `medium`, `long` |
| `hashtag_count` | `int` | `5` | `3`-`7` |
| `custom_instructions` | `str` | `""` | Free text appended to LLM prompts |

All enum-like fields are validated in `__post_init__` — invalid values raise `ValueError`.

#### Content Presets

`CONTENT_PRESETS` is a `dict[str, ContentProfile]` with 11 pre-built profiles (one per content type). When `--content-type` is set via CLI, the corresponding preset is loaded, pre-filling `target_audience`, `tone`, `platform`, `caption_length`, and `hashtag_count`. Individual CLI flags (e.g., `--tone calm`) override the preset values.

```python
from yacg.models import CONTENT_PRESETS

gaming_profile = CONTENT_PRESETS["gaming"]
# ContentProfile(content_type="gaming", tone="energetic", platform="all", ...)
```

### CaptionData

AI-generated caption data for a clip.

```python
@dataclass
class CaptionData:
    hook: str              # Opening hook line
    description: str       # Caption body text
    hashtags: list[str]    # Hashtag list
    category: str          # Content category
    virality_score: int    # LLM-predicted virality (0-100)
    full_caption: str      # Assembled ready-to-post caption
```

### SceneSegment

A detected scene boundary within a video.

```python
@dataclass
class SceneSegment:
    start_time: float
    end_time: float
    scene_index: int

    @property
    def duration(self) -> float: ...
```

### WordTimestamp

A single word with its timing from Whisper transcription.

```python
@dataclass
class WordTimestamp:
    word: str
    start: float        # seconds
    end: float          # seconds
    probability: float  # 0-1, default 1.0
```

### SegmentBoundary

LLM-identified segment boundary from transcript analysis.

```python
@dataclass
class SegmentBoundary:
    start_time: float
    end_time: float
    hook_summary: str
    segment_type: str   # "hook", "narrative_arc", "complete_thought", "emotional_peak"
```

### AudioFeatures

Audio analysis results for a video segment.

```python
@dataclass
class AudioFeatures:
    audio_peak_score: float    # 0-2, 90th percentile RMS energy (normalized to 0-10 by ViralityScorer)
    high_freq_score: float     # 0-1, fraction of frames >4kHz
    dynamic_range: float       # std of RMS
    zcr_score: float           # zero-crossing rate
    trigger_words: list[str]   # detected trigger words
    overall_energy: float      # mean RMS energy
```

### VisualFeatures

Visual analysis results for a video segment.

```python
@dataclass
class VisualFeatures:
    motion_score: float        # 0-1, optical flow magnitude
    face_presence: float       # 0-1, fraction of frames with faces
    visual_interest: float     # 0-1, HSV color variance
    composition_score: float   # 0-1, rule-of-thirds proximity
```

### SemanticFeatures

Semantic analysis results from LLM-based video understanding.

```python
@dataclass
class SemanticFeatures:
    emotional_intensity: float  # 0-10
    narrative_interest: float   # 0-10
    hook_potential: float       # 0-10
    asmr_quality: float         # 0-10
    visual_appeal: float        # 0-10
    uniqueness: float           # 0-10
    description: str            # LLM-generated description
```

### ViralityScore

Composite virality score with component breakdown.

```python
@dataclass
class ViralityScore:
    total_score: float                  # 0-100
    component_scores: dict[str, float]  # per-component breakdown
    confidence: float                   # 0-1 (1.0 with semantic, 0.5 without)
```

### ClipData

Complete data for a single extracted clip.

```python
@dataclass
class ClipData:
    scene: SceneSegment
    audio: AudioFeatures
    visual: VisualFeatures
    semantic: SemanticFeatures
    virality: ViralityScore
    output_path: Optional[str]
    caption: Optional[CaptionData]
    words: list[WordTimestamp]
    thumbnail_path: Optional[str]
```

### ProcessingResult

Final result of processing a video through the pipeline.

```python
@dataclass
class ProcessingResult:
    video_path: str
    video_title: str
    clips: list[ClipData]
    total_scenes: int
    processing_time_seconds: float
    errors: list[str]
```

---

## Pipeline

### ViralClipPipeline

The main orchestrator class. Defined in `yacg/pipeline.py`.

```python
from yacg import PipelineConfig, ViralClipPipeline

config = PipelineConfig(output_dir="./my_clips")
pipeline = ViralClipPipeline(config=config)
```

#### Constructor

The constructor accepts optional injectable implementations for all 8 pipeline components, plus a progress callback:

```python
pipeline = ViralClipPipeline(
    config=config,
    transcript_segmenter=None,  # Optional[TranscriptSegmenterProtocol]
    subtitle_burner=None,       # Optional[SubtitleBurnerProtocol]
    semantic_analyzer=None,     # Optional[SemanticAnalyzerProtocol]
    caption_generator=None,     # Optional[CaptionGeneratorProtocol]
    audio_analyzer=None,        # Optional[AudioAnalyzerProtocol]
    visual_analyzer=None,       # Optional[VisualAnalyzerProtocol]
    virality_scorer=None,       # Optional[ViralityScorerProtocol]
    clip_extractor=None,        # Optional[ClipExtractorProtocol]
    progress_callback=None,     # Optional[Callable[[str, int, int], None]]
)
```

All component parameters default to `None`, in which case the default implementation is lazily initialized on first use.

The `progress_callback` receives `(step_name: str, current: int, total: int)` and can be used for GUI/web progress reporting.

Exceptions raised by the `progress_callback` are caught silently and logged at DEBUG level. They will not propagate to the caller or affect pipeline execution.

Progress step names emitted by the pipeline: `transcribe`, `segment`, `analyze`, `analyze_segment`, `extract`, `subtitles`, `captions`.

#### `process_video()`

Process a local video file through the full pipeline (7 numbered steps plus post-processing).

```python
result = pipeline.process_video(
    video_path="/path/to/video.mp4",
    title="My Video",
    top_n=5,
    min_score=60.0,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_path` | `str` | *(required)* | Path to the video file |
| `title` | `str` | `""` | Video title for captions (auto-detected from FFprobe metadata if empty) |
| `top_n` | `int` | `20` | Number of top clips to extract |
| `min_score` | `float` | `70.0` | Minimum virality score threshold |

**Returns:** `ProcessingResult`

#### `process_youtube()`

Download and process a YouTube video.

```python
result = pipeline.process_youtube(
    url="https://youtube.com/watch?v=XXXXX",
    top_n=3,
    min_score=50.0,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | `str` | *(required)* | YouTube video URL |
| `top_n` | `int` | `20` | Number of top clips to extract |
| `min_score` | `float` | `70.0` | Minimum virality score threshold |

**Returns:** `ProcessingResult`

---

## Protocol Interfaces

The pipeline defines 8 `Protocol` interfaces in `pipeline.py` for dependency injection. All are `@runtime_checkable`.

### TranscriptSegmenterProtocol

```python
class TranscriptSegmenterProtocol(Protocol):
    def full_transcribe(self, video_path: str) -> list[WordTimestamp]: ...
    def segment_by_content(self, words: list[WordTimestamp], title: str, target_count: int = 20) -> list[SegmentBoundary]: ...
    def refine_boundaries(self, boundaries: list[SegmentBoundary], words: list[WordTimestamp]) -> list[SceneSegment]: ...
```

#### TranscriptSegmenter (Concrete Class)

The default implementation with Whisper transcription and Ollama LLM segmentation. Defined in `yacg/transcript_segmenter.py`.

```python
from yacg.transcript_segmenter import TranscriptSegmenter

segmenter = TranscriptSegmenter(
    whisper_model="small",           # Whisper model size
    ollama_host="http://localhost:11434",
    model_name="qwen2.5-vl:7b",     # Ollama model for segmentation (pipeline passes config.segmentation_model)
    whisper_device="auto",           # "auto", "cpu", or "cuda"
    whisper_compute_type="auto",     # "auto", "int8", "float16", "float32"
    pause_threshold=0.3,             # speech pause duration (seconds) for boundary snapping
    min_segment_duration=15.0,       # minimum segment length
    max_segment_duration=45.0,       # maximum segment length
    vad_filter=True,                 # Voice Activity Detection filter
    content_type="",                 # content type for genre-specific segmentation guidance
    channel_description="",          # optional channel context
    target_audience="",              # optional audience context
    custom_instructions="",          # optional custom LLM instructions
)
```

**Convenience method:**

```python
# High-level method chaining full_transcribe → segment_by_content → refine_boundaries
scenes, words = segmenter.segment_video(
    video_path="/path/to/video.mp4",
    title="My Video",
)
# scenes: list[SceneSegment], words: list[WordTimestamp]
```

The Whisper model instance is cached and keyed by `(model, device, compute_type)` — reused across calls and batch processing.

### SubtitleBurnerProtocol

```python
class SubtitleBurnerProtocol(Protocol):
    def get_video_dimensions(self, video_path: str) -> tuple[int, int]: ...
    def process_clip(self, video_path: str, words: list[WordTimestamp],
                     width: int, height: int, style: Optional[SubtitleStyle] = None) -> str: ...
```

### SemanticAnalyzerProtocol

```python
class SemanticAnalyzerProtocol(Protocol):
    def analyze_segment(self, video_path: str, start: float,
                        end: float, title: str = "") -> SemanticFeatures: ...
```

### CaptionGeneratorProtocol

```python
class CaptionGeneratorProtocol(Protocol):
    def analyze_video(self, video_path: str, title: str, transcript_text: str = "") -> CaptionData: ...
```

#### OllamaVideoAnalyzer (Concrete Class)

The default caption generator using Ollama VLM. Defined in `yacg/caption_generator.py`.

```python
from yacg.caption_generator import OllamaVideoAnalyzer

analyzer = OllamaVideoAnalyzer(
    model="qwen2.5-vl:7b",          # Ollama vision model
    ollama_host="http://localhost:11434",
    content_type="general",          # content type persona for captions
    channel_description="",          # channel context for captions
    target_audience="",              # audience context
    tone="",                         # caption tone (empty = default per content type)
    platform="",                     # target platform (empty = all)
    caption_length="",               # caption length (empty = medium)
    hashtag_count=5,                 # hashtags per caption
    custom_instructions="",          # appended to LLM prompt
)
```

The `analyze_video()` method accepts an additional `max_retries: int = 3` parameter not present in the Protocol interface:

```python
caption = analyzer.analyze_video(
    video_path="/path/to/clip.mp4",
    title="My Video",
    max_retries=3,          # retry count (default: 3)
    transcript_text="...",  # optional transcript for context
)
# caption: CaptionData
```

Caption generation always extracts 3 JPEG frames at 25%, 50%, and 75% of clip duration, regardless of `PipelineConfig.num_frames`. Uses `temperature=0.7` for creative output.

### AudioAnalyzerProtocol

```python
class AudioAnalyzerProtocol(Protocol):
    def analyze_segment(self, video_path: str, start_time: float,
                        end_time: float, words: Optional[list[WordTimestamp]] = None) -> AudioFeatures: ...
```

### VisualAnalyzerProtocol

```python
class VisualAnalyzerProtocol(Protocol):
    def analyze_segment(self, video_path: str, start_time: float,
                        end_time: float) -> VisualFeatures: ...
```

### ViralityScorerProtocol

```python
class ViralityScorerProtocol(Protocol):
    def calculate_score(self, audio: AudioFeatures, visual: VisualFeatures,
                        semantic: Optional[SemanticFeatures],
                        duration: float) -> ViralityScore: ...
```

### ClipExtractorProtocol

```python
class ClipExtractorProtocol(Protocol):
    def extract_clip(self, video_path: str, start_time: float,
                     end_time: float, output_path: str) -> bool: ...
```

---

## Analyzers

### AudioAnalyzer

Computes audio features using librosa. Defined in `yacg/core/audio_analyzer.py`.

```python
from yacg.core.audio_analyzer import AudioAnalyzer

analyzer = AudioAnalyzer(asmr_keywords=["tapping", "scratching", "whispering"])
features = analyzer.analyze_segment(
    video_path="/path/to/video.mp4",
    start_time=10.0,
    end_time=40.0,
    words=[...],  # optional list[WordTimestamp] — reused from pipeline transcription
)
# features: AudioFeatures
```

Detects: audio peaks, high-frequency content, tapping (onset transients), crinkle (spectral flux), mouth-sounds (mid-frequency energy), and trigger words (from pipeline's existing word timestamps).

### VisualAnalyzer

Computes visual features using OpenCV. Defined in `yacg/core/visual_analyzer.py`.

```python
from yacg.core.visual_analyzer import VisualAnalyzer

analyzer = VisualAnalyzer(config=config)  # optional PipelineConfig
features = analyzer.analyze_segment(
    video_path="/path/to/video.mp4",
    start_time=10.0,
    end_time=40.0,
)
# features: VisualFeatures
```

Computes: motion (optical flow), face presence (DNN SSD detector with Haar cascade fallback), visual interest (HSV variance), composition (rule-of-thirds).

`detect_faces(video_path: str, time_seconds: float) -> int` — Count faces at a specific timestamp. Uses DNN SSD detector with Haar cascade fallback.

### SemanticAnalyzer

Semantic analysis via Ollama VLM. Defined in `yacg/core/semantic_analyzer.py`.

```python
from yacg.core.semantic_analyzer import SemanticAnalyzer

analyzer = SemanticAnalyzer(
    model="qwen2.5-vl:7b",
    ollama_host="http://localhost:11434",
    num_frames=3,            # 1-5 JPEG frames per segment
    content_type="general",  # 11 content types — controls prompt language
    channel_description="",  # optional channel context
    target_audience="",      # optional audience context
    tone="",                 # optional tone override
    custom_instructions="",  # optional custom LLM instructions
)
features = analyzer.analyze_segment(
    video_path="/path/to/video.mp4",
    start_time=10.0,
    end_time=40.0,
    title="My Video",
)
# features: SemanticFeatures
```

Extracts `num_frames` JPEG frames at evenly-spaced positions and sends them to the Ollama VLM. Parses 6 ratings (0-10 scale). Uses `requests.Session()` for HTTP connection reuse.

`check_availability() -> bool` — Pre-flight check for Ollama connectivity and model availability. Useful for GUI/web integrations.

### ViralityScorer

Weighted multi-signal scoring engine. Defined in `yacg/core/virality_scorer.py`.

```python
from yacg.core.virality_scorer import ViralityScorer

scorer = ViralityScorer(weights={...}, config=config)
score = scorer.calculate_score(
    audio=audio_features,
    visual=visual_features,
    semantic=semantic_features,  # Optional — weights redistributed if None
    duration=25.0,
)
# score: ViralityScore (total_score 0-100)
```

When `semantic` is `None`, semantic weights are redistributed proportionally to audio and visual weights. Duration scoring uses an optimal range curve (7-30 seconds).

`duration_score(duration: float) -> float` — Public method exposing the 5-zone duration scoring algorithm (backward-compatible alias). Returns 0-10 score.

---

## Extractors

### ClipExtractor

Extracts and formats video clips. Defined in `yacg/extractors/clip_extractor.py`.

```python
from yacg.extractors.clip_extractor import ClipExtractor

extractor = ClipExtractor(context_padding=2.0, config=config)
success = extractor.extract_clip(
    video_path="/path/to/video.mp4",
    start_time=10.0,
    end_time=40.0,
    output_path="/path/to/output.mp4",
)
```

Adds context padding, applies SmartCropper for 9:16 vertical format, re-encodes with H.264/AAC. Includes 1 retry on failure and validates output is at least 10KB.

Additional batch methods:

| Method | Signature | Description |
|--------|----------|-------------|
| `batch_extract` | `(video_path, segments, output_dir, scores)` | Extract multiple clips in parallel |
| `extract_batch` | `(video_path, clips)` | Backward-compatible wrapper accepting `ClipData` objects |

### YouTubeDownloader

Downloads YouTube videos with H.264 codec preference. Defined in `yacg/youtube_downloader.py`.

```python
from yacg.youtube_downloader import YouTubeDownloader

downloader = YouTubeDownloader(output_dir="./downloads")
result = downloader.download(url="https://youtube.com/watch?v=XXXXX")
# result: dict with keys: video_path, title, duration, channel
```

| Method | Signature | Description |
|--------|----------|-------------|
| `download` | `(url: str) -> dict` | Download video; returns `{video_path, title, duration, channel}` |
| `extract_video_id` | `(url: str) -> str` | Parse video ID from 6 URL formats |

The downloader uses a 5-level format preference cascade to force H.264 codec selection (required for OpenCV compatibility). Progress is logged during download.

### SmartCropper

Face-aware vertical cropping. Defined in `yacg/extractors/smart_cropper.py`.

```python
from yacg.extractors.smart_cropper import SmartCropper

cropper = SmartCropper(config=config)
crop_params = cropper.get_crop_params(
    video_path="/path/to/video.mp4",
    start_time=10.0,
    target_ratio=9/16,
)
# crop_params: dict[str, int] with crop parameters

filter_str = cropper.get_ffmpeg_filter(
    video_path="/path/to/video.mp4",
    start_time=10.0,
    target_ratio=9/16,   # desired width/height ratio (default: 9/16)
    end_time=None,        # when provided, enables multi-frame face sampling
)
# filter_str: "crop=608:1080:656:0" (example)
```

Additional public methods:

```python
# Backward-compatible convenience wrapper
crop_filter = cropper.get_crop_filter(
    video_path="/path/to/video.mp4",
    width=1920,
    height=1080,
)
# crop_filter: "crop=608:1080:656:0" (delegates to get_ffmpeg_filter)

# Detect primary subject center point
x, y = cropper.detect_subject_center(
    video_path="/path/to/video.mp4",
    time_seconds=15.0,
)
# Returns (x, y) center of primary subject using face detection → brightness fallback
```

When `end_time` is provided to `get_crop_params()` or `get_ffmpeg_filter()`, SmartCropper samples 3 frames at 25%, 50%, and 75% of the segment for face detection, uses median positioning for stability, and validates spatial consistency (faces must be within 30% of frame width). This multi-frame sampling prevents crops from jumping when faces move.

Uses DNN SSD face detection (ResNet-10 Caffe model) with Haar cascade fallback, then brightness-center as final fallback. Opens the video file once per `get_crop_params()`/`get_ffmpeg_filter()` call. The `detect_subject_center()` method opens the video independently.

### SubtitleBurner

TikTok word-pop subtitle generation and burning. Defined in `yacg/subtitle_burner.py`.

```python
from yacg.subtitle_burner import SubtitleBurner
from yacg.models import SubtitleStyle

burner = SubtitleBurner()
output_path = burner.process_clip(
    video_path="/path/to/clip.mp4",
    words=[...],         # list[WordTimestamp] for the clip's time range
    width=1080,          # video width in pixels
    height=1920,         # video height in pixels
    style=SubtitleStyle(font_name="Arial", primary_color="&H0000FFFF"),  # optional
)
```

Generates ASS v4+ subtitles with configurable styling and burns them via FFmpeg libass filter. Auto-detects a suitable system font if none specified (`_find_system_font()` probes platform-specific font directories). FFmpeg errors are translated to human-readable messages.

Additional public methods for fine-grained control:

```python
# Generate ASS subtitle content without burning
ass_content = burner.generate_ass(
    words=[...],         # list[WordTimestamp]
    frame_width=1080,
    frame_height=1920,
    style=SubtitleStyle(),
)

# Burn pre-generated ASS content into a video
output = burner.burn_subtitles(
    clip_path="/path/to/clip.mp4",
    ass_content=ass_content,
)
```

Word grouping constants (not configurable): `WORD_GROUP_GAP = 0.2s` (pause triggers new group), `SENTENCE_BREAK_GAP = 0.3s` (minimum display time), `MAX_GROUP_DURATION = 2.0s` (maximum display time). Words appear 1-3 at a time in TikTok-style "word-pop" groups.

### SceneDetector (Legacy)

PySceneDetect-based scene detector. Not used by the default pipeline but available for dependency injection. Defined in `yacg/core/scene_detector.py`.

```python
from yacg.core.scene_detector import SceneDetector

detector = SceneDetector(
    threshold=3.0,
    min_scene_len=7.0,
    max_scene_len=60.0,
)
```

| Method | Description |
|--------|-------------|
| `detect_scenes(video_path)` | Detect scene boundaries using PySceneDetect |
| `merge_short_scenes(scenes)` | Merge scenes shorter than `min_scene_len` |
| `split_long_scenes(scenes)` | Split scenes longer than `max_scene_len` |

---

## Utilities

### Configuration

```python
from yacg.utils.config import load_config, save_default_config

# Load from file (or get defaults)
config = load_config("config.ini")

# Generate a default config file (or use CLI: generate-config)
save_default_config("config.ini")
```

### Video Utilities

Defined in `yacg/utils/video_utils.py`:

| Function | Description |
|----------|-------------|
| `extract_metadata(path)` | Get video metadata via FFprobe. Returns dict with keys: `filename`, `filepath`, `duration` (float), `width` (int), `height` (int), `file_size` (int), `fps` (float), `codec` (str), `audio_codec` (str), `sample_rate` (int), `channels` (int) |
| `extract_audio(video_path, output_path, start=None, end=None)` | Extract audio track to WAV via FFmpeg. Optional `start`/`end` for segment extraction |
| `extract_segment(video, start, end, output)` | Cut a video segment with H.264 re-encode |
| `get_frame_at_time(video, time)` | Capture a single frame as numpy array |
| `ensure_compatible_video(path)` | Transcode to compatible codec if needed |
| `temp_audio_file(suffix=".wav")` | Context manager yielding a temporary file path, cleaned up on exit |
| `translate_ffmpeg_error(stderr)` | Translate raw FFmpeg stderr to human-readable error messages |
| `get_cv2()` | Lazy OpenCV loader shared across modules |

### Bootstrap

```python
from yacg.bootstrap import ensure_ready

# Returns True if all dependencies are satisfied
ready = ensure_ready(verbose=True)
```

Checks FFmpeg/FFprobe availability and auto-installs missing Python packages via pip. Result is cached for 24 hours via `~/.yacg_bootstrap_ok` marker file.
