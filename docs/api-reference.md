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
from yacg.models import PipelineConfig

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
    asmr_mode=True,
    content_type="asmr",
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
| `top_n_clips` | `int` | `10` | Default number of clips to extract |
| `min_virality_score` | `float` | `70.0` | Default minimum virality score |
| `asmr_mode` | `bool` | `True` | Enable ASMR-optimized scoring |
| `content_type` | `str` | `"asmr"` | Content type for LLM prompts: `"asmr"` or `"general"` |
| `num_frames` | `int` | `3` | Number of JPEG frames per segment for VLM analysis (1-5) |
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
    audio_peak_score: float    # 0-1, 90th percentile RMS energy
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

#### `process_video()`

Process a local video file through the full 10-step pipeline.

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
| `top_n` | `int` | `10` | Number of top clips to extract |
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
| `top_n` | `int` | `10` | Number of top clips to extract |
| `min_score` | `float` | `70.0` | Minimum virality score threshold |

**Returns:** `ProcessingResult`

---

## Protocol Interfaces

The pipeline defines 8 `Protocol` interfaces in `pipeline.py` for dependency injection. All are `@runtime_checkable`.

### TranscriptSegmenterProtocol

```python
class TranscriptSegmenterProtocol(Protocol):
    def full_transcribe(self, video_path: str) -> list[WordTimestamp]: ...
    def segment_by_content(self, words: list[WordTimestamp], title: str) -> list[SegmentBoundary]: ...
    def refine_boundaries(self, boundaries: list[SegmentBoundary], words: list[WordTimestamp]) -> list[SceneSegment]: ...
```

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
    def analyze_video(self, video_path: str, title: str) -> CaptionData: ...
```

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

### SemanticAnalyzer

Semantic analysis via Ollama VLM. Defined in `yacg/core/semantic_analyzer.py`.

```python
from yacg.core.semantic_analyzer import SemanticAnalyzer

analyzer = SemanticAnalyzer(
    model="qwen2.5-vl:7b",
    ollama_host="http://localhost:11434",
    num_frames=3,            # 1-5 JPEG frames per segment
    content_type="asmr",     # "asmr" or "general" — controls prompt language
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
)
# filter_str: "crop=608:1080:656:0" (example)
```

Uses DNN SSD face detection (ResNet-10 Caffe model) with Haar cascade fallback, then brightness-center as final fallback. Opens the video file only once per crop operation.

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
| `extract_metadata(path)` | Get duration, dimensions, codec info via FFprobe |
| `extract_audio(video, output)` | Extract audio track to WAV via FFmpeg |
| `extract_segment(video, start, end, output)` | Cut a video segment with H.264 re-encode |
| `get_frame_at_time(video, time)` | Capture a single frame as numpy array |
| `ensure_compatible_video(path)` | Transcode to compatible codec if needed |
| `temp_audio_file(video, start, end)` | Context manager for temporary audio extraction |
| `translate_ffmpeg_error(stderr)` | Translate raw FFmpeg stderr to human-readable error messages |
| `get_cv2()` | Lazy OpenCV loader shared across modules |

### Bootstrap

```python
from yacg.bootstrap import ensure_ready

# Returns True if all dependencies are satisfied
ready = ensure_ready(verbose=True)
```

Checks FFmpeg/FFprobe availability and auto-installs missing Python packages via pip. Result is cached for 24 hours via `~/.yacg_bootstrap_ok` marker file.
