# Architecture

## Overview

YACG is a 7-step pipeline that takes a video file, segments it using LLM-driven transcript analysis, scores segments across audio/visual/semantic dimensions, extracts the top clips in parallel as vertical (9:16) videos with burned subtitles and thumbnails, and generates Instagram-optimized captions.

The pipeline uses **protocol-based dependency injection** — all 8 core components are defined as `Protocol` interfaces and can be swapped via constructor injection. Components are lazily initialized to keep import time fast.

## Pipeline Flow

```
                          ┌─────────────────────────┐
                          │     CLI (cli.py)          │
                          │  process / youtube        │
                          │  batch / check            │
                          │  show-config              │
                          │  generate-config          │
                          └─────────┬─────────────────┘
                                    │
                          ┌─────────▼─────────────────┐
                          │ bootstrap.py               │
                          │ ensure_ready()             │
                          │ ffmpeg + pip deps           │
                          │ (cached for 24h)            │
                          └─────────┬─────────────────┘
                                    │
                ┌───────────────────▼─────────────────────┐
                │    ViralClipPipeline (pipeline.py)       │
                │  Orchestrates the 7-step workflow          │
                │  8 Protocol-based injectable components   │
                │  progress_callback for GUI/web embedding  │
                └───┬───┬───┬───┬───┬───┬───┬───┬───┬────┘
                    │   │   │   │   │   │   │   │   │
   ┌────────────────┘   │   │   │   │   │   │   │   └────────┐
   ▼                    ▼   │   │   │   │   │   ▼            ▼
┌──────────┐    ┌──────────┐│   │   │   │   │ ┌──────────┐ ┌──────────┐
│Transcript│    │  Audio   ││   │   │   │   │ │ Caption  │ │   CSV    │
│Segmenter │    │ Analyzer ││   │   │   │   │ │Generator │ │  Report  │
│(step 1-2)│    │ (step 3) ││   │   │   │   │ │(step 7)  │ │(private) │
└──────────┘    └──────────┘│   │   │   │   │ └──────────┘ └──────────┘
                            │   │   │   │   │
                   ┌────────┘   │   │   │   └────────┐
                   ▼            ▼   ▼   ▼            ▼
             ┌──────────┐ ┌──────────┐  ┌──────────────┐
             │  Visual   │ │Virality  │  │    Clip      │
             │ Analyzer  │ │ Scorer   │  │  Extractor   │
             │ (step 3)  │ │(step 3-4)│  │  (step 5)    │
             └──────────┘ └──────────┘  └──────┬───────┘
                                               │
                                        ┌──────▼───────┐
                                        │SmartCropper   │
                                        │(DNN SSD face  │
                                        │ detection +   │
                                        │ Haar fallback)│
                                        └───────────────┘
                                               │
                                        ┌──────▼───────┐
                                        │SubtitleBurner │
                                        │ (step 6)      │
                                        │ ASS → FFmpeg  │
                                        │ Configurable  │
                                        │ SubtitleStyle │
                                        └───────────────┘

External Dependencies:
  ┌──────────┐  ┌─────────────┐  ┌──────────┐  ┌─────────┐
  │ FFmpeg / │  │faster-whisper│  │  Ollama  │  │  yt-dlp │
  │ FFprobe  │  │(auto device) │  │ (VLM API)│  │         │
  └──────────┘  └─────────────┘  └──────────┘  └─────────┘
```

## Pipeline Steps

| Step | Label | Component | What Happens |
|------|-------|-----------|--------------|
| 1/7 | Transcribe | `TranscriptSegmenter.full_transcribe()` | faster-whisper produces word-level timestamps from the full video (auto device/precision) |
| 2/7 | Segment | `TranscriptSegmenter.segment_by_content()` + `refine_boundaries()` | Transcript sent to Ollama text model (`qwen2.5:7b`) for segmentation; boundaries snapped to speech pauses |
| 3/7 | Analyze | `AudioAnalyzer` + `VisualAnalyzer` + `SemanticAnalyzer` + `ViralityScorer` | Each segment scored on audio, visual, and semantic signals (multi-frame VLM analysis) |
| 4/7 | Rank | Sort + filter | Segments sorted by virality score, filtered by threshold, top N selected |
| | | | *(dry-run exits here)* |
| 5/7 | Extract | `ClipExtractor` + `SmartCropper` | FFmpeg cuts clips **in parallel** (up to 4 workers) to `.yacg_tmp/` staging dir, applies face-aware 9:16 crop |
| 6/7 | Subtitles | `SubtitleBurner` | Word-pop ASS subtitles generated and burned **in parallel** via FFmpeg libass filter. **Failure deletes the clip** (mandatory step) |
| — | Thumbnails | FFmpeg | Midpoint JPEG thumbnail extracted for each clip (unlabeled in CLI output) |
| 7/7 | Captions | `OllamaVideoAnalyzer` | 3 JPEG frames (at 25%, 50%, 75% of clip) sent to Ollama VLM for Instagram caption generation. **Failure deletes clip and thumbnail** (mandatory step) |
| — | Stage | File move | Clips and thumbnails moved from `.yacg_tmp/` staging dir to final output dir. Staging dir cleaned up |
| — | Report | CSV write | `clips_report.csv` written with 17 columns of metadata |

## Data Flow

```
Video file
  │
  ├── [codec compat check: ensure_compatible_video()]
  │
  ▼
full_transcribe() → list[WordTimestamp]
  │
  ▼
segment_by_content() → list[SegmentBoundary]
  │
  ▼
refine_boundaries() → list[SceneSegment]
  │
  ▼ (per segment)
_analyze_segment() → ClipData
  ├── AudioAnalyzer.analyze_segment() → AudioFeatures
  ├── VisualAnalyzer.analyze_segment() → VisualFeatures
  ├── SemanticAnalyzer.analyze_segment() → SemanticFeatures (1-5 JPEG frames)
  └── ViralityScorer.calculate_score() → ViralityScore
  │
  ▼
sort + filter → list[ClipData] (top N)
  │
  ▼ (per clip, parallel — ThreadPoolExecutor, min(4, num_clips) workers, 600s batch timeout)
ClipExtractor.extract_clip() → MP4 file (9:16) in .yacg_tmp/ staging
  │
  ▼ (per clip, parallel — ThreadPoolExecutor, min(4, num_clips) workers, 600s batch timeout)
SubtitleBurner.process_clip(words, width, height, style) → MP4 with burned subtitles
  │
  ▼ (per clip)
Thumbnail extraction → JPEG thumbnail
  │
  ▼ (per clip, sequential)
OllamaVideoAnalyzer.analyze_video() → CaptionData
  │
  ▼
Move from staging (.yacg_tmp/) → output dir
  │
  ▼
ProcessingResult + clips_report.csv
```

### Key Data Flow Patterns

1. **Single-transcription reuse**: Whisper runs once on the full video. The resulting `list[WordTimestamp]` is reused for: LLM segmentation input, boundary refinement/snap-to-pause, subtitle word filtering per clip, and audio analyzer trigger word detection.

2. **Word filtering for subtitles**: Words are filtered to each clip's time range with 50ms boundary tolerance, then timestamps are re-zeroed relative to the clip start.

3. **Staging directory**: Clips are extracted to `.yacg_tmp/` staging dir, then moved to the final output directory only after subtitles, thumbnails, and captions all succeed. Failed clips are deleted in-place. Staging dir is cleaned up in a `finally` block.

4. **Mandatory post-processing**: Both subtitle burning and caption generation are mandatory. Failure in either deletes the clip file (and thumbnail if present), preventing un-subtitled or un-captioned clips from reaching the output.

## Protocol-Based Dependency Injection

The pipeline defines 8 `Protocol` interfaces (all `@runtime_checkable`) for full component substitution:

| Protocol | Methods | Default Implementation |
|----------|---------|----------------------|
| `TranscriptSegmenterProtocol` | `full_transcribe`, `segment_by_content`, `refine_boundaries` | `TranscriptSegmenter` |
| `SubtitleBurnerProtocol` | `get_video_dimensions`, `process_clip` | `SubtitleBurner` |
| `SemanticAnalyzerProtocol` | `analyze_segment` | `SemanticAnalyzer` |
| `CaptionGeneratorProtocol` | `analyze_video` | `OllamaVideoAnalyzer` |
| `AudioAnalyzerProtocol` | `analyze_segment` | `AudioAnalyzer` |
| `VisualAnalyzerProtocol` | `analyze_segment` | `VisualAnalyzer` |
| `ViralityScorerProtocol` | `calculate_score` | `ViralityScorer` |
| `ClipExtractorProtocol` | `extract_clip` | `ClipExtractor` |

```python
pipeline = ViralClipPipeline(
    config=config,
    transcript_segmenter=custom_segmenter,  # inject custom implementation
    audio_analyzer=mock_analyzer,
    progress_callback=my_callback,          # (step_name, current, total) → None
)
```

## Configuration Architecture

Configuration flows through three layers, each overriding the previous:

```
Layer 1: PipelineConfig defaults (models.py)
  └── All 23 fields have defaults. Scoring weights validated in __post_init__.

Layer 2: INI file overrides (utils/config.py)
  └── configparser reads [Model], [SceneDetection], [Segmentation], [ClipSelection],
      [ASMR Optimization], [ContentProfile], [Output], [Temporal], [Subtitle], [Scoring] sections.
      [ContentProfile] is preferred over [ASMR Optimization] for content_type.
      The [ASMR Optimization] section is accepted as a legacy alias for [ContentProfile] for the
      content_type key. A log message recommends renaming to [ContentProfile].

Layer 3: CLI argument overrides (cli.py)
  └── 40+ argparse flags override INI values. CLI flags always win.
```

Note: `top_n_clips` and `min_virality_score` are config-level fields that are effectively unused. `process_video()` defaults `top_n=20` and `min_score=70.0` via method parameters, and the CLI always passes its own defaults. No code path reads `config.top_n_clips` or `config.min_virality_score`. To control clip selection, pass `top_n` and `min_score` directly to `process_video()` or use CLI flags `--top-n` and `--min-score`.

## Package Structure

```
yacg/
    __init__.py              # Exports: ClipData, PipelineConfig, ViralClipPipeline, __version__
    __main__.py              # python -m entry point
    models.py                # Dataclasses: SceneSegment, AudioFeatures, VisualFeatures,
                             #   SemanticFeatures, ViralityScore, WordTimestamp, SegmentBoundary,
                             #   CaptionData, ClipData, SubtitleStyle, ContentProfile,
                             #   PipelineConfig, ProcessingResult
                             #   Constants: VALID_SCORING_KEYS, VALID_CONTENT_TYPES,
                             #   VALID_TONES, VALID_PLATFORMS, VALID_CAPTION_LENGTHS,
                             #   CONTENT_PRESETS
    cli.py                   # argparse CLI (6 subcommands: process, youtube, batch, check,
                             #   show-config, generate-config)
    bootstrap.py             # Dependency checking/auto-install (24h cache)
    pipeline.py              # Main orchestrator (ViralClipPipeline), 8 Protocol interfaces
    caption_generator.py     # Ollama VLM → Instagram captions (multi-frame, content-type-aware)
    subtitle_burner.py       # ASS subtitle generation + FFmpeg burn (configurable SubtitleStyle,
                             #   platform-aware font auto-detection)
    transcript_segmenter.py  # faster-whisper + Ollama segmentation (configurable device/compute)
    youtube_downloader.py    # yt-dlp wrapper (H.264 preference, progress logging)
    core/
        audio_analyzer.py    # librosa-based audio features (tapping, crinkle, mouth-sound detection)
        scene_detector.py    # PySceneDetect (legacy, unused — replaced by LLM segmentation)
        semantic_analyzer.py # Qwen2.5-VL via Ollama (multi-frame, content-type-aware prompts)
        virality_scorer.py   # Weighted multi-signal scorer (duration curve, semantic redistribution)
        visual_analyzer.py   # OpenCV optical flow, DNN SSD face detection (Haar cascade fallback)
    extractors/
        clip_extractor.py    # FFmpeg clip cutting (1 retry, 10KB output validation)
        smart_cropper.py     # Face-aware 9:16 vertical crop (DNN SSD → Haar → brightness fallback)
    utils/
        config.py            # INI config file loader (~33 keys across 10 sections)
        video_utils.py       # FFmpeg/FFprobe wrappers, translate_ffmpeg_error(), get_cv2()
```

## Module Dependencies

- **`models.py`** is the leaf node — every other module imports from it, it imports nothing from the package.
- **`utils/video_utils.py`** is the shared FFmpeg/OpenCV utility layer — used by audio, semantic, clip, subtitle, and caption modules.
- **`pipeline.py`** is the sole orchestrator — it imports all components but nothing imports it except `cli.py` and `__init__.py`.
- All heavy dependencies (OpenCV, librosa, faster-whisper) are **lazy-imported** behind accessor guards to keep import time fast.

## External Systems

### Ollama

YACG communicates with Ollama via HTTP (`POST /api/generate`). Three modules call Ollama, all using `requests.Session()` for HTTP connection reuse:

| Module | Purpose | Input | Model |
|--------|---------|-------|-------|
| `TranscriptSegmenter` | Segment identification from transcript | Text only | `segmentation_model` (default: `qwen2.5:7b`, text-only) |
| `SemanticAnalyzer` | Semantic scoring of video segments | Text + 1-5 JPEG frames (configurable via `num_frames`) | `model_name` (default: `qwen2.5-vl:7b`) |
| `OllamaVideoAnalyzer` | Instagram caption generation | Text + transcript + 3 JPEG frames (25%, 50%, 75%) — frame count hardcoded, not affected by `num_frames` config | `model_name` (default: `qwen2.5-vl:7b`) |

Retry behavior: all three use 3 retries with exponential backoff. Default host: `http://localhost:11434` (configurable via `--ollama-host`).

**LLM temperatures:** `SemanticAnalyzer` uses `temperature=0.3` for deterministic, consistent scoring. `OllamaVideoAnalyzer` uses `temperature=0.7` for creative, varied caption generation. Neither is configurable via CLI or config. This explains why semantic scores are more consistent than captions across runs of the same video.

### faster-whisper

Runs full-video transcription with configurable device and precision. Model cached across batch runs.

- Default model: `small` (configurable: `tiny`, `base`, `small`, `medium`, `large-v3`)
- Device: `"auto"` — uses CUDA if available, else CPU (configurable via `--whisper-device`)
- Compute type: `"auto"` — float16 on CUDA, int8 on CPU (configurable via `--whisper-compute-type`)
- VAD filter: configurable (`--vad-filter` / `--no-vad-filter`)

### FFmpeg / FFprobe

Used for video metadata extraction, audio extraction, segment cutting, smart cropping, subtitle burning, thumbnail extraction, and codec compatibility checks. Both must be on `PATH`. FFmpeg errors are translated to human-readable messages via `translate_ffmpeg_error()`.

### yt-dlp

Downloads YouTube videos with H.264 codec preference via `YouTubeDownloader.download()`. Progress is logged during download.

### Hardcoded Timeouts

All timeout values are hardcoded and not configurable:

| Operation | Timeout | Source File |
|-----------|---------|------------|
| FFprobe metadata extraction | 30s | `video_utils.py` |
| Audio extraction (FFmpeg) | 120s | `video_utils.py` |
| Semantic analysis (Ollama) | 120s | `semantic_analyzer.py` |
| Caption generation (Ollama) | 120s | `caption_generator.py` |
| Transcript segmentation (Ollama) | 180s | `transcript_segmenter.py` |
| Segment extraction (FFmpeg) | 300s | `video_utils.py` |
| Clip extraction (FFmpeg) | 300s | `clip_extractor.py` |
| Subtitle burning (FFmpeg) | 300s | `subtitle_burner.py` |
| ThreadPoolExecutor batch | 600s | `pipeline.py` |
| Video transcoding (FFmpeg) | 3600s | `video_utils.py` |

## Error Handling

- **Fatal errors** (no transcription, no segments): Pipeline returns early with errors in `ProcessingResult.errors`
- **Per-segment errors**: Caught individually, logged, pipeline continues with remaining segments
- **Per-clip mandatory errors** (subtitles, captions): Clip AND thumbnail deleted, error recorded — un-subtitled or un-captioned clips never reach output
- **Per-clip non-fatal errors** (thumbnail generation, clip extraction): Error recorded, pipeline continues
- **Ollama failures**: 3 retries with exponential backoff; `RuntimeError` after exhaustion
- **FFmpeg failures**: 1 retry on clip extraction failure; errors translated to human-readable messages
- **CSV report failure**: Caught, error appended to `ProcessingResult.errors`
- **Clip size validation**: Extracted clips smaller than 10KB (10,240 bytes) are detected as potentially corrupt and automatically deleted
- **Semantic analysis frame fallback**: If all multi-frame extractions fail, `SemanticAnalyzer` falls back to a single frame at the segment midpoint
- **Error summary**: Errors are printed as a highlighted block at the end of processing

All non-fatal errors accumulate in `ProcessingResult.errors` for post-run inspection.

### Degraded-Mode Scoring

When Ollama/VLM semantic analysis is unavailable (returns `None`), the `ViralityScorer` adapts:

- Semantic weights are redistributed proportionally across audio and visual components
- Confidence drops from 1.0 to 0.5
- Clips scored without VLM analysis will have different weight distributions and lower confidence values

When the DNN face detection model is missing, face detection falls back to the Haar cascade (lower accuracy for side profiles and diverse skin tones). Each ASMR audio sub-detector (tapping, crinkle, mouth-sound) has independent error handling — failure of one doesn't affect others.

**ASMR audio sub-detector boost values:**

| Detector | Measurement | Boost Target | Max Boost |
|----------|------------|--------------|-----------|
| Tapping | Onset transient density | `audio_peak_score` | +0.1 |
| Crinkle | High-frequency spectral flux | `high_freq_score` | +0.05 |
| Mouth-sound | Mid-frequency (1-4kHz) energy ratio | `zcr_score` | +0.02 |

**Trigger word lists** (not configurable):

- **ASMR content:** `tingles`, `relax`, `sleep`, `cozy`, `gentle`, `dragon`, `scales`, `whisper`, `magic`
- **Non-ASMR content:** `amazing`, `incredible`, `wow`, `unbelievable`, `shocking`, `secret`, `hack`, `trick`, `tip`

These affect the `trigger_words` field in `AudioFeatures` and vary based on content type.

### Exit Codes

- `0`: No errors
- `1`: Total failure (errors and zero clips produced)
- `2`: Partial success (some clips produced but with errors — applies to all commands)

## Lazy Initialization

`ViralClipPipeline` lazily initializes all 8 component objects. Each has a `_get_*()` accessor that imports and constructs on first call. This avoids loading heavy dependencies (OpenCV, librosa, whisper) for lightweight commands like `check` or `--help`. Components injected via constructor skip lazy initialization.

## Bootstrap Caching

The dependency check result (`bootstrap.ensure_ready()`) is cached via a `~/.yacg_bootstrap_ok` marker file with a 24-hour TTL, avoiding the startup penalty on every invocation.
