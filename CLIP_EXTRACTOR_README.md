# Viral Clip Extractor

Automatically detect, score, and extract viral-potential clips from long-form videos. Uses multi-modal AI analysis (audio, visual, semantic via Qwen2.5-VL) with scoring weights optimized for ASMR content on Instagram Reels.

## Overview

The Viral Clip Extractor processes video files through a five-stage pipeline: scene detection, audio analysis, visual analysis, semantic (LLM) analysis, and virality scoring. Top-scoring segments are extracted as standalone clips with optional 9:16 vertical cropping and AI-generated captions. Output includes MP4 clips and a CSV report for integration with content staging tools.

## Prerequisites

- **Python 3.10+**
- **FFmpeg** (with ffprobe) - video processing
- **Ollama** with **Qwen2.5-VL:7b** model - semantic video analysis
- Recommended: 8GB+ RAM, GPU for faster Ollama inference
- Optional: `faster-whisper` for ASMR trigger word detection, `yt-transcriber` for transcript support

## Installation

```bash
# Install Python dependencies
pip install numpy opencv-python-headless librosa requests yt-dlp "scenedetect[opencv]"

# Install and start Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen2.5-vl:7b

# Verify installation
python test_clip_system.py
```

## Quick Start

```bash
# Process a local video
python -m viral_clip_extractor process --video my_video.mp4 --title "My Video"

# Process a YouTube video
python -m viral_clip_extractor youtube --url https://youtube.com/watch?v=XXXXX

# Check system dependencies
python -m viral_clip_extractor check
```

Output clips and CSV report are written to `./clip_output/` by default.

## CLI Reference

### `process` - Process a local video file

```bash
python -m viral_clip_extractor process --video VIDEO_PATH [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | (required) | Path to local video file |
| `--title` | auto-detected | Video title for semantic context |
| `--output-dir` | `./clip_output` | Output directory |
| `--top-n` | `10` | Number of top clips to extract |
| `--min-score` | `70` | Minimum virality score (0-100) |
| `--model` | `qwen2.5-vl:7b` | Ollama model name |
| `--no-captions` | | Disable caption generation |
| `--no-semantic` | | Disable LLM analysis (faster, lower quality) |
| `--no-vertical` | | Keep original aspect ratio |
| `--config` | | Path to config.ini file |
| `-v, --verbose` | | Enable debug logging |

### `youtube` - Process a YouTube video

```bash
python -m viral_clip_extractor youtube --url URL [options]
```

Same options as `process` except `--url` replaces `--video`. Downloads the video via yt-dlp before processing.

### `batch` - Process all videos in a directory

```bash
python -m viral_clip_extractor batch --videos-dir DIRECTORY [options]
```

Processes all `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm` files in the directory. Each video gets a subdirectory in the output.

### `check` - Verify system dependencies

```bash
python -m viral_clip_extractor check
```

Checks FFmpeg, Python packages, Ollama service, and Qwen2.5-VL model availability.

## Configuration

Create a `config.ini` file to customize pipeline behavior. The helper script (`python clip_helper.py`, option 5) can generate a default config for you.

### Config sections

**[Model]** - AI model settings
- `model_name`: Ollama model (default: `qwen2.5-vl:7b`)
- `ollama_host`: Ollama API URL (default: `http://localhost:11434`)

**[SceneDetection]** - Scene boundary detection
- `threshold`: AdaptiveDetector sensitivity, lower = more scenes (default: `3.0`)
- `min_scene_len`: Minimum scene duration in seconds (default: `7.0`)
- `max_scene_len`: Maximum scene duration in seconds (default: `60.0`)

**[ClipSelection]** - Clip filtering
- `top_n_clips`: Maximum clips to extract (default: `10`)
- `min_virality_score`: Score threshold 0-100 (default: `70.0`)

**[Features]** - Feature toggles
- `enable_semantic`: Enable LLM analysis (default: `true`)
- `enable_captions`: Generate captions (default: `true`)
- `vertical_crop`: Apply 9:16 crop (default: `true`)

**[Scoring]** - Virality scoring weights (JSON dict)
- `weights`: Component weights that must sum to ~1.0

Default ASMR-optimized weights:
```json
{
  "hook": 0.20,
  "emotional": 0.15,
  "audio_peaks": 0.15,
  "asmr": 0.12,
  "motion": 0.12,
  "narrative": 0.10,
  "high_freq": 0.10,
  "uniqueness": 0.08,
  "visual": 0.07,
  "duration": 0.05
}
```

To emphasize different content types, adjust the weights. For example, for high-action content, increase `motion` and `hook` while decreasing `asmr`.

## Architecture

```
Input Video
    |
    v
[Scene Detector] -- PySceneDetect AdaptiveDetector
    |                 Merge short scenes, split long ones
    v
[Audio Analyzer] -- librosa: RMS energy, spectral centroid,
    |                 ZCR, onset detection, ASMR patterns
    |
[Visual Analyzer] -- OpenCV: optical flow, face detection,
    |                  color variance, rule-of-thirds
    |
[Semantic Analyzer] -- Qwen2.5-VL via Ollama: emotional
    |                    intensity, narrative, hook, ASMR quality
    v
[Virality Scorer] -- Weighted combination (0-100 scale)
    |
    v
[Clip Extractor] -- FFmpeg: cut, re-encode H.264/AAC
    |                 SmartCropper: face-aware 9:16 crop
    v
[Caption Generator] -- Optional Instagram captions
    |
    v
Output: MP4 clips + CSV report
```

### Module structure

```
viral_clip_extractor/
  models.py              # Shared dataclasses (SceneSegment, AudioFeatures, etc.)
  pipeline.py            # Main pipeline orchestrator
  cli.py                 # CLI entry point (process/youtube/batch/check)
  youtube_downloader.py  # yt-dlp wrapper
  transcript_bridge.py   # yt-transcriber integration
  core/
    scene_detector.py    # PySceneDetect wrapper
    audio_analyzer.py    # librosa-based audio features
    visual_analyzer.py   # OpenCV-based visual features
    semantic_analyzer.py # Qwen2.5-VL semantic analysis
    virality_scorer.py   # Weighted scoring engine
  extractors/
    clip_extractor.py    # FFmpeg clip extraction
    smart_cropper.py     # Face-aware vertical cropping
  utils/
    config.py            # INI config loader
    video_utils.py       # FFmpeg/FFprobe utilities
```

## CSV Output

The `clips_report.csv` contains one row per extracted clip:

| Column | Description |
|--------|-------------|
| `clip_filename` | Output MP4 filename |
| `start_time` | Clip start in source video (seconds) |
| `end_time` | Clip end in source video (seconds) |
| `duration` | Clip duration (seconds) |
| `virality_score` | Composite score 0-100 |
| `hook` | Generated caption hook line |
| `description` | Generated caption body |
| `hashtags` | Generated hashtags |
| `full_caption` | Complete Instagram caption |
| `category` | Content category |
| `audio_peak` | Audio peak energy score |
| `motion_score` | Visual motion intensity |
| `face_presence` | Fraction of frames with faces |
| `asmr_quality` | ASMR trigger quality (0-10, from LLM) |
| `processing_timestamp` | When the clip was processed |

Use this CSV to sort, filter, and stage clips for posting.

## Troubleshooting

**FFmpeg not found**
```bash
sudo apt install ffmpeg   # Ubuntu/Debian
brew install ffmpeg        # macOS
```

**Ollama not running**
```bash
ollama serve    # Start in foreground
# or
setsid ollama serve &   # Start in background
```

**Qwen model not found**
```bash
ollama pull qwen2.5-vl:7b
```

**No scenes detected**
- Lower the `threshold` in config (try `2.0` for subtle ASMR transitions)
- The video may be too short or have no detectable scene changes

**Out of memory during semantic analysis**
- Use `--no-semantic` to skip LLM analysis
- Use a smaller model: `--model qwen2.5-vl:3b`
- Reduce `max_scene_len` to process shorter segments

**yt-dlp download fails**
- Update yt-dlp: `pip install -U yt-dlp`
- Check that the video URL is valid and publicly accessible

**Low virality scores**
- Lower `--min-score` threshold (try `50`)
- Adjust scoring weights in `config.ini` for your content type
- Enable semantic analysis (removes `--no-semantic`)

**Run the full system check:**
```bash
python test_clip_system.py
```
