# CLI Reference

YACG is invoked as a Python module:

```bash
python -m yacg <command> [options]
```

## Commands

### `process` — Process a Local Video

Extract clips from a local video file.

```bash
python -m yacg process --video <path> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | *(required)* | Path to local video file |
| `--title` | *(auto-detected)* | Video title for caption generation |

All [common options](#common-options) are also accepted.

**Example:**

```bash
python -m yacg process \
  --video ~/Videos/stream_2026-03-01.mp4 \
  --title "March Stream Highlights" \
  --top-n 5 \
  --min-score 60 \
  --whisper-model small
```

---

### `youtube` — Process a YouTube Video

Download a YouTube video and extract clips.

```bash
python -m yacg youtube --url <url> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--url` | *(required)* | YouTube video URL |

All [common options](#common-options) are also accepted.

Supported URL formats include:

- `https://youtube.com/watch?v=XXXXX`
- `https://youtu.be/XXXXX`
- `https://www.youtube.com/watch?v=XXXXX`

**Example:**

```bash
python -m yacg youtube \
  --url "https://youtube.com/watch?v=dQw4w9WgXcQ" \
  --top-n 3
```

---

### `batch` — Process a Directory of Videos

Process all video files in a directory. The Whisper model is loaded once and shared across all videos.

```bash
python -m yacg batch --videos-dir <path> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--videos-dir` | *(required)* | Directory containing video files |

All [common options](#common-options) are also accepted. Each video gets its own subdirectory under `--output-dir`.

Recognized video extensions: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`.

**Example:**

```bash
python -m yacg batch \
  --videos-dir ~/Videos/ASMR/ \
  --output-dir ~/Clips/ \
  --min-score 50
```

---

### `check` — Verify Dependencies

Check that all required system dependencies are installed.

```bash
python -m yacg check
```

No additional flags. Checks:

- FFmpeg and FFprobe availability
- Python packages: numpy, opencv, librosa, requests, yt-dlp, scenedetect, faster-whisper
- Ollama service connectivity
- `qwen2.5-vl:7b` vision model availability
- `qwen2.5:7b` segmentation model availability
- Whisper model cache status (whether the model needs to be downloaded on first run)
- DNN face detection model availability (optional — Caffe SSD model)

**Example output:**

```
  ffmpeg: OK (ffmpeg version 6.1.1)
  ffprobe: OK
  numpy: OK
  opencv-python-headless: OK
  librosa: OK
  requests: OK
  yt-dlp: OK
  scenedetect[opencv]: OK
  faster-whisper: OK
  Ollama: OK (3 models loaded)
  qwen2.5-vl:7b: OK
  qwen2.5:7b: OK
  Whisper model 'small': cached
  DNN face detection: OK (Caffe SSD model found)

All required dependencies satisfied.
```

---

### `show-config` — Print Current Configuration

Display the current configuration with all field descriptions and defaults.

```bash
python -m yacg show-config [--config <path>]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | *(none)* | Path to config INI file (shows defaults if omitted) |

**Example:**

```bash
python -m yacg show-config
python -m yacg show-config --config my_config.ini
```

---

### `generate-config` — Generate Default Config File

Write a fully-commented default INI config file with all available keys.

```bash
python -m yacg generate-config [--output <path>]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `config.ini` | Output path for the generated config file |

**Example:**

```bash
python -m yacg generate-config
python -m yacg generate-config --output my_config.ini
```

---

## Common Options

These flags are shared across `process`, `youtube`, and `batch`:

### Core Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `./clip_output` | Where clips and reports are written |
| `--top-n` | `10` | Maximum number of clips to extract |
| `--min-score` | `70` | Clips below this virality score (0-100) are discarded |
| `--config` | *(none)* | INI config file; CLI flags override config values |
| `-v` / `--verbose` | off | Show debug-level log output |
| `--dry-run` | off | Run analysis only (steps 1-4), skip extraction |

### Model Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `qwen2.5-vl:7b` | Ollama vision model for semantic analysis and captions |
| `--whisper-model` | `small` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`). Larger = more accurate but slower/more RAM |
| `--segmentation-model` | *(uses `--model`)* | Text-only Ollama model for transcript segmentation. Default `qwen2.5:7b` avoids loading the vision encoder |
| `--ollama-host` | `http://localhost:11434` | Ollama API base URL |
| `--scoring-weights` | *(none)* | JSON dict of scoring weight overrides (e.g., `'{"hook": 0.3}'`) |

### Whisper Hardware Options

| Flag | Default | Description |
|------|---------|-------------|
| `--whisper-device` | `auto` | Whisper inference device: `auto` (CUDA if available), `cpu`, or `cuda` |
| `--whisper-compute-type` | `auto` | Whisper precision: `auto` (float16 on CUDA, int8 on CPU), `int8`, `float16`, `float32` |

### VLM Analysis Options

| Flag | Default | Description |
|------|---------|-------------|
| `--num-frames` | `3` | Number of JPEG frames per segment for VLM analysis (1-5) |

### Scene Detection Options

| Flag | Default | Description |
|------|---------|-------------|
| `--scene-threshold` | `3.0` | Scene detection threshold |
| `--min-scene-len` | `7.0` | Minimum scene length in seconds |
| `--max-scene-len` | `60.0` | Maximum scene length in seconds |

### Segmentation Tuning Options

| Flag | Default | Description |
|------|---------|-------------|
| `--pause-threshold` | `0.3` | Speech pause duration (seconds) that triggers a segment boundary |
| `--min-segment-duration` | `15.0` | Minimum segment duration in seconds |
| `--max-segment-duration` | `45.0` | Maximum segment duration in seconds |
| `--vad-filter` / `--no-vad-filter` | on | Voice Activity Detection filter for Whisper transcription |
| `--context-padding` | `2.0` | Seconds of padding added around clip boundaries |

### Content Type Options

| Flag | Default | Description |
|------|---------|-------------|
| `--asmr-mode` / `--no-asmr-mode` | on | Enable/disable ASMR-optimized scoring |
| `--content-type` | `asmr` | Content type for LLM prompts: `asmr` or `general` |

### Subtitle Styling Options

| Flag | Default | Description |
|------|---------|-------------|
| `--subtitle-font` | *(auto-detected)* | Font name for subtitles. Auto-detects a suitable system font if empty |
| `--subtitle-font-size` | `0.055` | Font size as fraction of video height (e.g., 0.055 = 5.5%) |
| `--subtitle-color` | `&H00FFFFFF` | Primary subtitle color in ASS `&HAABBGGRR` format (default: white) |
| `--subtitle-outline-color` | `&H00000000` | Outline color in ASS format (default: black) |
| `--subtitle-outline-width` | `3.0` | Outline width in pixels |
| `--subtitle-shadow` | `1.5` | Shadow depth in pixels |
| `--subtitle-margin-v` | `0.38` | Vertical margin as fraction of video height (default: 38%) |
| `--subtitle-margin-h` | `0.15` | Horizontal margin as fraction of video width (default: 15%) |

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success — all videos processed without errors |
| `1` | Failure — missing dependencies, invalid arguments, or all videos failed |
| `2` | Partial success — clips were produced but some errors occurred (applies to all commands, not just `batch`) |

## Output Structure

After processing, the output directory contains:

```
clip_output/
  my_video_clip_01_score85.mp4        # Vertical clip with burned subtitles
  my_video_clip_01_score85_thumb.jpg  # Midpoint JPEG thumbnail
  my_video_clip_02_score72.mp4
  my_video_clip_02_score72_thumb.jpg
  clips_report.csv                    # Metadata for all clips
```

Clip filenames follow the pattern `{title_slug}_clip_{NN}_score{score}.mp4`. If no title is provided, the `{title_slug}_` prefix is omitted.

### CSV Report Columns

The `clips_report.csv` contains 17 columns:

| Column | Type | Description |
|--------|------|-------------|
| `Clip_Filename` | str | Output filename |
| `Start_Time` | float | Segment start time (seconds, 2 decimal places) |
| `End_Time` | float | Segment end time (seconds) |
| `Duration` | float | Clip duration (seconds) |
| `Virality_Score` | float | Composite virality score (0-100) |
| `Hook` | str | Caption hook line |
| `Description` | str | Caption description |
| `Hashtags` | str | Space-joined hashtags |
| `Full_Caption` | str | Assembled Instagram-ready caption |
| `Category` | str | Content category |
| `Audio_Peak` | float | 90th percentile RMS energy |
| `Motion_Score` | float | Optical flow magnitude |
| `Face_Presence` | float | Face detection ratio |
| `Zero_Crossing_Rate` | float | ZCR score |
| `Composition` | float | Rule-of-thirds score |
| `ASMR_Quality` | float | Semantic ASMR quality score |
| `Processing_Timestamp` | ISO 8601 | Processing run timestamp |
