# CLI Reference

YACG is invoked via command line:

```bash
yacg <command> [options]
# or
python -m yacg <command> [options]
```

The `yacg` command is available after `pip install -e .` (or `pip install .`).

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
- `https://youtube.com/shorts/XXXXX`
- `https://youtube.com/embed/XXXXX`
- `https://youtube.com/v/XXXXX` (legacy format)

**Example:**

```bash
python -m yacg youtube \
  --url "https://youtube.com/watch?v=dQw4w9WgXcQ" \
  --top-n 3
```

YouTube videos are downloaded to a `downloads/` subdirectory under `--output-dir`. YACG forces H.264 codec selection during download because OpenCV cannot decode AV1/VP9. If H.264 is unavailable at the requested quality, the video may be automatically transcoded after download.

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

In batch mode, the Whisper model is loaded once and shared across all videos, saving 5-15 seconds per video compared to processing them individually. Video titles are auto-generated from filenames by replacing underscores and hyphens with spaces and title-casing (e.g., `my_cool_video.mp4` becomes "My Cool Video"). Use the `process` command with `--title` for precise control over titles.

In the batch summary, a video with ANY errors counts as "had errors" even if it produced clips (partial success). Exit code 2 indicates at least one video had partial success.

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

!!! note "Ollama Check Limitation"
    The `check` command always tests Ollama connectivity at `http://localhost:11434`, regardless of the `--ollama-host` setting. Users with remote Ollama servers should verify connectivity manually (e.g., `curl http://your-host:11434/api/tags`).

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
  qwen2.5:7b (segmentation): OK
  Whisper model 'small': CACHED
  DNN face detection: OK (models in ~/.yacg/models)

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

`generate-config` exits with code 1 if the output file already exists — it will not overwrite. Delete or rename the existing file first.

---

## Common Options

These flags are shared across `process`, `youtube`, and `batch`:

### Core Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `./clip_output` | Where clips and reports are written |
| `--top-n` | `20` | Maximum number of clips to extract |
| `--min-score` | `70` | Clips below this virality score (0-100) are discarded. Score ranges: below 50 = rarely viral, 60-70 = moderate, 70-85 = strong, 85+ = exceptional |
| `--config` | *(none)* | INI config file; CLI flags override config values |
| `-v` / `--verbose` | off | Show debug-level log output. Also reveals warnings suppressed by default: pynvml FutureWarning (GPU detection), librosa audioread FutureWarning, PySoundFile UserWarning |
| `--dry-run` | off | Run analysis only (steps 1-4) — no extraction, subtitles, or captions |

### Model Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `qwen2.5-vl:7b` | Ollama vision model for semantic analysis and captions |
| `--whisper-model` | `small` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`). Larger = more accurate but slower/more RAM |
| `--segmentation-model` | `qwen2.5:7b` | Text-only Ollama model for transcript segmentation. Avoids loading the vision encoder. Uses the PipelineConfig default (`qwen2.5:7b`) when omitted from the CLI. Falls back to `--model` only when explicitly set to an empty string in an INI config file (not reachable from CLI alone) |
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
| `--num-frames` | `3` | Number of JPEG frames per segment for VLM semantic analysis (1-5). Set to `1` if your VLM produces garbled output with multiple images. Affects semantic analysis only — caption generation always uses 3 frames regardless of this setting |

### Scene Detection Options (Legacy)

!!! warning "Legacy Options"
    These parameters configure the legacy PySceneDetect-based scene detector. The current pipeline uses LLM-driven transcript segmentation instead. These options are accepted for backward compatibility but have no effect on the default pipeline.

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
| `--asmr-mode` / `--no-asmr-mode` | off | Enable/disable ASMR-optimized scoring. Auto-enabled when `--content-type asmr` is set |
| `--content-type` | `general` | Content type preset for LLM prompts. Options: `general`, `gaming`, `cooking`, `asmr`, `educational`, `fitness`, `comedy`, `music`, `beauty`, `tech`, `vlog`. Loads a preset that sets tone, audience, platform, and caption defaults (overridable individually) |
| `--channel-description` | *(empty)* | Channel/creator description for caption context |
| `--target-audience` | *(empty)* | Target audience description for tone adjustment |
| `--tone` | `engaging` | Caption tone: `energetic`, `calm`, `professional`, `casual`, `humorous`, `inspirational`, `dramatic`, `engaging` |
| `--platform` | `all` | Target platform: `tiktok`, `reels`, `shorts`, `all` |
| `--caption-length` | `medium` | Caption length: `short`, `medium`, `long` |
| `--hashtag-count` | `5` | Number of hashtags per caption (3-7) |
| `--custom-instructions` | *(empty)* | Custom instructions appended to LLM prompts |

### Subtitle Styling Options

| Flag | Default | Description |
|------|---------|-------------|
| `--subtitle-font` | *(auto-detected)* | Font name for subtitles. When empty, auto-detects by probing in order: Liberation Sans, Arial, Helvetica Neue, Helvetica, DejaVu Sans, Noto Sans, sans-serif. Uses `fc-list` on Linux/macOS; assumes Arial on Windows. In minimal environments (Docker), install at least one of these fonts (e.g., `apt install fonts-liberation`) |
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
| `1` | Failure — missing dependencies, invalid arguments, all videos failed, no subcommand given, invalid YouTube URL, no video files found in batch directory, config file already exists (generate-config), `--min-score` outside 0-100 range, or bootstrap dependency check failure |
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

The title slug is generated by lowercasing, replacing non-alphanumeric characters with hyphens, and truncating to 40 characters. The `clips_report.csv` is overwritten (not appended) on each run — rename or copy previous reports if needed.

All clip outputs are encoded with libx264, CRF 23, `fast` preset, AAC 128k audio, and `+faststart` flag. Subtitle burning re-encodes video with the same settings but copies the audio stream unchanged. These encoding settings are hardcoded and not configurable via CLI or config file.

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
