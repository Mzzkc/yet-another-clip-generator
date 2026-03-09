# Getting Started

## System Requirements

- **OS:** Linux, macOS, or WSL2 on Windows
- **Python:** 3.10 or newer
- **RAM:** 8GB minimum, 16GB recommended for the 7B model
- **Storage:** ~10GB for default model weights (qwen2.5-vl:7b + qwen2.5:7b + Whisper small) + video storage
- **GPU:** Optional but recommended for Ollama inference and Whisper transcription (`--whisper-device cuda`)

## Installation

### 1. Install System Dependencies

=== "Ubuntu / Debian"

    ```bash
    sudo apt update
    sudo apt install ffmpeg python3 python3-pip python3-venv
    ```

=== "macOS"

    ```bash
    brew install ffmpeg python3
    ```

=== "Windows (WSL2)"

    ```bash
    # Install WSL2 with Ubuntu first, then:
    sudo apt update
    sudo apt install ffmpeg python3 python3-pip python3-venv
    ```

Verify FFmpeg is installed:

```bash
ffmpeg -version
ffprobe -version
```

### 2. Install Ollama

=== "Linux / macOS"

    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```

=== "Windows"

    Download the installer from [ollama.com/download](https://ollama.com/download).

Start the Ollama service and pull the required models:

```bash
ollama serve          # Start in a separate terminal
ollama pull qwen2.5-vl:7b   # ~4.7GB — vision model for analysis + captions
ollama pull qwen2.5:7b      # ~4.7GB — text model for transcript segmentation
```

**Vision models** (for semantic analysis and caption generation):

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `qwen2.5-vl:3b` | ~2GB | Fast | Good |
| `qwen2.5-vl:7b` | ~4.7GB | Moderate | Best (default) |

**Segmentation model** (for transcript analysis — text-only, no vision encoder needed):

| Model | Size | Speed | Notes |
|-------|------|-------|-------|
| `qwen2.5:7b` | ~4.7GB | Fast | Default — text-only, efficient for JSON output |
| `qwen2.5-vl:7b` | ~4.7GB | Slower | Works but wastes VRAM on unused vision encoder |

### 3. Install YACG

```bash
git clone https://github.com/Mzzkc/yacg.git
cd yacg
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

After installation, the `yacg` command is available in addition to `python -m yacg`.

!!! note "Auto-Install Behavior"
    On first run, YACG checks for missing Python packages and attempts to install them via pip. In a virtualenv (recommended), this is safe. Outside a virtualenv, it may fall back to `pip install --user` or `pip install --user --break-system-packages` on PEP 668 systems. Using a virtualenv (as shown above) avoids this.

### 4. Verify Setup

```bash
python -m yacg check
```

This checks for FFmpeg, FFprobe, all Python packages, Ollama connectivity, model availability (including `qwen2.5:7b` segmentation model), Whisper model cache status, and DNN face detection model availability.

### 5. Install DNN Face Detection Model (Optional)

YACG uses a DNN SSD face detector for high-quality face-aware cropping. Without the model files, it falls back to the less accurate Haar cascade detector (lower quality for side profiles and diverse skin tones).

To enable the DNN detector, download two files and place them in `~/.yacg/models/` (YACG also checks `<package_install_dir>/models/` inside the installed package):

```bash
mkdir -p ~/.yacg/models
# Download from OpenCV's GitHub repository:
curl -L -o ~/.yacg/models/deploy.prototxt \
  "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
curl -L -o ~/.yacg/models/res10_300x300_ssd_iter_140000.caffemodel \
  "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
```

Run `yacg check` to verify the model is detected.

!!! note "First-Run Download"
    The Whisper transcription model is downloaded from HuggingFace Hub on first use (~500MB for `small`, up to ~3GB for `large-v3`). This requires an internet connection. Run `yacg check` to see if the model is cached.

## Quick Start

### Process a Local Video

```bash
python -m yacg process \
  --video /path/to/video.mp4 \
  --title "My Video" \
  --output-dir ./my_clips
```

> **Tip:** After `pip install -e .`, you can also use the shorter `yacg` command instead of `python -m yacg`.

Output appears in `./my_clips/`:

- `my_video_clip_01_score85.mp4`, `my_video_clip_02_score72.mp4`, ... — vertical clips with subtitles
- `my_video_clip_01_score85_thumb.jpg`, ... — JPEG thumbnails for each clip
- `clips_report.csv` — metadata for all clips (17 columns)

### Process a YouTube Video

```bash
python -m yacg youtube \
  --url "https://youtube.com/watch?v=XXXXX" \
  --top-n 5
```

### Process a Directory of Videos

```bash
python -m yacg batch \
  --videos-dir /path/to/videos/ \
  --min-score 60
```

Each video gets its own subdirectory under `--output-dir`.

## Configuration

YACG uses sensible defaults but supports INI-style configuration files for customization.

### Generate a Default Config

```bash
python -m yacg generate-config --output config.ini
```

This generates a fully-commented INI file with all available configuration keys and their defaults.

### Use a Config File

```bash
python -m yacg process \
  --video video.mp4 \
  --config config.ini
```

CLI flags override config file values.

### Config Sections

```ini
[Model]
model_name = qwen2.5-vl:7b
ollama_host = http://localhost:11434
whisper_model = small
segmentation_model = qwen2.5:7b
whisper_device = auto
whisper_compute_type = auto
num_frames = 3

[SceneDetection]
; NOTE: These settings configure the legacy PySceneDetect scene detector.
; The current pipeline uses LLM-driven transcript segmentation instead.
; These values have no effect on the default pipeline.
threshold = 3.0
min_scene_len = 7.0
max_scene_len = 60.0

[Segmentation]
pause_threshold = 0.3
min_segment_duration = 15.0
max_segment_duration = 45.0
vad_filter = true

[ClipSelection]
; WARNING: These fields are effectively unused. The CLI always passes --top-n (default: 20)
; and --min-score (default: 70) directly, bypassing these config values entirely.
; process_video() also defaults top_n=20 and min_score=70.0 via method parameters.
; Use CLI flags --top-n and --min-score instead.
top_n_clips = 10
min_virality_score = 70.0

[ASMR Optimization]
; Legacy alias: [ASMR Optimization] is accepted for backward compatibility.
; The content_type key can also be set here. Prefer [ContentProfile] for new configs.
asmr_mode = false
; Set to true for ASMR content (auto-enabled when content_type = asmr)

[ContentProfile]
content_type = general
; Options: general, gaming, cooking, asmr, educational, fitness, comedy, music, beauty, tech, vlog
channel_description =
target_audience =
tone = engaging
; Options: energetic, calm, professional, casual, humorous, inspirational, dramatic, engaging
platform = all
; Options: tiktok, reels, shorts, all
caption_length = medium
; Options: short, medium, long
hashtag_count = 5
; Range: 3-7
custom_instructions =
```

### Content Type Effects

The `--content-type` flag (or `content_type` in config) does more than set preset values — it changes LLM behavior in four pipeline stages:

1. **Segmentation** — the LLM receives genre-specific guidance on what makes a good segment (e.g., gaming looks for clutch moments and reactions, cooking looks for technique reveals)
2. **Semantic analysis** — the VLM evaluates emotional intensity, hooks, and sensory quality using content-type-specific criteria
3. **Caption generation** — the VLM adopts a content-type-specific persona and platform awareness
4. **Audio analysis** — trigger word detection uses content-specific keyword lists (ASMR keywords vs. general engagement keywords)

Content types with specialized prompts: `gaming`, `cooking`, `educational`, `asmr`, `fitness`, `music`, `comedy`. Types using generic prompts: `beauty`, `tech`, `vlog`, `general`.

#### Content Preset Values

Each content type preset sets these defaults (overridable individually via CLI flags or config):

| Content Type | Target Audience | Tone | Platform | Caption Length | Hashtags |
|-------------|----------------|------|----------|---------------|----------|
| `general` | general social media viewers | engaging | all | medium | 5 |
| `gaming` | gamers and gaming enthusiasts | energetic | all | short | 5 |
| `cooking` | home cooks and food enthusiasts | casual | reels | medium | 5 |
| `asmr` | ASMR listeners seeking relaxation and tingles | calm | all | short | 4 |
| `educational` | learners and curious minds | professional | shorts | long | 4 |
| `fitness` | fitness enthusiasts and gym-goers | energetic | reels | medium | 5 |
| `comedy` | people looking for entertainment and laughs | humorous | tiktok | short | 5 |
| `music` | music lovers and artists | engaging | all | short | 5 |
| `beauty` | beauty and skincare enthusiasts | casual | reels | medium | 5 |
| `tech` | tech enthusiasts and early adopters | professional | shorts | medium | 4 |
| `vlog` | viewers interested in personal stories and daily life | casual | all | medium | 5 |

```ini
[Output]
output_dir = ./clip_output
dry_run = false

[Temporal]
context_padding = 2.0

[Subtitle]
font_name =
font_size_pct = 0.055
primary_color = &H00FFFFFF
outline_color = &H00000000
outline_width = 3.0
shadow = 1.5
margin_v_pct = 0.38
margin_h_pct = 0.15

[Scoring]
weights = {"hook": 0.20, "emotional": 0.15, "audio_peaks": 0.15,
           "asmr": 0.12, "motion": 0.12, "narrative": 0.10,
           "high_freq": 0.10, "uniqueness": 0.08, "composition": 0.05,
           "visual": 0.02, "duration": 0.05}
```

!!! note "Scoring Weights JSON Format"
    If the `weights` JSON in your config file is malformed, YACG silently falls back to default weights with a warning — it does not raise an error. Use `--verbose` to confirm your custom weights were applied.

### Scoring Weights

The virality scorer uses 11 weighted components. Adjust these for your content niche:

| Component | Default Weight | Signal Source |
|-----------|---------------|---------------|
| `hook` | 0.20 | Semantic: hook potential (0-10) |
| `emotional` | 0.15 | Semantic: emotional intensity (0-10) |
| `audio_peaks` | 0.15 | Audio: peak energy (0-2, normalized to 0-10) |
| `asmr` | 0.12 | Semantic: ASMR quality (0-10) |
| `motion` | 0.12 | Visual: optical flow (0-1, normalized to 0-10) |
| `narrative` | 0.10 | Semantic: narrative interest (0-10) |
| `high_freq` | 0.10 | Audio: high-frequency content (0-1, normalized to 0-10) |
| `uniqueness` | 0.08 | Semantic: uniqueness (0-10) |
| `composition` | 0.05 | Visual: rule-of-thirds (0-1, normalized to 0-10) |
| `visual` | 0.02 | Visual: color interest (0-1, normalized to 0-10) |
| `duration` | 0.05 | Clip length (optimal 7-30s) |

The default weights sum to 1.14. The scoring formula divides by total weight, so the sum itself has no effect on scores — only the relative proportions matter. The `asmr` component's weight of 0.12 gives ASMR signals meaningful influence in the composite score. Adjust weights as needed for your content niche. Pass them as a JSON dictionary in the `[Scoring]` section.

### How Scoring Works

Each component's raw value is normalized to a 0-10 scale, multiplied by its weight, and summed:

```
score = (weighted_sum / total_weight) × 10.0    (clamped to 0-100)
```

**Normalization ranges:** `audio_peaks` 0-2, `high_freq` 0-1, `motion` 0-1, `visual` 0-1, `composition` 0-1 — all scaled to 0-10. Semantic dimensions (`hook`, `emotional`, etc.) are already 0-10 from the VLM.

**Duration scoring:** Uses a 5-zone piecewise curve tuned for short-form video: 0-5s ramps linearly (score = duration), 5-7s accelerates from 5 to 10, 7-30s is optimal (score 10), 30-60s decays linearly from 10 to 5, >60s continues decaying from 5 at 0.1 points per second. Clips in the 7-30s sweet spot get the maximum duration score.

**Without VLM:** When Ollama is unavailable, semantic weights are redistributed proportionally to audio and visual components. Confidence drops from 1.0 to 0.5.

**Interpreting scores:** Below 50 rarely indicates viral potential. 60-70 is moderate, 70-85 is strong, 85+ is exceptional. Lower `--min-score` to produce more clips; raise it to be more selective.

## Troubleshooting

### Ollama Not Running

```
Error: Failed to connect to Ollama
```

```bash
# Check if running
ps aux | grep ollama

# Start it
ollama serve
```

### Model Not Found

```
Error: Model qwen2.5-vl:7b not found
```

```bash
ollama pull qwen2.5-vl:7b
```

### FFmpeg Not Found

```
Error: ffprobe: command not found
```

=== "Ubuntu / Debian"

    ```bash
    sudo apt install ffmpeg
    ```

=== "macOS"

    ```bash
    brew install ffmpeg
    ```

### Out of Memory

- Use a smaller Ollama model: `--model qwen2.5-vl:3b`
- Use a smaller Whisper model: `--whisper-model tiny` or `--whisper-model base` (default is `small`)
- Close other GPU-intensive applications

### No Clips Extracted

If YACG finds segments but extracts zero clips:

- Lower `--min-score` (e.g., `--min-score 40`)
- Increase `--top-n` (e.g., `--top-n 20`)
- Check that the video has audible speech (transcript segmentation requires it)

### No Speech Detected / Language Detection Failed

```
RuntimeError: Language detection confidence too low
```

Whisper requires clear audible speech. Language detection confidence must exceed 0.5 (below = error) and 0.8 (below = warning). If your video is mostly music, ambient audio, or multi-language:

- Ensure the video has clear speech in a single primary language
- Try a larger Whisper model: `--whisper-model medium` or `--whisper-model large-v3`
- Try disabling VAD: `--no-vad-filter`

### Slow Processing / Unexpected Delays

Several things can cause unexpectedly long processing:

- **Codec transcoding:** Compatible codecs (no transcoding needed): H.264, H.265/HEVC, MPEG-4, MPEG-2, MJPEG. All other codecs (AV1, VP9, ProRes, etc.) are automatically transcoded to H.264 before processing, which can take up to 1 hour for long videos. Pre-convert with `ffmpeg -i input.mkv -c:v libx264 -crf 23 output.mp4` to skip this step.
- **Long videos:** Videos exceeding 1 hour trigger a processing time warning. Processing time scales with video length.
- **First-run model download:** The Whisper model is downloaded on first use (~500MB-3GB depending on model size).
- **Disk space:** YACG warns if less than 1 GB of free space is available in the output directory.

### Leftover `.yacg_tmp/` Directory

YACG uses a staging directory (`.yacg_tmp/` inside the output dir) during clip extraction. If the process is interrupted (killed, power loss, crash), this directory may remain. It can be safely deleted.

### Garbled VLM Output

If **semantic analysis** returns nonsensical results, your vision model may not handle multiple images well. Try `--num-frames 1` to send only one frame per segment. Note: `--num-frames` only affects semantic analysis — caption generation always uses 3 frames at 25%, 50%, and 75% positions regardless of this setting. If captions are garbled, try a different VLM model. Use `--verbose` to see suppressed warnings that may help diagnose the issue.

### Bootstrap Install Failure

If auto-install fails and the error message references `requirements-clip-extractor.txt`, ignore that stale message — the referenced file does not exist (legacy reference from old project name). Install manually with:

```bash
pip install -e .    # development install
# or
pip install .       # user install
```

### Partial Success (Exit Code 2)

Exit code 2 means clips were produced but some errors occurred. Common causes:

- Individual subtitle or caption generation failed for specific clips (those clips are deleted)
- In batch mode, a video with ANY errors counts as "had errors" in the summary, even if it produced clips

Check the `=== ERRORS (N) ===` block at the end of output for details.
