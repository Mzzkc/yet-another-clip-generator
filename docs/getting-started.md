# Getting Started

## System Requirements

- **OS:** Linux, macOS, or WSL2 on Windows
- **Python:** 3.10 or newer
- **RAM:** 8GB minimum, 16GB recommended for the 7B model
- **Storage:** ~6GB for model weights + video storage
- **GPU:** Optional but recommended for Ollama inference

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

### 3. Install Viral Clip Extractor

```bash
git clone https://github.com/Mzzkc/yacg.git
cd yacg
pip install -r requirements-clip-extractor.txt
```

### 4. Verify Setup

```bash
python -m viral_clip_extractor check
```

This checks for FFmpeg, FFprobe, all Python packages, Ollama connectivity, model availability (including `qwen2.5:7b` segmentation model), Whisper model cache status, and DNN face detection model availability.

## Quick Start

### Process a Local Video

```bash
python -m viral_clip_extractor process \
  --video /path/to/video.mp4 \
  --title "My Video" \
  --output-dir ./my_clips
```

Output appears in `./my_clips/`:

- `my_video_clip_01_score85.mp4`, `my_video_clip_02_score72.mp4`, ... — vertical clips with subtitles
- `my_video_clip_01_score85_thumb.jpg`, ... — JPEG thumbnails for each clip
- `clips_report.csv` — metadata for all clips (17 columns)

### Process a YouTube Video

```bash
python -m viral_clip_extractor youtube \
  --url "https://youtube.com/watch?v=XXXXX" \
  --top-n 5
```

### Process a Directory of Videos

```bash
python -m viral_clip_extractor batch \
  --videos-dir /path/to/videos/ \
  --min-score 60
```

Each video gets its own subdirectory under `--output-dir`.

## Configuration

VCE uses sensible defaults but supports INI-style configuration files for customization.

### Generate a Default Config

```bash
python -m viral_clip_extractor generate-config --output config.ini
```

This generates a fully-commented INI file with all available configuration keys and their defaults.

### Use a Config File

```bash
python -m viral_clip_extractor process \
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
threshold = 3.0
min_scene_len = 7.0
max_scene_len = 60.0

[Segmentation]
pause_threshold = 0.3
min_segment_duration = 15.0
max_segment_duration = 45.0
vad_filter = true

[ClipSelection]
top_n_clips = 10
min_virality_score = 70.0

[ASMR Optimization]
asmr_mode = true
content_type = asmr

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

### Scoring Weights

The virality scorer uses 11 weighted components. Adjust these for your content niche:

| Component | Default Weight | Signal Source |
|-----------|---------------|---------------|
| `hook` | 0.20 | Semantic: hook potential (0-10) |
| `emotional` | 0.15 | Semantic: emotional intensity (0-10) |
| `audio_peaks` | 0.15 | Audio: peak energy (0-1) |
| `asmr` | 0.12 | Semantic: ASMR quality (0-10) |
| `motion` | 0.12 | Visual: optical flow (0-1) |
| `narrative` | 0.10 | Semantic: narrative interest (0-10) |
| `high_freq` | 0.10 | Audio: high-frequency content (0-1) |
| `uniqueness` | 0.08 | Semantic: uniqueness (0-10) |
| `composition` | 0.05 | Visual: rule-of-thirds (0-1) |
| `visual` | 0.02 | Visual: color interest (0-1) |
| `duration` | 0.05 | Clip length (optimal 7-30s) |

The default weights intentionally sum to 1.14 (not 1.0) to provide ASMR content a slight scoring boost. Adjust as needed for your content niche. Pass them as a JSON dictionary in the `[Scoring]` section.

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

If VCE finds segments but extracts zero clips:

- Lower `--min-score` (e.g., `--min-score 40`)
- Increase `--top-n` (e.g., `--top-n 20`)
- Check that the video has audible speech (transcript segmentation requires it)
