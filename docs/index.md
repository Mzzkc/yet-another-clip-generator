# YACG

**Automatically extract viral-potential clips from long-form videos using multi-modal AI analysis.**

YACG (YACG) takes a long-form video — local file or YouTube URL — and produces short, vertical (9:16) clips optimized for TikTok, Instagram Reels, and YouTube Shorts. It uses LLM-driven transcript segmentation, multi-signal virality scoring, face-aware smart cropping, word-pop subtitle burning, and AI-generated Instagram captions.

---

## Key Features

- **Transcript-first segmentation** — faster-whisper transcription + Ollama LLM identifies natural content boundaries (hooks, narrative arcs, emotional peaks)
- **Multi-modal virality scoring** — audio (librosa), visual (OpenCV), and semantic (Qwen2.5-VL) signals combined into a weighted 0-100 score
- **Smart vertical cropping** — face-aware 9:16 crop using DNN SSD face detection with Haar cascade fallback and brightness-center fallback
- **Configurable subtitles** — TikTok-style ASS word-pop subtitles with configurable font, color, size, and margins, burned directly into clips via FFmpeg
- **Instagram captions & thumbnails** — AI-generated hooks, descriptions, hashtags, virality predictions, and midpoint thumbnails for each clip
- **Multi-frame VLM analysis** — semantic and caption analysis uses multiple JPEG frames (configurable 1-5) per segment for richer context
- **Protocol-based dependency injection** — 8 swappable component interfaces for custom implementations and testing
- **Parallel extraction** — clip cutting and subtitle burning run in parallel via ThreadPoolExecutor
- **Progress callbacks** — hookable progress reporting for GUI/web embedding
- **YouTube support** — download and process YouTube videos via yt-dlp
- **Batch processing** — process entire directories of videos in one command
- **Content type system** — ASMR-optimized by default, with `--content-type general` for non-ASMR content
- **GPU acceleration** — automatic CUDA detection for faster-whisper (configurable device/compute type)

---

## Quick Example

Process a local video and extract the top 5 clips:

```bash
python -m yacg process \
  --video my_video.mp4 \
  --title "My Video Title" \
  --top-n 5 \
  --min-score 60
```

Process a YouTube video:

```bash
python -m yacg youtube \
  --url "https://youtube.com/watch?v=XXXXX" \
  --top-n 3
```

Dry run — analyze without extracting:

```bash
python -m yacg process \
  --video my_video.mp4 \
  --dry-run
```

Check that all dependencies are installed:

```bash
python -m yacg check
```

---

## How It Works

YACG runs a 10-step pipeline on each video:

1. **Transcribe** — full-video transcription with word-level timestamps (faster-whisper, auto device/precision)
2. **Segment** — LLM identifies viral-worthy segments from the transcript (text-only model, default `qwen2.5:7b`)
3. **Analyze** — each segment scored on audio, visual, and semantic dimensions (multi-frame VLM analysis)
4. **Rank** — segments sorted by composite virality score, filtered by threshold
5. **Extract** — top clips cut with FFmpeg in parallel, smart-cropped to 9:16 vertical (to staging dir)
6. **Subtitles** — word-pop subtitles generated and burned into each clip in parallel
7. **Thumbnails** — midpoint JPEG thumbnail extracted for each clip
8. **Captions** — Instagram-optimized captions generated via Ollama VLM (multi-frame analysis)
9. **Stage** — clips and thumbnails moved from staging dir to final output
10. **Report** — CSV report with 17 columns of metadata for all clips

---

## Output

For each processed video, YACG produces:

- Vertical MP4 clips with burned-in subtitles in `clip_output/` (named `{title}_clip_01_score85.mp4`)
- JPEG thumbnails for each clip (`{title}_clip_01_score85_thumb.jpg`)
- A `clips_report.csv` with 17 columns of metadata for all extracted clips
- Instagram-ready captions with hooks, hashtags, and virality scores

---

## Requirements

- Python 3.10+
- FFmpeg and FFprobe
- Ollama with a vision-language model (default: `qwen2.5-vl:7b`) and text model (default: `qwen2.5:7b`)
- 8GB RAM minimum (16GB recommended)
- CUDA toolkit (optional, for GPU-accelerated Whisper transcription)

See [Getting Started](getting-started.md) for full installation instructions.
