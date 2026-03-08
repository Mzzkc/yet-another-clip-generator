# Viral Clip Extractor

Automatically extract viral-potential short-form clips from long-form video.
Takes a video (local file or YouTube URL), detects scenes, scores viral
potential, crops to 9:16, burns in subtitles, and outputs ready-to-post clips
for TikTok, Reels, and Shorts.

## Quick Start

```bash
git clone <repo> && cd yacg
./setup.sh            # installs system deps + Python package
```

Or manually:

```bash
pip install -e ".[dev]"
```

## Requirements

- Python 3.10+
- FFmpeg (with libass for subtitle burn-in)
- Ollama with `qwen2.5-vl:7b` (semantic analysis and captions)
- faster-whisper (word-level transcription)

## Usage

```bash
# Check environment
vce check

# Process a local video
vce process --video video.mp4 --title "My Video"

# Process a YouTube video
vce youtube --url https://youtube.com/watch?v=XXXXX

# Batch process a directory
vce batch --videos-dir /path/to/videos/
```

## Project Structure

```
viral_clip_extractor/       # Main package
    core/                   # Analysis engines (audio, scene, semantic, visual, virality)
    extractors/             # Clip extraction and smart cropping
    utils/                  # Config and video utilities
    cli.py                  # CLI entry point
    pipeline.py             # Orchestrates the full pipeline
    subtitle_burner.py      # ASS subtitle burn-in via FFmpeg
    transcript_segmenter.py # Word-level transcript segmentation
    caption_generator.py    # LLM-powered caption generation
tests/                      # Test suite (pytest)
docs/                       # Documentation (MkDocs)
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Documentation

Full docs are in `docs/`. Build with:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## License

MIT
