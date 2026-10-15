# YACG — Yet Another Clip Generator

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
yacg check

# Process a local video
yacg process --video video.mp4 --title "My Video"

# Process a YouTube video
yacg youtube --url https://youtube.com/watch?v=XXXXX

# Batch process a directory
yacg batch --videos-dir /path/to/videos/
```

### Whisper on another machine (optional)

Transcription runs locally by default. To use a GPU elsewhere, run a Whisper server
there (any server with the standard `POST /v1/audio/transcriptions` endpoint, e.g. a
[faster-whisper server](https://github.com/speaches-ai/speaches)) and point yacg at it:

```bash
export WHISPER_API_URL=http://gpu-box:8000
yacg process --video video.mp4 --title "My Video" --whisper-backend remote
```

`--whisper-url` overrides the variable, and both can live in the config file
(`whisper_backend`, `whisper_url` under `[Model]`). yacg sends 16 kHz mono FLAC and
keeps the same word timestamps and language checks as a local run. Set
`WHISPER_API_KEY` if the server needs a key.

## Project Structure

```
yacg/                       # Main package
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
