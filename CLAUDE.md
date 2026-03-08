# YACG — Yet Another Clip Generator

Extracts viral-potential short-form clips from long-form video. Replaces paid
services like OpusClip. Outputs ready-to-post clips for TikTok, Reels, and
Shorts with burned-in subtitles and auto-generated captions.

## Repository Layout

| Path | Purpose |
|------|---------|
| `yacg/` | Main Python package |
| `yacg/cli.py` | CLI entry point (`yacg` command) |
| `yacg/pipeline.py` | Orchestrates the full processing pipeline |
| `yacg/core/` | Analysis engines: audio, scene, semantic, visual, virality |
| `yacg/extractors/` | Clip extraction and smart 9:16 cropping |
| `yacg/utils/` | Config loading, video utilities |
| `yacg/subtitle_burner.py` | ASS subtitle burn-in via FFmpeg libass |
| `yacg/transcript_segmenter.py` | Word-level transcript segmentation |
| `yacg/caption_generator.py` | LLM-powered caption generation |
| `tests/` | Test suite (pytest) |
| `docs/` | MkDocs documentation |
| `docs/design/` | Archived original design and roadmap docs |
| `pyproject.toml` | Project metadata, dependencies, tool config |
| `setup.sh` | One-command environment setup |

## Design Principles

1. **Every feature is mandatory.** No optional flags, no silent fallbacks.
   Semantic analysis, captions, subtitles, and vertical crop must all succeed
   or the clip fails.

2. **Failure = error.** If a processing step cannot produce a real result, it
   must raise an exception. Never silently degrade.

3. **Face-aware vertical crop.** The 9:16 crop must anchor on a detected face
   or fall back to center of visual focus. Never use an arbitrary offset.

4. **Transcript-first segmentation.** Clip boundaries are driven by speech
   content (complete thoughts, hooks, narrative arcs). Visual and audio signals
   are scoring inputs, not boundary signals.

## What Goes Where

- **New analysis engines** → `yacg/core/`
- **New extraction/post-processing** → `yacg/extractors/`
- **Shared helpers** → `yacg/utils/`
- **Tests** → `tests/` (top-level, not inside the package)
- **Documentation** → `docs/`

## What Does NOT Belong in This Repo

- Mozart score files (`yacg-*.yaml`)
- Mozart state files (`.mozart-*.jsonl`, `.mozart-*.json`, `.mozart-*.db`)
- IDE configuration (`.vscode/`, `.idea/`)
- Virtual environments (`.venv/`, `venv/`)
- Output artifacts (`clip_output/`, `downloads/`)

These are all listed in `.gitignore`.

## Key Technical Details

- **LLM:** Ollama with `qwen2.5-vl:7b` for semantic analysis and captions
- **Transcription:** faster-whisper for word-level timestamps (required)
- **Video processing:** FFmpeg with libass for ASS subtitle burn-in
- **Subtitle safe zone:** 62% from top, 15% side margins (clears TikTok/Reels/Shorts UI)
- **Scene detection:** PySceneDetect with OpenCV backend
- **Audio analysis:** librosa for energy, tempo, spectral features

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## Building Docs

```bash
pip install -e ".[docs]"
mkdocs serve
```
