# Design: Transcript-First Pipeline + Subtitles + Hardening + Documentation

**Date**: 2026-03-05
**Status**: Active — being implemented via Mozart score

## Problem

YACG produces clips without burned-in subtitles, making them unusable for TikTok/Reels/Shorts without manual post-processing. The tool also has `--no-semantic`, `--no-captions`, and `--no-vertical` flags that allow disabling core features, creating a false sense of reliability by silently degrading output quality instead of surfacing failures.

Additionally, the current pipeline uses PySceneDetect (visual transitions) as the primary segmentation signal, which produces arbitrary clip boundaries that don't align with speech content. Viral clips are defined by what's being *said* — complete thoughts, hooks, narrative arcs — not by visual cuts. The pipeline needs to be inverted: transcript-first segmentation using LLM analysis of the full Whisper transcription.

Finally, the project lacks documentation. It needs a GitHub Pages documentation site covering installation, usage, CLI reference, and architecture.

## Product Context

YACG replaces paid services like OpusClip. It takes long-form video and produces ready-to-post short-form clips with:
- Semantic analysis for intelligent clip selection
- Vertical 9:16 crop with face-aware framing
- Speech-to-text subtitles (TikTok word-pop style)
- AI-generated social media captions (hook, description, hashtags)
- Virality scoring

Every feature is core. If any fails, the clip fails.

## Changes

### 1. Transcript-First Segmentation (`transcript_segmenter.py`)

**Pipeline inversion**: Replace PySceneDetect as the primary segmentation signal. The new flow:

1. Run faster-whisper on the FULL source video (before any segmentation) to get word-level timestamps
2. Format the transcript into timestamped text blocks
3. Send the full transcript to Ollama (qwen2.5-vl) with a segmentation prompt asking it to identify viral-worthy segments — complete thoughts, hooks, narrative arcs, 15-45s target duration
4. LLM responds with JSON array: `[{start_time, end_time, hook_summary, segment_type}, ...]`
5. Snap boundaries to the nearest speech pause (gap >300ms between words) so clips don't cut mid-word
6. Validate: no overlaps, within video bounds, min/max duration enforcement

**New module**: `transcript_segmenter.py` with class `TranscriptSegmenter`:
- `full_transcribe(video_path, whisper_model) → list[WordTimestamp]` — runs faster-whisper with `word_timestamps=True`
- `segment_by_content(words, title, ollama_host, model) → list[SegmentBoundary]` — sends transcript to Ollama, parses response
- `refine_boundaries(segments, words) → list[SceneSegment]` — snaps to speech pauses, validates

**Key design choice**: The LLM gets the *text* of the transcript, not video frames. It segments by content. Visual/audio analysis happens later as scoring signals. The single Whisper transcription is reused for both segmentation and subtitle burning (no double transcription).

**PySceneDetect demotion**: `scene_detector.py` is not deleted but becomes secondary — available for optional boundary validation (avoid cutting mid-visual-transition) rather than primary segmentation.

**`transcript_bridge.py` removal**: The yt-transcriber bridge is replaced by direct faster-whisper usage. Delete or deprecate.

**Failure model**: If Ollama fails to segment, the pipeline raises. No fallback to scene-based segmentation. If Whisper returns no words (silent video), raise.

### 2. New: Burned-In Subtitles (`subtitle_burner.py`)

**Transcription**: Receives pre-computed word-level timestamps from the full-video Whisper run (step 1 above). Does NOT re-transcribe each clip. Slices the relevant words for each clip's time range.

**Subtitle format**: ASS (Advanced SubStation Alpha) — supported natively by FFmpeg's libass filter. Chosen over SRT because ASS supports:
- Per-word timing and animation
- Font styling, outline, shadow
- Precise positioning

**Style**: TikTok word-pop
- 1-3 words at a time, appearing in sync with speech
- Large bold sans-serif font (scaled to frame dimensions)
- White fill, black outline (3px), drop shadow
- Positioned in the platform-safe zone (see below)

**Platform-safe positioning** (9:16 frame, e.g. 1080x1920):
- Vertical: text anchored at ~62% from top (~1190px). Clears:
  - TikTok bottom chrome (username/caption, ~18% of height)
  - Instagram Reels bottom chrome (~16%)
  - YouTube Shorts bottom chrome (~15%)
- Horizontal: centered, with 15% margin on each side (~162px). Clears:
  - Right-side action buttons on all three platforms
- Font size: ~5.5% of frame height (~106px on 1920h), adjusted proportionally for other resolutions

**Pipeline integration**: Post-processing step after clip extraction, before caption generation.

```
clip.mp4 → faster-whisper (word timestamps) → generate .ass file → ffmpeg hardburn → replace clip.mp4
```

**Whisper config**: Model defaults to `base` for speed. Configurable via `--whisper-model` CLI flag (tiny/base/small/medium/large-v3).

**Failure model**: If transcription returns no words (silent clip) or FFmpeg burn-in fails, the clip is an error. No un-subtitled clip is produced. There is no flag to disable subtitles.

### 3. Remove `--no-semantic` Flag

Semantic analysis is mandatory. Remove:
- CLI flag `--no-semantic` from argparse
- `enable_semantic` from `PipelineConfig`
- All conditional paths that skip semantic analysis
- Tests that exercise `--no-semantic` combinations

If Ollama/semantic analysis fails, the pipeline reports an error for that segment. No fallback to default scores.

### 4. Remove `--no-captions` Flag

AI caption generation is mandatory. Remove:
- CLI flag `--no-captions` from argparse
- `enable_captions` from `PipelineConfig`
- All conditional paths that skip caption generation

If Ollama caption generation fails after retries, the clip is an error.

### 5. Remove `--no-vertical` Flag

Vertical 9:16 crop is mandatory. Remove:
- CLI flag `--no-vertical` from argparse
- `vertical_crop` toggle from `PipelineConfig`
- All conditional paths that skip vertical cropping

### 6. Fix Smart Cropper

**Known bug**: when `face_aware=False`, the SmartCropper produces off-center crops. The fallback should use motion/brightness center or visual focus detection, not dead-center the frame.

### 7. Update Test Suite

- Remove all tests exercising removed flag combinations (the "8 combinations" matrix collapses to 1 path)
- Add subtitle integration tests:
  - Clip with speech produces subtitled output
  - Silent clip fails with clear error
  - Subtitle timing matches speech
  - ASS file is valid
  - Platform positioning is correct for 9:16 frame
- Add semantic-failure-is-error tests
- Add caption-failure-is-error tests

## Files Affected

| File | Change |
|------|--------|
| `yacg/transcript_segmenter.py` | **New** — LLM-driven transcript segmentation |
| `yacg/subtitle_burner.py` | **New** — ASS generation + FFmpeg burn-in (receives pre-computed words) |
| `yacg/pipeline.py` | **Major rewrite** — transcript-first flow, subtitle step, no conditionals |
| `yacg/models.py` | Remove toggle fields, add `whisper_model`, transcript/subtitle fields |
| `yacg/cli.py` | Remove 3 flags, add `--whisper-model` |
| `yacg/core/semantic_analyzer.py` | Delete `_default_features()`, raise on failure |
| `yacg/caption_generator.py` | Raise on failure, no `None` returns |
| `yacg/extractors/clip_extractor.py` | Remove `vertical` param (always crop) |
| `yacg/extractors/smart_cropper.py` | Fix fallback crop logic (visual-focus center) |
| `yacg/core/scene_detector.py` | Demoted to optional boundary refinement |
| `yacg/transcript_bridge.py` | **Removed/deprecated** — replaced by direct faster-whisper |
| `yacg/bootstrap.py` | `faster_whisper` → required dep |
| `yacg/utils/config.py` | Remove INI reads for removed fields |
| `tests/*.py` | Rewrite flag-combo tests, add segmentation/subtitle/failure tests |
| `docs/` | **New** — GitHub Pages documentation site |

## Dependencies

No new dependencies. faster-whisper and FFmpeg libass are already available.

## New Pipeline Flow

```
1. Validate video + extract metadata
2. Full-video Whisper transcription (word-level timestamps)
3. LLM segmentation: send transcript to Ollama → clip boundaries
4. Snap boundaries to speech pauses (>300ms gaps)
5. Analyze each segment (audio + visual + semantic scoring)
6. Score virality
7. Rank + filter + select top_n
8. Extract clips (always vertical 9:16, face-aware crop)
9. Burn subtitles (reuses Whisper data from step 2)
10. Generate captions (Ollama — hook, description, hashtags)
11. Write CSV report (filename → full_caption mapping is critical)
```

The CSV maps clip filenames to generated captions. An external tool reads this CSV to clipboard-copy captions and stage videos for scheduling on TikTok/Instagram.

## Design Notes

- **Vertical crop must always be face-aware or center-focused.** The smart cropper should always anchor on the detected face, or fall back to the center of visual focus — never an arbitrary crop offset.
- **Single Whisper run.** The full-video transcription is done once and reused for segmentation, subtitle burning, and trigger word detection. No double transcription.
- **LLM segments by content, not visuals.** The Ollama segmentation prompt receives transcript text (what's being said), not video frames. Visual analysis happens later as a scoring signal.

## Out of Scope

- Subtitle font customization (hardcoded style for now)
- Subtitle language selection (English only, auto-detected by Whisper)
- Caption text burned into video (captions remain CSV-only metadata)
- Animated text effects beyond basic word-pop timing
