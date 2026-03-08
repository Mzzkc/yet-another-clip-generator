# Agent Handoff: VCE Subtitles + Transcript-First Pipeline

**Date**: 2026-03-05
**Status**: Score cancelled mid-run, needs redesign and recomposition

## What Happened This Session

1. **Investigated vce-hardening.yaml Mozart score "failure"** — all 13 sheets actually completed successfully. Root cause was a regex validation on the final sheet: `(?i)final verdict.*pass` didn't match because "Final Verdict" and "PASS" were on separate lines. Fixed to `(?si)` in `vce-hardening.yaml:1012`.

2. **Assessed project state** — 224 tests passing, 80% coverage, all CLI subcommands working. The hardening score did its job.

3. **Ran full pipeline on user's video** (`https://youtu.be/-Zds0V0kg94`) — 5 clips extracted to `/mnt/c/Users/mckys/Videos/test_clips/`. Ollama semantic analyzer returned 500 errors on every segment, silently fell back to defaults. Scores clustered at 40-41 (garbage). User confirmed: crop positioning was off, clip boundaries were arbitrary.

4. **Designed and composed `vce-subtitles.yaml`** — 11 stages, 21 sheets. Submitted to Mozart, ran to sheet 2, then **cancelled** because scope expanded.

## Why It Was Cancelled

The user wants **transcript-first segmentation** included in this score, not as future work. This is a pipeline architecture inversion:

**Current pipeline**: PySceneDetect (visual cuts) → analyze segments → score → extract
**New pipeline**: Whisper (full transcript) → segment by CONTENT (thoughts, hooks, arcs) → score with all signals → snap boundaries to speech pauses → extract

The design doc at `/home/emzi/Projects/yacg/docs/plans/2026-03-05-subtitles-and-hardening-design.md` needs to be updated to move transcript-first segmentation from "Future Work" to a core change.

## Full Scope for the Score

### 1. Transcript-First Segmentation (NEW — pipeline inversion)
- Run Whisper on the FULL source video first (before any segmentation)
- Use transcript content to define clip boundaries: complete thoughts, hooks, narrative arcs
- PySceneDetect becomes secondary — used to validate/adjust boundaries (don't cut mid-visual-transition)
- Visual and audio analysis remain as scoring signals, not boundary signals
- The transcript used for segmentation can be reused for subtitles (no double-transcription)

### 2. Burned-In TikTok Word-Pop Subtitles
- New `subtitle_burner.py`: faster-whisper word-level timestamps → ASS subtitle file → FFmpeg hardburn
- Style: 1-3 words at a time, large bold text, white fill + black outline + drop shadow
- Platform-safe positioning: 62% from top, 15% side margins (clears TikTok/Reels/Shorts UI)
- Failure = error (no un-subtitled clips)

### 3. Remove All Optional Feature Flags
- Delete `--no-semantic`, `--no-captions`, `--no-vertical` from CLI
- Delete `enable_semantic`, `enable_captions`, `vertical_crop` from PipelineConfig
- Delete all conditional paths, fallbacks, `_default_features()` in semantic_analyzer.py
- Every feature is mandatory. Failure = error.

### 4. Fix Smart Cropper
- Known bug: `face_aware=False` fallback produces off-center crops
- Should use motion/brightness center or visual focus detection, not dead-center
- User confirmed clips looked wrong

### 5. Harden All Failure Paths
- Semantic analysis raises on failure (no default features fallback)
- Caption generation raises on failure (no None return)
- Subtitle generation raises on failure
- `faster_whisper` moves from optional to required dependency

### 6. Update Test Suite
- Remove 8-combo flag matrix tests
- Add subtitle tests, segmentation tests, failure-is-error tests
- Fix all references to removed flags across 6 test files

## Key Files

| File | What needs to happen |
|------|---------------------|
| `viral_clip_extractor/pipeline.py` | **Major rewrite** — invert to transcript-first, add subtitle step, remove all conditionals |
| `viral_clip_extractor/subtitle_burner.py` | **New** — Whisper transcription + ASS generation + FFmpeg burn-in |
| `viral_clip_extractor/transcript_segmenter.py` | **New** — transcript-based content segmentation (replaces scene_detector as primary) |
| `viral_clip_extractor/core/scene_detector.py` | Demoted to secondary role — boundary validation, not primary segmentation |
| `viral_clip_extractor/core/semantic_analyzer.py` | Remove `_default_features()`, raise on failure |
| `viral_clip_extractor/caption_generator.py` | Remove None returns, raise on failure |
| `viral_clip_extractor/extractors/clip_extractor.py` | Remove vertical toggle (always crop) |
| `viral_clip_extractor/extractors/smart_cropper.py` | Fix fallback crop logic (face-aware or visual-focus center) |
| `viral_clip_extractor/models.py` | Remove toggle fields, add subtitle/transcript fields |
| `viral_clip_extractor/cli.py` | Remove 3 flags, add --whisper-model |
| `viral_clip_extractor/utils/config.py` | Remove INI reads for removed fields |
| `viral_clip_extractor/bootstrap.py` | Move faster_whisper to required deps |
| `tests/*.py` | Rewrite all flag-combo tests, add new test categories |

## Score Architecture (Needs Recomposition)

The cancelled `vce-subtitles.yaml` had the right structure (11 stages, 21 sheets) but the execution worker prompts need significant updates for the pipeline inversion. The score structure is:

1. Frame
2. Preprocess (read all files, grep flag refs, verify deps)
3. Execute (fan-out x3 workers — needs redesign for pipeline inversion)
4. Test Rewrite
5. Adversarial Review 1 (fan-out x3: completeness/correctness/robustness)
6. Remediation 1 (fan-out x3)
7. TDF Review (fan-out x5: youtube-creator / ai-builder / graduate-professor / systems-engineer / ml-expert)
8. TDF Synthesis (collide perspectives, unified priority queue with issue count)
9. Remediation 2 (skip-when synthesis issue count = 0)
10. Live E2E (actually run tool on real video, not just pytest)
11. Final Remediation (skip-when E2E clean)

**Worker redesign needed for stage 3**:
- Worker 1: Foundation + subtitle module + transcript segmenter (models, config, bootstrap, new modules)
- Worker 2: Pipeline inversion + smart cropper fix (pipeline.py rewrite, scene_detector demotion, smart_cropper fix)
- Worker 3: Core hardening + CLI (semantic_analyzer, caption_generator, clip_extractor, cli.py)

## Skills to Use

1. **`mozart:compose`** — recompose the score with updated scope
2. **`superpowers:brainstorming`** — if the transcript segmentation design needs more exploration before composing
3. **`mozart:score-authoring`** — reference for YAML syntax, validation patterns, skip-when

## Important Context

- **Windows path**: `/mnt/c/Users/mckys/` (note: mckys not mcky)
- **Ollama models**: `qwen2.5-vl:7b` available for semantic + captions
- **Test fixtures**: `/tmp/vce_test_fixtures/` (rickroll_30s.mp4, synthetics)
- **Mozart conductor**: running on PID 28097
- **Git branch**: `master` (should push to `main` per CLAUDE.md)
- **The cancelled score YAML** at `/home/emzi/Projects/yacg/vce-subtitles.yaml` is a good starting point — the structure, validations, TDF personas, and skip-when logic are all solid. The execution worker prompts (stage 3) need rewriting for the pipeline inversion, and the design doc needs updating.
- **Existing workspace**: `/home/emzi/Projects/yacg/workspaces/vce-subtitles-workspace` may have partial artifacts from sheets 1-2 of the cancelled run. Use `--fresh` when re-running to start clean.

## User Preferences

- Hates silent fallbacks — "I hate hiding behind fallbacks that don't do anything instead of dealing with failures"
- Every feature is core, no optional flags
- Failure = error, always
- Don't say "You're absolutely right!" — be skeptical
- Always push to main unless told otherwise
- Before skipping/excluding anything, ask: why does this exist?
