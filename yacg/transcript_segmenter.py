"""
LLM-driven transcript segmentation engine.

Replaces PySceneDetect as the primary source of clip boundaries.
Runs faster-whisper on the full source video to get word-level timestamps,
sends the formatted transcript to Ollama for content-based segment
identification, then refines boundaries by snapping to speech pauses
(stage A) and to local-minima in the audio RMS profile (stage B —
sub-word precision for ASMR/whispered cadence content).
"""

import json
import logging
import math
import re
import struct
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import requests

from yacg.models import (
    SceneSegment,
    SegmentBoundary,
    WordTimestamp,
)

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
# Default Ollama request timeout (seconds).  The Tier-1 cross-domain prompt
# is significantly larger than the round-1-3 prompt (~3500 chars vs ~1000)
# AND the segmenter model is typically larger (qwen3:30b vs qwen3:14b),
# so inference can take 5-15 minutes on a long single ASMR transcript when
# using the larger model.  1200s default accommodates that with headroom;
# can be lowered for short content via constructor arg if added later.
_REQUEST_TIMEOUT = 1200
_SPEECH_PAUSE_THRESHOLD = 0.3  # seconds — gap defining a speech pause
_MIN_SEGMENT_DURATION = 15.0  # seconds
_MAX_SEGMENT_DURATION = 45.0  # seconds


# ----------------------------------------------------------------------------
# Acoustic RMS analysis — stage B of refine_boundaries.
#
# The motivation: whisper's word_start/end timestamps mark the LOUD voiced
# portion of each word, missing leading sibilants (~0.10-0.20s) and trailing
# sustains (~0.10-0.20s).  For ASMR/whispered cadence content where the
# speaker draws words out softly, a "clean" cut at the whisper word
# boundary lands inside the leading aspiration of the next word OR
# truncates the trailing aspiration of the current word.
#
# Stage B snaps each LLM-selected boundary to the nearest LOCAL MINIMUM in
# the audio RMS profile (20ms-bin resolution) within a small search window.
# Critical for ASMR where there is often NO true acoustic silence — only
# relative quiet between sustained phrases.
#
# Round-4 tier-0 work validated this approach against composer ear-checks:
# RMS-snapped boundaries produced 3/3 clean cuts where whisper-word-snapped
# boundaries produced 0/3.
# ----------------------------------------------------------------------------


def _audio_rms_profile(
    video_path: str,
    t_start: float,
    t_end: float,
    bin_ms: int = 20,
    sample_rate: int = 16000,
) -> list[tuple[float, float]]:
    """Extract a window of audio via ffmpeg and compute per-bin RMS in dBFS.

    Args:
        video_path: Source video / audio file path.
        t_start: Window start time (seconds, video time-base).
        t_end: Window end time (seconds).
        bin_ms: RMS bin size in milliseconds (default 20 — fine enough to
            resolve syllable-level transitions in slow ASMR cadence).
        sample_rate: Mono PCM resample rate (default 16000 — sufficient for
            envelope/RMS analysis, ~10x faster than full 48k).

    Returns:
        List of ``(time_seconds, dbfs)`` tuples covering the window.
        ``time_seconds`` is in video time-base.  ``dbfs`` is in [-100, 0],
        with -100 representing pure digital silence.

    Raises:
        RuntimeError: If ffmpeg fails to extract audio.
    """
    if t_end <= t_start:
        return []

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = tf.name
    try:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(t_start), "-i", str(video_path),
            "-t", str(t_end - t_start),
            "-ac", "1", "-ar", str(sample_rate),
            "-c:a", "pcm_s16le", wav_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed during RMS audio extract: {result.stderr[:200]}"
            )

        with wave.open(wav_path, "rb") as w:
            sr = w.getframerate()
            n_frames = w.getnframes()
            data = w.readframes(n_frames)
        if not data:
            return []

        samples = struct.unpack(f"<{len(data) // 2}h", data)
        bin_samples = max(1, int(sr * bin_ms / 1000))

        out: list[tuple[float, float]] = []
        for i in range(0, len(samples), bin_samples):
            chunk = samples[i:i + bin_samples]
            if not chunk:
                break
            sq = sum(s * s for s in chunk) / len(chunk)
            if sq > 0:
                rms = sq ** 0.5
                dbfs = 20 * math.log10(rms / 32768)
            else:
                dbfs = -100.0
            out.append((t_start + (i / sr), dbfs))
        return out
    finally:
        try:
            Path(wav_path).unlink()
        except OSError:
            pass


def _snap_boundary_to_local_minimum(
    video_path: str,
    target_time: float,
    search_radius_s: float = 0.5,
    prefer: str = "after",
    smoothing_bins: int = 3,
) -> float:
    """Snap a boundary to the NEAREST local minimum in the audio RMS profile.

    For ASMR content there is often no true acoustic silence — only LOCAL
    MINIMA in the RMS envelope between sustained phrases.  This function
    extracts a ±``search_radius_s`` audio window around ``target_time`` and
    finds the FIRST local minimum in the chosen direction — the dip
    attached to the boundary word's articulation tail (for END) or leading
    aspiration (for START), NOT the deepest silence further away (which
    would land in next/prior-word territory).

    Round-4 lesson: snapping to the GLOBAL window minimum overshoots into
    inter-word silences (e.g. CLIP 2 END at "control." snapped from 706.66
    to 707.06 — into the silence before "3." — adding 400ms of extra
    silence the composer didn't want).  Local-minimum-first matches the
    hand-tuned v7 picks within ~50ms.

    Args:
        video_path: Source video file.
        target_time: Boundary time (seconds) to snap.
        search_radius_s: Half-width of the search window (default 0.5s).
        prefer: "after" (push the cut later — clip ENDS, scan forward from
            target) or "before" (push the cut earlier — clip STARTS, scan
            backward from target).
        smoothing_bins: Apply a centered moving-average over this many bins
            before searching for minima.  3 = light smoothing (60ms window
            at 20ms bins) — kills spurious single-bin dips while preserving
            real articulation gaps.

    Returns:
        The snapped boundary time (seconds).  Falls back to ``target_time``
        if the audio extract fails or no local minimum is found in window.
    """
    t_low = max(0.0, target_time - search_radius_s)
    t_high = target_time + search_radius_s
    try:
        profile = _audio_rms_profile(video_path, t_low, t_high)
    except (RuntimeError, OSError) as exc:
        logger.warning(
            "Acoustic snap: RMS extract failed at %.2fs: %s — using target unchanged",
            target_time, exc,
        )
        return target_time
    if len(profile) < 3:
        return target_time

    # Apply centered moving-average smoothing to suppress single-bin spikes.
    half = max(1, smoothing_bins // 2)
    smoothed: list[tuple[float, float]] = []
    for i, (t, _) in enumerate(profile):
        lo = max(0, i - half)
        hi = min(len(profile), i + half + 1)
        avg_db = sum(d for _, d in profile[lo:hi]) / (hi - lo)
        smoothed.append((t, avg_db))

    # Find target's index in the profile.
    target_idx = min(
        range(len(smoothed)),
        key=lambda i: abs(smoothed[i][0] - target_time),
    )

    # Walk in the chosen direction looking for the FIRST local minimum.
    # Local minimum = bin strictly lower than both neighbors.
    if prefer == "after":
        # Scan target_idx forward.  Return the first bin that's lower than
        # both its neighbors.
        for i in range(target_idx, len(smoothed) - 1):
            prev_db = smoothed[i - 1][1] if i > 0 else smoothed[i][1] + 1
            cur_db = smoothed[i][1]
            next_db = smoothed[i + 1][1]
            if cur_db <= prev_db and cur_db <= next_db:
                return smoothed[i][0]
        # No local minimum found forward — fall back to the deepest bin
        # in the forward range.
        forward = smoothed[target_idx:]
        if forward:
            return min(forward, key=lambda x: x[1])[0]
    else:  # "before"
        # Scan target_idx backward.
        for i in range(target_idx, 0, -1):
            prev_db = smoothed[i - 1][1]
            cur_db = smoothed[i][1]
            next_db = smoothed[i + 1][1] if i + 1 < len(smoothed) else smoothed[i][1] + 1
            if cur_db <= prev_db and cur_db <= next_db:
                return smoothed[i][0]
        backward = smoothed[: target_idx + 1]
        if backward:
            return min(backward, key=lambda x: x[1])[0]

    return target_time


class TranscriptSegmenter:
    """LLM-driven transcript segmentation engine.

    Replaces PySceneDetect as the primary source of clip boundaries.
    Runs faster-whisper for word-level timestamps, sends transcript to
    Ollama for content-based segmentation, then refines boundaries by
    snapping to speech pauses.

    Args:
        whisper_model: faster-whisper model size (tiny/base/small/medium/large-v3).
        ollama_host: Ollama API base URL.
        model_name: Ollama model name for segmentation.
    """

    def __init__(
        self,
        whisper_model: str = "small",
        ollama_host: str = "http://localhost:11434",
        model_name: str = "qwen2.5-vl:7b",
        whisper_device: str = "auto",
        whisper_compute_type: str = "auto",
        pause_threshold: float = _SPEECH_PAUSE_THRESHOLD,
        min_segment_duration: float = _MIN_SEGMENT_DURATION,
        max_segment_duration: float = _MAX_SEGMENT_DURATION,
        vad_filter: bool = True,
        content_type: str = "",
        channel_description: str = "",
        target_audience: str = "",
        custom_instructions: str = "",
    ) -> None:
        self.whisper_model = whisper_model
        self.ollama_host = ollama_host.rstrip("/")
        self.model_name = model_name
        self.whisper_device = whisper_device
        self.whisper_compute_type = whisper_compute_type
        self.pause_threshold = pause_threshold
        self.min_segment_duration = min_segment_duration
        self.max_segment_duration = max_segment_duration
        # Reuse HTTP connections for Ollama API calls
        self._session = requests.Session()
        # VAD filter removes non-speech segments before transcription.
        # Set to False for ASMR/ambient content where non-speech audio
        # (tapping, scratching, whispering) is the primary content and
        # VAD would incorrectly discard relevant segments.
        self.vad_filter = vad_filter
        self.content_type = content_type
        self.channel_description = channel_description
        self.target_audience = target_audience
        self.custom_instructions = custom_instructions
        # Cached WhisperModel instance — avoids 5-15s reload per video
        # in batch mode.
        self._cached_model = None
        self._cached_model_key: tuple[str, str, str] | None = None

    def full_transcribe(self, video_path: str) -> list[WordTimestamp]:
        """Run faster-whisper on entire video with word-level timestamps.

        Args:
            video_path: Path to the source video file.

        Returns:
            List of WordTimestamp objects covering the entire video.

        Raises:
            RuntimeError: If faster-whisper is not installed.
            RuntimeError: If Whisper returns no words (silent video or failure).
            RuntimeError: If transcription crashes.
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise RuntimeError(
                "faster-whisper is required but not installed"
            )

        try:
            # Resolve "auto" to actual hardware settings
            device = self.whisper_device
            compute_type = self.whisper_compute_type
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"
            if compute_type == "auto":
                compute_type = "float16" if device == "cuda" else "int8"

            # Cache the WhisperModel across calls to avoid 5-15s reload
            # per video in batch mode
            model_key = (self.whisper_model, device, compute_type)
            if self._cached_model is not None and self._cached_model_key == model_key:
                model = self._cached_model
            else:
                # Check if model is cached locally before loading
                self._log_whisper_download_status(self.whisper_model)
                model = WhisperModel(
                    self.whisper_model, device=device, compute_type=compute_type
                )
                self._cached_model = model
                self._cached_model_key = model_key

            try:
                segments, info = model.transcribe(
                    video_path,
                    word_timestamps=True,
                    beam_size=5,
                    vad_filter=self.vad_filter,
                )
            except (IndexError, StopIteration) as exc:
                raise RuntimeError(
                    f"No speech detected in video (no audio track or "
                    f"completely silent): {video_path}"
                ) from exc

            logger.info(
                "Transcribing %s (language: %s, prob: %.2f)",
                video_path,
                info.language,
                info.language_probability,
            )

            # Language detection gating
            if info.language_probability < 0.5:
                raise RuntimeError(
                    f"Language detection confidence too low: "
                    f"detected '{info.language}' with probability "
                    f"{info.language_probability:.2f} (threshold: 0.50). "
                    f"The audio may not contain intelligible speech."
                )
            if info.language_probability < 0.8:
                logger.warning(
                    "Low language detection confidence: %s (%.2f). "
                    "Transcription quality may be degraded.",
                    info.language,
                    info.language_probability,
                )

            words: list[WordTimestamp] = []
            segment_count = 0
            transcribe_start = time.time()
            for segment in segments:
                segment_count += 1
                if segment_count % 10 == 0:
                    elapsed = time.time() - transcribe_start
                    logger.info(
                        "Transcribing... %d segments processed, %d words so far (%.1fs elapsed)",
                        segment_count, len(words), elapsed,
                    )
                try:
                    if segment.words:
                        for w in segment.words:
                            words.append(
                                WordTimestamp(
                                    word=w.word.strip(),
                                    start=w.start,
                                    end=w.end,
                                    probability=w.probability,
                                )
                            )
                except (IndexError, StopIteration) as exc:
                    raise RuntimeError(
                        f"No speech detected in video (no audio track or "
                        f"completely silent): {video_path}"
                    ) from exc

            if not words:
                raise RuntimeError(
                    f"No speech detected in video: {video_path}"
                )

            logger.info(
                "Transcription complete: %d words, language=%s (%.1f%%)",
                len(words),
                info.language,
                info.language_probability * 100,
            )
            return words

        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Whisper transcription failed: {exc}"
            ) from exc

    def segment_by_content(
        self, words: list[WordTimestamp], title: str,
        target_count: int = 20,
    ) -> list[SegmentBoundary]:
        """Send transcript to Ollama for LLM-driven segment identification.

        Formats word-level timestamps into a timestamped transcript, sends to
        Ollama with a segmentation prompt, and parses the JSON array response.
        Validates returned timestamps against the transcript time range and
        clamps out-of-range values.

        Args:
            words: Word-level timestamps from full_transcribe().
            title: Video title for context.
            target_count: Target number of segments to request from the LLM.

        Returns:
            List of SegmentBoundary objects with LLM-identified segments.

        Raises:
            RuntimeError: If Ollama fails after retries.
            RuntimeError: If Ollama returns unparseable response after retries.
            RuntimeError: If LLM finds no viral-worthy segments.
        """
        transcript_text = self._format_transcript(words)
        prompt = self._create_segmentation_prompt(
            transcript_text, title, target_count=target_count,
        )

        # Determine valid time range from transcript
        total_duration = words[-1].end if words else 0.0

        last_error: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response_text = self._query_ollama(prompt)
                boundaries = self._parse_segmentation_response(response_text)
                if boundaries:
                    # Validate timestamps against transcript range
                    boundaries = self._validate_timestamps(
                        boundaries, total_duration,
                    )
                    logger.info(
                        "LLM identified %d segments (attempt %d)",
                        len(boundaries),
                        attempt,
                    )
                    return boundaries
                # Empty list — retry with a different attempt
                last_error = RuntimeError(
                    "LLM returned empty segment list"
                )
                logger.warning(
                    "LLM returned no segments on attempt %d/%d",
                    attempt,
                    _MAX_RETRIES,
                )
            except RuntimeError as exc:
                last_error = exc
                logger.warning(
                    "Segmentation attempt %d/%d failed: %s",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )

            if attempt < _MAX_RETRIES:
                self._backoff(attempt)

        # All retries exhausted
        if last_error and "returned empty" not in str(last_error):
            error_msg = str(last_error)
            if "connection failed" in error_msg.lower() or "timed out" in error_msg.lower() or "invalid" in error_msg.lower():
                raise RuntimeError(
                    f"Ollama connection failed after {_MAX_RETRIES} attempts"
                ) from last_error
            raise RuntimeError(
                f"Failed to parse Ollama segmentation response "
                f"after {_MAX_RETRIES} attempts"
            ) from last_error

        raise RuntimeError(
            "LLM found no viral-worthy segments in transcript"
        )

    def refine_boundaries(
        self,
        segments: list[SegmentBoundary],
        words: list[WordTimestamp],
        video_path: str | None = None,
        acoustic_snap_radius_s: float = 0.5,
    ) -> list[SceneSegment]:
        """Snap segment boundaries to nearest speech pauses.

        Two-stage refinement:

          * **Stage A (semantic)**: Adjusts LLM-suggested boundaries to fall on
            speech pauses (gaps > pause_threshold between words) so clips don't
            cut mid-word.  Validates no overlaps, enforces min/max duration.
          * **Stage B (acoustic, OPT-IN)**: When ``video_path`` is supplied,
            snaps each refined boundary to the nearest LOCAL MINIMUM in the
            audio RMS profile within ±``acoustic_snap_radius_s``.  Necessary
            for ASMR/whispered cadence content where whisper word_start/end
            timestamps mark only the LOUD voiced portion of each word and miss
            leading sibilants + trailing sustains.  See the module-level
            docstring on `_audio_rms_profile` for the round-4 tier-0
            validation that motivates this stage.

        Args:
            segments: LLM-identified segment boundaries.
            words: Full transcript word timestamps.
            video_path: Optional path to the source video.  When provided,
                stage B (acoustic RMS snap) runs.  When None, only stage A
                runs (preserves pre-round-4 behavior).
            acoustic_snap_radius_s: Half-width of the stage-B search window
                (default 0.5s).  Larger windows risk snapping to a different
                gap; smaller windows risk missing the local minimum.

        Returns:
            List of SceneSegment objects with refined boundaries.
        """
        if not words:
            return []

        pauses = self._find_speech_pauses(words)
        video_start = words[0].start
        video_end = words[-1].end

        refined: list[tuple[float, float]] = []
        for seg in segments:
            start = self._snap_to_nearest_pause(
                seg.start_time, pauses, prefer="before"
            )
            end = self._snap_to_nearest_pause(
                seg.end_time, pauses, prefer="after"
            )

            # Clamp to video bounds
            start = max(video_start, start)
            end = min(video_end, end)

            # Ensure minimum duration.  When the LLM picks a tight clip
            # whose pause-snapped end falls short of min_segment_duration,
            # we need to extend.  The OLD code added the deficit
            # mechanically (`end + needed`), which puts the new end
            # wherever it lands — frequently mid-clause or mid-word.  The
            # NEW code snaps the extended end to the nearest sentence
            # boundary at or after the target time, only falling back to
            # mechanical extension when no usable pause exists.
            duration = end - start
            if duration < self.min_segment_duration:
                target_end = min(video_end, start + self.min_segment_duration)
                snapped_end = self._snap_to_nearest_pause(
                    target_end, pauses, prefer="after"
                )
                # Snap may overshoot max_segment_duration; cap to
                # max+small slack so a sentence break can still land
                # cleanly past max.  If snap returns the unchanged target
                # (no pause found within the search window), fall back to
                # mechanical extension at the target time.
                slack_max = start + self.max_segment_duration
                if snapped_end > target_end and snapped_end <= slack_max:
                    end = snapped_end
                elif snapped_end > slack_max:
                    end = target_end
                else:
                    end = snapped_end
                end = min(video_end, end)
                duration = end - start

            # Enforce maximum duration.  Find a pause BEFORE the max
            # duration mark to stay within bounds; the existing snap
            # function already prefers sentence-ending pauses.
            if duration > self.max_segment_duration:
                target_end = start + self.max_segment_duration
                end = self._snap_to_nearest_pause(
                    target_end, pauses, prefer="before"
                )
                # If snap pulled us below min, accept the snap (a clean
                # sentence break shorter than max is still preferable to a
                # mid-clause cut at exactly max).  Hard-cap only if the
                # snap also dropped below min (no good pause within range).
                if end - start < self.min_segment_duration:
                    end = start + self.max_segment_duration
                end = min(video_end, end)

            # Skip segments that are still too short after adjustment
            if end - start < self.min_segment_duration:
                logger.warning(
                    "Dropping segment %.1f-%.1f: too short (%.1fs)",
                    start,
                    end,
                    end - start,
                )
                continue

            refined.append((start, end))

        # Resolve overlaps: sort by start time and adjust shared boundaries
        refined.sort(key=lambda x: x[0])
        non_overlapping: list[tuple[float, float]] = []
        for start, end in refined:
            if non_overlapping and start < non_overlapping[-1][1]:
                prev_start, prev_end = non_overlapping[-1]
                # Find midpoint pause between overlapping segments
                midpoint = (start + prev_end) / 2
                mid_pause = self._snap_to_nearest_pause(
                    midpoint, pauses, prefer="after"
                )
                # Check if adjusting would shrink previous segment too much
                if mid_pause - prev_start < self.min_segment_duration:
                    # Drop current segment to preserve previous
                    logger.warning(
                        "Dropping overlapping segment %.1f-%.1f: "
                        "would shrink previous below minimum",
                        start,
                        end,
                    )
                    continue
                # Adjust previous segment's end and current segment's start
                non_overlapping[-1] = (prev_start, mid_pause)
                start = mid_pause
                if end - start < self.min_segment_duration:
                    continue
            non_overlapping.append((start, end))

        # ---- Stage B: acoustic RMS local-minimum snap (opt-in) ----
        # When the caller supplies a video_path, each stage-A boundary is
        # snapped to the nearest local minimum in the audio RMS profile
        # within ±acoustic_snap_radius_s.  Critical for ASMR/whispered
        # cadence content where whisper word_start/end timestamps miss
        # leading sibilants and trailing sustains.  Skipped when no video
        # path supplied — preserves pre-round-4 behavior.
        if video_path is not None and non_overlapping:
            acoustic_snapped: list[tuple[float, float]] = []
            for stage_a_start, stage_a_end in non_overlapping:
                snapped_start = _snap_boundary_to_local_minimum(
                    video_path,
                    stage_a_start,
                    search_radius_s=acoustic_snap_radius_s,
                    prefer="before",
                )
                snapped_end = _snap_boundary_to_local_minimum(
                    video_path,
                    stage_a_end,
                    search_radius_s=acoustic_snap_radius_s,
                    prefer="after",
                )
                # Clamp to video bounds, preserve min duration
                snapped_start = max(video_start, snapped_start)
                snapped_end = min(video_end, snapped_end)
                if snapped_end - snapped_start < self.min_segment_duration:
                    # Acoustic snap pulled below min — fall back to stage-A
                    # boundaries for this segment.
                    logger.debug(
                        "Acoustic snap dropped %.2f-%.2f below min duration "
                        "(%.2fs); reverting to stage-A (%.2f-%.2f)",
                        snapped_start, snapped_end,
                        snapped_end - snapped_start,
                        stage_a_start, stage_a_end,
                    )
                    acoustic_snapped.append((stage_a_start, stage_a_end))
                else:
                    if (abs(snapped_start - stage_a_start) > 0.01
                            or abs(snapped_end - stage_a_end) > 0.01):
                        logger.info(
                            "Acoustic snap: %.2f→%.2f  %.2f→%.2f "
                            "(deltas %+.3fs / %+.3fs)",
                            stage_a_start, snapped_start,
                            stage_a_end, snapped_end,
                            snapped_start - stage_a_start,
                            snapped_end - stage_a_end,
                        )
                    acoustic_snapped.append((snapped_start, snapped_end))
            non_overlapping = acoustic_snapped

        # Convert to SceneSegment objects
        result: list[SceneSegment] = []
        for idx, (start, end) in enumerate(non_overlapping):
            result.append(
                SceneSegment(
                    start_time=start,
                    end_time=end,
                    scene_index=idx,
                )
            )

        logger.info(
            "Refined %d LLM segments → %d scene segments",
            len(segments),
            len(result),
        )
        return result

    def segment_video(
        self, video_path: str, title: str
    ) -> tuple[list[SceneSegment], list[WordTimestamp]]:
        """High-level entry point: transcribe → segment → refine.

        Args:
            video_path: Path to the source video file.
            title: Video title for LLM context.

        Returns:
            Tuple of (refined scene segments, all word timestamps).
            The word timestamps are returned for reuse by subtitle burning.
        """
        words = self.full_transcribe(video_path)
        boundaries = self.segment_by_content(words, title)
        # Pass video_path through so refine_boundaries can run stage B
        # (acoustic RMS local-minimum snap).  Round-4 tier-0 work proved
        # this is necessary to land cuts on actual articulation
        # boundaries vs whisper's loud-portion-only word_start/end.
        scenes = self.refine_boundaries(boundaries, words, video_path=video_path)
        return scenes, words

    def _format_transcript(
        self, words: list[WordTimestamp], block_seconds: float = 10.0,
        section_count: int = 3,
    ) -> str:
        """Format words into timestamped text blocks for the LLM prompt.

        Groups words into blocks of approximately block_seconds duration,
        producing lines like: [00:18.08 - 00:28.00] We're no strangers to love ...

        When ``section_count > 1``, the transcript is also pre-split into
        equal-duration SECTIONS (default 3 thirds: early/middle/late) with
        explicit headers between them.  This counters the LLM
        lost-in-the-middle bias where models pick all clips from the early
        portion of long transcripts; explicit section markers force the
        model to consider each section as a distinct selection scope.
        Round-4 evidence: with no section markers, qwen3:14b and qwen3:30b
        both clustered all picks in the first 17% of a 23-min source.
        """
        if not words:
            return ""

        # Build all blocks first (single pass over words).
        blocks: list[tuple[float, float, str]] = []
        block_start = words[0].start
        block_words: list[str] = []

        for idx, w in enumerate(words):
            if w.start - block_start >= block_seconds and block_words:
                block_end = words[idx - 1].end if idx > 0 else w.start
                blocks.append((block_start, block_end, " ".join(block_words)))
                block_words = []
                block_start = w.start
            block_words.append(w.word)

        if block_words:
            block_end = words[-1].end
            blocks.append((block_start, block_end, " ".join(block_words)))

        # IMPORTANT: timestamps formatted as raw SECONDS (decimal floats)
        # not MM:SS — round-4 evidence showed qwen3:14b copies the input
        # timestamp format into its JSON output; MM:SS strings break JSON
        # parsing because `03:44.78` isn't a valid JSON number (the colon
        # is interpreted as a key/value separator).  Raw seconds in the
        # input + explicit "use seconds" prompt instruction = LLM outputs
        # parseable JSON with second-precision timestamps.
        if section_count <= 1:
            return "\n".join(
                f"[{s:.2f} - {e:.2f}] {t}"
                for s, e, t in blocks
            )

        # Compute section boundaries by total source duration.
        source_start = words[0].start
        source_end = words[-1].end
        section_dur = (source_end - source_start) / section_count
        section_labels = (
            ["EARLY", "MIDDLE", "LATE"] if section_count == 3
            else [f"SECTION {i + 1} of {section_count}" for i in range(section_count)]
        )

        lines: list[str] = []
        current_section = -1
        for s, e, t in blocks:
            block_section = min(
                section_count - 1,
                int((s - source_start) // section_dur),
            )
            if block_section != current_section:
                current_section = block_section
                label = section_labels[block_section]
                section_start = source_start + block_section * section_dur
                section_end = source_start + (block_section + 1) * section_dur
                lines.append("")
                lines.append(
                    f"=== {label} THIRD "
                    f"({section_start:.2f}s - {section_end:.2f}s) "
                    f"===  ← pick AT LEAST ONE clip from this section"
                )
            lines.append(f"[{s:.2f} - {e:.2f}] {t}")
        return "\n".join(lines).strip()

    def _format_transcript_legacy(
        self, words: list[WordTimestamp], block_seconds: float = 10.0
    ) -> str:
        """LEGACY no-sections formatter — kept for backward-compat callers."""
        if not words:
            return ""

        lines: list[str] = []
        block_start = words[0].start
        block_words: list[str] = []

        for idx, w in enumerate(words):
            if w.start - block_start >= block_seconds and block_words:
                block_end = words[idx - 1].end if idx > 0 else w.start
                lines.append(
                    f"[{self._fmt_time(block_start)} - "
                    f"{self._fmt_time(block_end)}] "
                    f"{' '.join(block_words)}"
                )
                block_words = []
                block_start = w.start

            block_words.append(w.word)

        # Flush remaining words
        if block_words:
            block_end = words[-1].end
            lines.append(
                f"[{self._fmt_time(block_start)} - "
                f"{self._fmt_time(block_end)}] "
                f"{' '.join(block_words)}"
            )

        return "\n".join(lines)

    def _find_speech_pauses(
        self, words: list[WordTimestamp]
    ) -> list[tuple[float, float, bool]]:
        """Find all speech pauses (gaps > pause_threshold) in the word list.

        Returns:
            List of (pause_start, pause_end, is_sentence_end) tuples where
            pause_start is the end time of the word before the gap,
            pause_end is the start time of the word after the gap, and
            is_sentence_end is True under any of:
              - the previous word ends with sentence punctuation (. ? !)
              - the gap is long enough to imply a sentence break (threshold
                varies by content type — see below)
              - the next word begins capitalized AND the gap exceeds
                ``cap_pause_threshold`` (also content-type-aware)

        faster-whisper's word-level stream often omits trailing punctuation
        on the individual word (punctuation lives at segment level), so a
        pure punctuation check rarely fires and downstream snap-to-sentence
        logic degenerates to "any small pause."

        Content-type-aware thresholds:

          * General/default: long_pause >= 0.5s, capitalized-next >= 0.35s.
            Fits most narrative speech where pauses >0.5s are sentence
            boundaries.
          * ASMR (or any content_type containing "asmr"): long_pause >= 1.5s,
            capitalized-next >= 1.0s. ASMR delivery is FULL of dramatic
            mid-sentence pauses (0.5-1.4s) used for emphasis, breath, and
            sleep induction. Treating those as sentence ends causes the
            snap to commit to mid-clause boundaries.

        The thresholds intentionally over-correct: better to miss a few
        real sentence ends than to falsely mark dramatic pauses as
        sentence ends and cut clips mid-thought.
        """
        ct = (self.content_type or "").lower()
        if "asmr" in ct:
            long_pause_threshold = 1.5
            cap_pause_threshold = 1.0
        else:
            long_pause_threshold = 0.5
            cap_pause_threshold = 0.35

        pauses: list[tuple[float, float, bool]] = []
        for i in range(len(words) - 1):
            gap = words[i + 1].start - words[i].end
            if gap > self.pause_threshold:
                prev_word = words[i].word.strip()
                next_word = words[i + 1].word.strip()
                ends_in_punct = bool(prev_word) and prev_word[-1] in ".?!"
                is_long_pause = gap >= long_pause_threshold
                next_starts_cap = (
                    bool(next_word)
                    and next_word[0].isupper()
                    and not next_word[0].isdigit()
                )
                is_sentence_end = (
                    ends_in_punct
                    or is_long_pause
                    or (next_starts_cap and gap >= cap_pause_threshold)
                )
                pauses.append((words[i].end, words[i + 1].start, is_sentence_end))
        return pauses

    def _snap_to_nearest_pause(
        self,
        target_time: float,
        pauses: list[tuple[float, float, bool]],
        prefer: str = "before",
    ) -> float:
        """Snap a timestamp to the nearest speech pause boundary.

        When both a sentence-ending pause and a mid-sentence pause exist
        within range, prefers the sentence-ending one (within a 5s
        tolerance window). Falls back to the closest any-type pause.

        Args:
            target_time: The time to snap.
            pauses: List of (pause_start, pause_end, is_sentence_end) tuples.
            prefer: 'before' snaps to the pause end before target (clip
                starts when speech resumes). 'after' snaps to the pause
                start after target (clip ends when speech stops).

        Returns:
            The snapped time, or target_time unchanged if no suitable pause.
        """
        _SENTENCE_TOLERANCE = 5.0  # seconds

        if not pauses:
            return target_time

        if prefer == "before":
            # Find pause whose end is closest to and <= target_time
            best: float | None = None
            best_dist = float("inf")
            best_sentence: float | None = None
            best_sentence_dist = float("inf")
            for pause_start, pause_end, is_sentence_end in pauses:
                if pause_end <= target_time:
                    dist = target_time - pause_end
                    if dist < best_dist:
                        best_dist = dist
                        best = pause_end
                    if is_sentence_end and dist < best_sentence_dist:
                        best_sentence_dist = dist
                        best_sentence = pause_end
            # Prefer sentence-ending pause if within tolerance
            if best_sentence is not None and best_sentence_dist <= _SENTENCE_TOLERANCE:
                return best_sentence
            return best if best is not None else target_time

        else:  # prefer == "after"
            best = None
            best_dist = float("inf")
            best_sentence: float | None = None
            best_sentence_dist = float("inf")
            for pause_start, pause_end, is_sentence_end in pauses:
                if pause_start >= target_time:
                    dist = pause_start - target_time
                    if dist < best_dist:
                        best_dist = dist
                        best = pause_start
                    if is_sentence_end and dist < best_sentence_dist:
                        best_sentence_dist = dist
                        best_sentence = pause_start
            # Prefer sentence-ending pause if within tolerance
            if best_sentence is not None and best_sentence_dist <= _SENTENCE_TOLERANCE:
                return best_sentence
            return best if best is not None else target_time

    def _create_segmentation_prompt(
        self, transcript_text: str, title: str,
        target_count: int = 20,
    ) -> str:
        """Build the LLM segmentation prompt with the transcript.

        Cross-domain prompt design (round-4 approach): the LLM is asked to
        SELECT clips using domain-specific reasoning about content
        structure, virality, and editing rhythm — not just "find viral
        segments" with a vague target.  The prompt explicitly invokes:

          * Content-structure knowledge (hypnosis induction shape, gaming
            highlight pattern, cooking technique arc, etc — per content_type)
          * Virality patterns (hook-first, complete payoff, clean end beat)
          * Editing principles (cut on completion, hold on resolution)
          * Brand context (channel description, target audience, custom
            instructions, optional creator notes)

        Mandatory rationale field per clip — when the LLM cannot articulate
        why a boundary is right, the boundary is probably wrong; the
        rationale lets composers debug bad picks.

        Round-4 evidence: round-1-to-3 prompts asked for "viral segments"
        and the LLM picked arbitrary 25-second windows because nothing in
        the prompt told it what coherence looks like for the target
        content type.
        """
        title_line = f' titled "{title}"' if title else ""
        ct = (self.content_type or "general").lower()

        # Content-type-aware guidance for what makes a "good clip" — also
        # invokes content-specific structural patterns the model should
        # recognize and respect when picking boundaries.
        content_guidance = self._get_content_type_guidance()
        structural_guidance = self._get_structural_guidance(ct)

        # Brand context sections
        context_parts: list[str] = []
        if self.channel_description:
            context_parts.append(
                f"CHANNEL CONTEXT: {self.channel_description}"
            )
        if self.target_audience:
            context_parts.append(
                f"TARGET AUDIENCE: {self.target_audience}"
            )
        context_block = ""
        if context_parts:
            context_block = "\n".join(context_parts) + "\n\n"

        custom_block = ""
        if self.custom_instructions:
            custom_block = (
                f"ADDITIONAL INSTRUCTIONS FROM CREATOR:\n"
                f"{self.custom_instructions}\n\n"
            )

        return (
            f"You are SELECTING clips from a long-form video{title_line} "
            f"for posting to TikTok, Instagram Reels, and YouTube Shorts. "
            f"Each clip must be a complete, self-contained piece of work "
            f"that lands as intended on the brand's audience.\n"
            f"\n"
            f"# Brand context\n"
            f"\n"
            f"{context_block}"
            f"{custom_block}"
            f"# What you're picking from\n"
            f"\n"
            f"The transcript below is timestamped, with each line "
            f"representing a span of speech and its time range:\n"
            f"\n"
            f"{transcript_text}\n"
            f"\n"
            f"# Your task\n"
            f"\n"
            f"SELECT 5-10 of the BEST clips from this transcript — the most "
            f"compelling moments scattered ACROSS THE WHOLE VIDEO, not "
            f"contiguous coverage of one section.  Each clip must:\n"
            f"\n"
            f"- Be {self.min_segment_duration:.0f}-"
            f"{self.max_segment_duration:.0f} seconds long\n"
            f"- Be a COMPLETE coherent unit (not a fragment, not a teaser, "
            f"not a mid-thought cut)\n"
            f"- NOT overlap with other selected clips\n"
            f"- NOT be adjacent to other selected clips (leave gaps between "
            f"clips — adjacent picks usually mean the LLM is chopping one "
            f"long passage into pieces; instead pick the SINGLE BEST passage "
            f"and move on to find the next best moment elsewhere)\n"
            f"- Use timestamps from the transcript above for start_time / "
            f"end_time\n"
            f"\n"
            f"You are picking 5-10 STANDOUT MOMENTS, not summarizing or "
            f"covering the video.  Most of the video should NOT be in any "
            f"clip.\n"
            f"\n"
            f"DISTRIBUTION: The transcript above is split into "
            f"=== EARLY / MIDDLE / LATE THIRD === sections.  You MUST pick "
            f"AT LEAST ONE clip from EACH section.  Even if the early third "
            f"has the best opening hook, the middle and late thirds contain "
            f"the development and payoff of the script — the most "
            f"emotionally resonant moments are typically NOT in the first "
            f"third.  Read each section to its end before deciding.\n"
            f"\n"
            f"# What makes a good clip for this content\n"
            f"\n"
            f"{content_guidance}\n"
            f"\n"
            f"# Structural patterns to recognize and respect\n"
            f"\n"
            f"{structural_guidance}\n"
            f"\n"
            f"# Universal short-form viral principles\n"
            f"\n"
            f"- HOOK in first 2-3 seconds — opening line creates curiosity "
            f"or sensory reaction\n"
            f"- PATTERN INTERRUPT or unexpected shift somewhere in the "
            f"middle — makes the viewer commit to watching to the end\n"
            f"- COMPLETE PAYOFF — the suggestion lands, the trigger fires, "
            f"the punchline delivers; the viewer feels they got something\n"
            f"- CLEAN END BEAT — not cut mid-breath, not on a cliffhanger "
            f"that forces follow-up; lands on a resolution moment\n"
            f"\n"
            f"# Editing rhythm\n"
            f"\n"
            f"- CUT ON COMPLETION of a thought, not in the middle\n"
            f"- HOLD on resolution beats — don't truncate the last word of "
            f"a key moment\n"
            f"- NEVER cut in the middle of a key delivery (suggestion, "
            f"punchline, reveal)\n"
            f"\n"
            f"# Output format\n"
            f"\n"
            f"Output ONLY a valid JSON array with NO additional text.\n"
            f"\n"
            f"start_time and end_time MUST be decimal SECONDS (e.g. 12.4, "
            f"not 00:12.40 or 0:12).  The timestamps in the transcript "
            f"above are also in seconds — copy them as plain decimal "
            f"numbers into your JSON.  MM:SS format will break JSON "
            f"parsing.\n"
            f"\n"
            f"segment_type must be one of: hook, narrative_arc, "
            f"complete_thought, emotional_peak\n"
            f"The hook_summary field is mandatory — one sentence "
            f"describing what makes this clip land.  If you cannot "
            f"articulate why a boundary is right, the boundary is "
            f"probably wrong.\n"
            f"\n"
            f"Example:\n"
            f'[{{"start_time": 12.4, "end_time": 47.8, '
            f'"hook_summary": "what makes this clip land", '
            f'"segment_type": "hook"}}]\n'
        )

    def _get_structural_guidance(self, content_type: str) -> str:
        """Return content-type-specific structural patterns for the LLM.

        Names the recognizable structures for each content type so the
        LLM doesn't cut across them.  For ASMR/hypnosis specifically:
        induction shape, fractionation cycles, trigger implantation,
        nested loops — well-defined patterns the LLM should respect when
        choosing boundaries.
        """
        ct = (content_type or "general").lower()
        if "asmr" in ct or "hypno" in ct:
            return (
                "Hypnosis ASMR has identifiable structural shapes:\n"
                "- INDUCTION ARC: setup → induction → deepener → suggestion → trigger → close. "
                "Cutting mid-induction loses the setup that primes the listener; cutting "
                "before the suggestion lands wastes the induction work.\n"
                "- FRACTIONATION CYCLES: deepen → return-toward-surface → deepen further. "
                "Each cycle is a unit; mid-cycle cuts leave the listener stuck mid-transition.\n"
                "- TRIGGER IMPLANTATION: setup ('when I say X, you'll Y') → implantation → "
                "test. Cut the test out and the trigger doesn't anchor.\n"
                "- NESTED LOOPS: opening multiple suggestion frames before closing them. "
                "Nesting INSIDE a clip is fine; opening a loop without closing it is broken.\n"
                "- COMFORT / SAFETY language at clip endings matters — listeners use this "
                "for sleep and trance. Ending on an open hook with no comfort beat is jarring.\n"
            )
        elif "gaming" in ct:
            return (
                "Gaming highlights have identifiable structural shapes:\n"
                "- BUILDUP: positioning → engagement → climax → resolution\n"
                "- REACTION arcs: surprise/triumph beats need full setup + reaction tail\n"
                "- COMMENTARY peaks: complete thoughts; cutting mid-joke kills the bit\n"
            )
        elif "cooking" in ct:
            return (
                "Cooking content has identifiable structural shapes:\n"
                "- TECHNIQUE arc: setup → execution → reveal\n"
                "- HOOK + payoff: surprising ingredient → satisfying result\n"
            )
        elif "educational" in ct:
            return (
                "Educational content has identifiable structural shapes:\n"
                "- INSIGHT arc: question/setup → explanation → aha-moment\n"
                "- HOOK + reveal: counterintuitive claim → evidence → conclusion\n"
            )
        else:
            return (
                "Find complete narrative units:\n"
                "- HOOK + DEVELOPMENT + RESOLUTION arcs\n"
                "- COMPLETE THOUGHTS that stand alone without surrounding context\n"
                "- EMOTIONAL PEAKS with full setup and reaction\n"
            )

    def _get_content_type_guidance(self) -> str:
        """Return content-type-specific guidance for segment identification."""
        ct = self.content_type.lower()

        if ct == "gaming":
            return (
                "- HIGHLIGHT PLAYS: Kills, wins, clutch moments, epic fails\n"
                "- REACTIONS: Genuine surprise, rage, excitement, celebration\n"
                "- COMMENTARY PEAKS: Funny or insightful commentary moments\n"
                "- NARRATIVE ARCS: Mini-stories with setup and payoff "
                "(clutch rounds, comebacks)\n"
            )
        elif ct == "cooking":
            return (
                "- COMPLETE TECHNIQUES: A full cooking technique from start "
                "to finish (15-45 seconds)\n"
                "- SATISFYING MOMENTS: Plating, sizzling, pouring, "
                "revealing finished dishes\n"
                "- TIPS & TRICKS: Quick actionable cooking advice\n"
                "- HOOKS: Surprising ingredients, unexpected methods, "
                "\"secret\" techniques\n"
            )
        elif ct == "educational":
            return (
                "- KEY INSIGHTS: Core ideas explained clearly and concisely\n"
                "- AHA MOMENTS: Surprising facts, counterintuitive "
                "explanations, mind-blowing connections\n"
                "- HOOKS: Provocative questions, bold claims, curiosity gaps\n"
                "- COMPLETE EXPLANATIONS: Self-contained ideas that make "
                "sense without surrounding context\n"
            )
        elif ct == "asmr":
            return (
                "- TRIGGER SEQUENCES: Concentrated ASMR triggers (tapping, "
                "scratching, whispering)\n"
                "- IMMERSIVE MOMENTS: Deeply relaxing, tingle-inducing "
                "passages\n"
                "- HOOKS: Unique or unexpected sounds that grab attention\n"
                "- COMPLETE SEQUENCES: Full trigger sequences that don't "
                "cut mid-action\n"
            )
        elif ct == "fitness":
            return (
                "- EXERCISE DEMOS: Complete exercise demonstrations with "
                "form cues\n"
                "- MOTIVATIONAL PEAKS: Inspirational moments, personal "
                "bests, transformation reveals\n"
                "- TIPS & FORM CHECKS: Quick actionable fitness advice\n"
                "- HOOKS: Impressive feats, before/after moments, "
                "challenge setups\n"
            )
        elif ct == "music":
            return (
                "- PERFORMANCE PEAKS: Best vocal/instrumental moments\n"
                "- HOOKS: Catchy melodies, impressive riffs, unexpected "
                "harmonies\n"
                "- COMPLETE PHRASES: Musical phrases that form a "
                "satisfying unit\n"
                "- EMOTIONAL PEAKS: Moments of high musical intensity "
                "or beauty\n"
            )
        elif ct == "comedy":
            return (
                "- PUNCHLINES: Complete joke setups with payoffs\n"
                "- REACTIONS: Genuine laughter, surprise, or absurdity\n"
                "- HOOKS: Opening lines that set up curiosity or absurdity\n"
                "- COMPLETE BITS: Self-contained comedic moments\n"
            )
        else:
            # General/default — works for any content type
            return (
                "- HOOKS: Opening statements that grab attention immediately\n"
                "- COMPLETE THOUGHTS: Self-contained ideas or stories "
                "(15-45 seconds)\n"
                "- NARRATIVE ARCS: Mini-stories with setup and payoff\n"
                "- EMOTIONAL PEAKS: Moments of high emotion, surprise, or "
                "humor\n"
            )

    def _query_ollama(self, prompt: str) -> str:
        """Send a text-only prompt to Ollama and return the response text.

        Raises:
            RuntimeError: On connection failure or non-200 response.
        """
        url = f"{self.ollama_host}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                # Low temperature (0.3) for deterministic JSON output.
                # Segmentation needs precise, reproducible timestamps — not
                # creative variation. (Contrast with caption_generator's 0.7
                # which benefits from creative diversity.)
                "temperature": 0.3,
                "top_p": 0.9,
            },
        }

        try:
            resp = self._session.post(
                url, json=payload, timeout=_REQUEST_TIMEOUT
            )
        except requests.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama. Please start it with "
                f"'ollama serve' and ensure it is running at "
                f"{self.ollama_host}"
            )
        except requests.Timeout:
            raise RuntimeError(
                f"Ollama request timed out after {_REQUEST_TIMEOUT}s"
            )
        except (requests.exceptions.MissingSchema, requests.exceptions.InvalidURL):
            raise RuntimeError(
                f"Invalid Ollama URL: {self.ollama_host}"
            )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama returned status {resp.status_code}"
            )

        data = resp.json()
        return data.get("response", "")

    def _parse_segmentation_response(
        self, response_text: str
    ) -> list[SegmentBoundary]:
        """Parse the LLM's JSON array response into SegmentBoundary objects.

        Extracts a JSON array from the response text (handles markdown
        code fences and surrounding text), validates required fields,
        and converts to SegmentBoundary objects.

        Returns:
            List of SegmentBoundary objects, or empty list if parsing fails.
        """
        # Strip markdown code fences if present
        text = response_text.strip()
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        text = text.strip()

        # Find JSON array in the text
        start_idx = text.find("[")
        end_idx = text.rfind("]")
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            logger.warning(
                "No JSON array found in LLM response: %.200s", text
            )
            return []

        json_str = text[start_idx : end_idx + 1]

        try:
            items = json.loads(json_str)
        except json.JSONDecodeError as exc:
            # Defensive fallback: some LLMs (qwen3:14b observed in round 4)
            # output timestamps as MM:SS strings (e.g. `03:44.78`) instead
            # of decimal seconds.  These break JSON because the colon is
            # interpreted as a key/value separator.  Convert MM:SS or H:MM:SS
            # patterns inside numeric value positions to decimal seconds and
            # retry parsing.
            mmss_pattern = re.compile(
                r'("(?:start_time|end_time)":\s*)(\d{1,2}):(\d{2})(?:[.:](\d{1,3}))?'
            )
            def _to_seconds(m: re.Match[str]) -> str:
                key_part = m.group(1)
                mm = int(m.group(2))
                ss = int(m.group(3))
                ms_str = m.group(4) or ""
                ms = int(ms_str) / (10 ** len(ms_str)) if ms_str else 0.0
                return f"{key_part}{mm * 60 + ss + ms:.2f}"
            json_str_fixed = mmss_pattern.sub(_to_seconds, json_str)
            if json_str_fixed != json_str:
                try:
                    items = json.loads(json_str_fixed)
                    logger.info(
                        "JSON parse: recovered from MM:SS-format timestamps via "
                        "regex fallback (%d substitutions)",
                        len(mmss_pattern.findall(json_str)),
                    )
                except json.JSONDecodeError as exc2:
                    logger.warning(
                        "Failed to parse JSON even after MM:SS fixup: %s", exc2,
                    )
                    logger.warning(
                        "Raw LLM response (first 500 chars): %r", text[:500]
                    )
                    return []
            else:
                logger.warning("Failed to parse JSON from LLM response: %s", exc)
                logger.warning(
                    "Raw LLM response (first 500 chars): %r", text[:500]
                )
                logger.warning(
                    "Extracted json_str (first 500 chars): %r", json_str[:500]
                )
                return []

        if not isinstance(items, list):
            logger.warning("LLM response is not a JSON array")
            return []

        boundaries: list[SegmentBoundary] = []
        core_keys = {"start_time", "end_time", "segment_type"}
        # LLMs frequently rename "hook_summary" to a synonym — accept common
        # alternatives rather than silently dropping valid segments.
        # The model also generates "{segment_type}_summary" keys (e.g.
        # "complete_thought_summary", "emotional_peak_summary"), so we
        # also check for any key ending in "_summary".
        _summary_aliases = ("hook_summary", "summary", "description", "hook_description")
        for item in items:
            if not isinstance(item, dict):
                continue
            if not core_keys.issubset(item.keys()):
                logger.warning(
                    "Segment missing required keys: %s",
                    core_keys - item.keys(),
                )
                continue
            # Resolve hook_summary from known aliases
            summary_val = ""
            for alias in _summary_aliases:
                if alias in item:
                    summary_val = str(item[alias])
                    break
            # Fallback: any key ending in "_summary" (e.g. "complete_thought_summary")
            if not summary_val:
                for key in item:
                    if key.endswith("_summary") and key not in core_keys:
                        summary_val = str(item[key])
                        break
            if not summary_val:
                logger.warning(
                    "Segment missing hook_summary (and aliases %s): %s",
                    _summary_aliases, {k for k in item.keys()} - core_keys,
                )
                continue
            try:
                boundaries.append(
                    SegmentBoundary(
                        start_time=float(item["start_time"]),
                        end_time=float(item["end_time"]),
                        hook_summary=summary_val,
                        segment_type=str(item["segment_type"]),
                    )
                )
            except (ValueError, TypeError) as exc:
                logger.warning("Invalid segment data: %s — %s", item, exc)
                continue

        return boundaries

    @staticmethod
    def _validate_timestamps(
        boundaries: list[SegmentBoundary],
        total_duration: float,
    ) -> list[SegmentBoundary]:
        """Validate and clamp LLM-returned timestamps to [0, total_duration].

        Logs a warning for each out-of-range timestamp before clamping.

        Args:
            boundaries: Parsed segment boundaries from LLM.
            total_duration: Total transcript duration in seconds.

        Returns:
            Boundaries with timestamps clamped to valid range.
        """
        validated: list[SegmentBoundary] = []
        for seg in boundaries:
            start = seg.start_time
            end = seg.end_time
            if start < 0 or start > total_duration:
                logger.warning(
                    "LLM returned out-of-range start_time %.2f "
                    "(valid: 0–%.2f), clamping",
                    start, total_duration,
                )
                start = max(0.0, min(start, total_duration))
            if end < 0 or end > total_duration:
                logger.warning(
                    "LLM returned out-of-range end_time %.2f "
                    "(valid: 0–%.2f), clamping",
                    end, total_duration,
                )
                end = max(0.0, min(end, total_duration))
            if end <= start:
                logger.warning(
                    "Dropping segment with invalid range after clamping: "
                    "%.2f–%.2f",
                    start, end,
                )
                continue
            validated.append(
                SegmentBoundary(
                    start_time=start,
                    end_time=end,
                    hook_summary=seg.hook_summary,
                    segment_type=seg.segment_type,
                )
            )
        return validated

    @staticmethod
    def _log_whisper_download_status(model_size: str) -> None:
        """Log a message if the Whisper model is not cached locally.

        Informs the user that a potentially large download is about to
        happen, so they don't think the tool is hung.
        """
        try:
            from huggingface_hub import try_to_load_from_cache
            repo_id = f"guillaumekln/faster-whisper-{model_size}"
            cached = try_to_load_from_cache(repo_id, "model.bin")
            if cached is None:
                logger.info(
                    "Downloading Whisper model '%s'... this may take "
                    "several minutes on first run.",
                    model_size,
                )
        except ImportError:
            # huggingface_hub not available — can't check cache, proceed
            pass
        except Exception:
            # Non-critical — proceed with model loading
            pass

    @staticmethod
    def _backoff(attempt: int) -> None:
        """Exponential backoff: 2s, 4s, 8s.

        Sleeps in 1-second increments so KeyboardInterrupt is not
        blocked for the entire delay.
        """
        delay = 2**attempt
        logger.info("Backing off for %ds before retry", delay)
        for _ in range(delay):
            time.sleep(1)

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        """Format seconds as MM:SS.ff for transcript display."""
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins:02d}:{secs:05.2f}"
