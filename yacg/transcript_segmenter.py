"""
LLM-driven transcript segmentation engine.

Replaces PySceneDetect as the primary source of clip boundaries.
Runs faster-whisper on the full source video to get word-level timestamps,
sends the formatted transcript to Ollama for content-based segment
identification, then refines boundaries by snapping to speech pauses.
"""

import json
import logging
import re
import time

import requests

from yacg.models import (
    SceneSegment,
    SegmentBoundary,
    WordTimestamp,
)

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_REQUEST_TIMEOUT = 180
_SPEECH_PAUSE_THRESHOLD = 0.3  # seconds — gap defining a speech pause
_MIN_SEGMENT_DURATION = 15.0  # seconds
_MAX_SEGMENT_DURATION = 45.0  # seconds


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
    ) -> list[SceneSegment]:
        """Snap segment boundaries to nearest speech pauses.

        Adjusts LLM-suggested boundaries to fall on speech pauses (gaps >300ms
        between words) so clips don't cut mid-word. Validates no overlaps,
        enforces min/max duration, and returns SceneSegment objects for
        pipeline compatibility.

        Args:
            segments: LLM-identified segment boundaries.
            words: Full transcript word timestamps.

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

            # Ensure minimum duration
            duration = end - start
            if duration < self.min_segment_duration:
                # Try extending end to reach minimum
                needed = self.min_segment_duration - duration
                end = min(video_end, end + needed)
                duration = end - start

            # Enforce maximum duration
            if duration > self.max_segment_duration:
                # Find a pause BEFORE the max duration mark to stay within bounds
                target_end = start + self.max_segment_duration
                end = self._snap_to_nearest_pause(
                    target_end, pauses, prefer="before"
                )
                # If snap pulled us below min, hard cap at max as last resort
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
        scenes = self.refine_boundaries(boundaries, words)
        return scenes, words

    def _format_transcript(
        self, words: list[WordTimestamp], block_seconds: float = 10.0
    ) -> str:
        """Format words into timestamped text blocks for the LLM prompt.

        Groups words into blocks of approximately block_seconds duration,
        producing lines like: [00:18.08 - 00:28.00] We're no strangers to love ...
        """
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
        """Find all speech pauses (gaps > 300ms) in the word list.

        Returns:
            List of (pause_start, pause_end, is_sentence_end) tuples where
            pause_start is the end time of the word before the gap,
            pause_end is the start time of the word after the gap, and
            is_sentence_end is True if the word before the pause ends with
            sentence-ending punctuation (. ? !).
        """
        pauses: list[tuple[float, float, bool]] = []
        for i in range(len(words) - 1):
            gap = words[i + 1].start - words[i].end
            if gap > self.pause_threshold:
                is_sentence_end = (
                    len(words[i].word) > 0
                    and words[i].word[-1] in ".?!"
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
        """Build the LLM segmentation prompt with the transcript."""
        title_line = f' titled "{title}"' if title else ""

        # Content-type-aware guidance for what makes a "good clip"
        content_guidance = self._get_content_type_guidance()

        # Build optional context sections
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

        # Custom instructions override section
        custom_block = ""
        if self.custom_instructions:
            custom_block = (
                f"ADDITIONAL INSTRUCTIONS FROM CREATOR:\n"
                f"{self.custom_instructions}\n\n"
            )

        return (
            f"You are analyzing the transcript of a video{title_line} to "
            f"identify viral-worthy clip segments for TikTok, Instagram "
            f"Reels, and YouTube Shorts.\n"
            f"\n"
            f"{context_block}"
            f"{custom_block}"
            f"TRANSCRIPT (with timestamps):\n"
            f"{transcript_text}\n"
            f"\n"
            f"Identify segments that would make compelling short-form clips. "
            f"Look for:\n"
            f"{content_guidance}"
            f"\n"
            f"Rules:\n"
            f"- Each segment MUST be 15-45 seconds long\n"
            f"- Each segment MUST contain a complete thought (no mid-sentence "
            f"cuts)\n"
            f"- Segments MUST NOT overlap\n"
            f"- Use the timestamps from the transcript to set boundaries\n"
            f"- You MUST identify at least {target_count} segments. "
            f"Scan the entire transcript thoroughly — do not stop early. "
            f"Every 15-45 second stretch with a coherent thought is a "
            f"valid segment\n"
            f"\n"
            f"Output ONLY a valid JSON array with NO additional text.\n"
            f"segment_type must be one of: hook, narrative_arc, "
            f"complete_thought, emotional_peak\n"
            f"\n"
            f"Example:\n"
            f'[{{"start_time": 0.0, "end_time": 30.0, '
            f'"hook_summary": "brief description", '
            f'"segment_type": "hook"}}]\n'
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
            logger.warning("Failed to parse JSON from LLM response: %s", exc)
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
