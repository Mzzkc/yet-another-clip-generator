"""
Semantic content analysis module using Qwen2.5-VL via Ollama.

Uses a vision-language model to understand video content meaning, scoring
emotional intensity, narrative interest, hook potential, ASMR quality,
visual appeal, and uniqueness. Sends representative JPEG frames (not raw
MP4 bytes) to the VLM for accurate visual analysis.
"""

import base64
import json
import logging
import time
from typing import Optional

import requests

from yacg.models import SemanticFeatures
from yacg.utils.video_utils import get_frame_at_time

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_REQUEST_TIMEOUT = 120


class SemanticAnalyzer:
    """Analyze semantic content of video segments using Qwen2.5-VL via Ollama.

    Extracts video segments, encodes them as base64, sends them to the
    vision-language model, and parses virality ratings from the JSON response.

    Args:
        model: Ollama model name (default: ``qwen2.5-vl:7b``).
        ollama_host: Ollama API base URL.
    """

    def __init__(
        self,
        model: str = "qwen2.5-vl:7b",
        ollama_host: str = "http://localhost:11434",
        num_frames: int = 3,
        content_type: str = "general",
        channel_description: str = "",
        target_audience: str = "",
        tone: str = "",
        custom_instructions: str = "",
    ) -> None:
        self.model = model
        self.ollama_host = ollama_host.rstrip("/")
        self.content_type = content_type
        self.channel_description = channel_description
        self.target_audience = target_audience
        self.tone = tone
        self.custom_instructions = custom_instructions
        # Number of frames to extract per segment for VLM analysis.
        # NOTE: Some VL models produce garbled output with multiple images.
        # If visual scores degrade, try setting num_frames=1 to fall back
        # to single-frame analysis. This was changed from 1 to 3 to improve
        # temporal coverage but has not been verified with all VLM backends.
        self.num_frames = num_frames
        if num_frames > 1:
            logger.debug(
                "Semantic analyzer using %d frames per segment — "
                "set num_frames=1 if VLM produces garbled visual scores",
                num_frames,
            )
        # Reuse HTTP connections across Ollama API calls
        self._session = requests.Session()

    def check_availability(self) -> bool:
        """Check whether the Ollama service is running and the model is loaded.

        Returns:
            True if the service responds and the configured model is listed.
        """
        try:
            response = self._session.get(
                f"{self.ollama_host}/api/tags", timeout=5
            )
            if response.status_code != 200:
                logger.warning(
                    "Ollama returned status %d when listing models",
                    response.status_code,
                )
                return False

            models = response.json().get("models", [])
            available = [m["name"] for m in models]
            if self.model in available:
                logger.info("Model %s is available", self.model)
                return True

            logger.warning(
                "Model %s not found. Available: %s", self.model, available
            )
            return False
        except requests.ConnectionError:
            logger.error("Cannot connect to Ollama at %s", self.ollama_host)
            return False
        except requests.Timeout:
            logger.error("Timeout checking Ollama at %s", self.ollama_host)
            return False
        except Exception as exc:
            logger.error("Unexpected error checking Ollama: %s", exc)
            return False

    def analyze_segment(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        title: str = "",
    ) -> SemanticFeatures:
        """Analyze semantic virality signals for a video segment.

        Extracts a short clip, sends it to Qwen2.5-VL via Ollama, and
        parses the JSON response into ``SemanticFeatures``.

        On total failure (network errors, malformed responses after all
        retries), raises ``RuntimeError``.

        Args:
            video_path: Path to the source video file.
            start_time: Segment start in seconds.
            end_time: Segment end in seconds.
            title: Video title for prompt context.

        Returns:
            A ``SemanticFeatures`` instance with 0-10 ratings.
        """
        duration = end_time - start_time
        if duration <= 0:
            raise RuntimeError(
                f"Invalid segment duration ({start_time:.2f}-{end_time:.2f}): "
                f"duration must be positive"
            )

        # Extract frames directly from the source video using absolute
        # timestamps. This avoids creating an intermediate temp MP4 per
        # segment, saving 150-300MB of temp I/O for a 10-segment video.
        #
        # COUPLING NOTE: Frame timestamps are computed as
        #   start_offset + duration * i / (num_frames + 1)
        # which yields absolute positions in the source video. If this is
        # ever refactored to extract from a temp segment file, set
        # start_offset=0.0 — otherwise frames will be from wrong positions.
        frame_images = self._extract_frames_base64(
            video_path, duration, num_frames=self.num_frames,
            start_offset=start_time,
        )
        if not frame_images:
            raise RuntimeError(
                f"Failed to extract frames from {video_path} "
                f"[{start_time:.2f}s-{end_time:.2f}s]"
            )

        # Build the analysis prompt
        prompt = self._create_virality_prompt(duration, title)

        # Query Ollama with retries
        return self._query_with_retries(prompt, frame_images)

    def _create_virality_prompt(self, duration: float, title: str) -> str:
        """Build the virality-analysis prompt for the VLM.

        Args:
            duration: Segment duration in seconds.
            title: Video title for context.

        Returns:
            The formatted prompt string.
        """
        title_line = f' from "{title}"' if title else ""
        ct = self.content_type.lower()

        # Content-type-aware rating guidance
        emotional_desc = self._get_emotional_guidance(ct)
        hook_desc = self._get_hook_guidance(ct)
        asmr_desc = self._get_sensory_guidance(ct)

        # Build optional context sections
        context_parts: list[str] = []
        if self.channel_description:
            context_parts.append(
                f"Channel context: {self.channel_description}"
            )
        if self.target_audience:
            context_parts.append(
                f"Target audience: {self.target_audience}"
            )
        if self.tone:
            context_parts.append(
                f"Content tone: {self.tone}"
            )
        context_block = ""
        if context_parts:
            context_block = "\n".join(context_parts) + "\n\n"

        custom_block = ""
        if self.custom_instructions:
            custom_block = (
                f"Additional rating context: {self.custom_instructions}\n\n"
            )

        return (
            f"Analyze this {duration:.1f}s video segment{title_line}.\n"
            "\n"
            f"{context_block}"
            f"{custom_block}"
            "Rate each factor on a 0-10 scale. Use the full range: "
            "5 is average, 1-3 is below average, 7-9 is notably good, "
            "10 is exceptional. Avoid clustering all scores in 5-7.\n"
            "\n"
            f"1. EMOTIONAL_INTENSITY: {emotional_desc}\n"
            "2. NARRATIVE_INTEREST: Does this create curiosity or tell a story?\n"
            f"3. HOOK_POTENTIAL: {hook_desc}\n"
            f"4. ASMR_QUALITY: {asmr_desc}\n"
            "5. VISUAL_APPEAL: Aesthetic quality and composition\n"
            "6. UNIQUENESS: How memorable or unusual is this moment?\n"
            "\n"
            "Output ONLY valid JSON in this exact format with NO additional text:\n"
            "{\n"
            '  "emotional_intensity": 0,\n'
            '  "narrative_interest": 0,\n'
            '  "hook_potential": 0,\n'
            '  "asmr_quality": 0,\n'
            '  "visual_appeal": 0,\n'
            '  "uniqueness": 0,\n'
            '  "brief_description": "..."\n'
            "}"
        )

    @staticmethod
    def _get_emotional_guidance(content_type: str) -> str:
        """Return content-type-specific emotional intensity guidance."""
        guidance = {
            "gaming": "How intense is the gameplay moment? (excitement, tension, triumph, frustration)",
            "cooking": "How appetizing and emotionally satisfying is this moment? (craving, delight, anticipation)",
            "educational": "How impactful is this insight? (surprise, curiosity, understanding)",
            "asmr": "How deeply relaxing or tingle-inducing is this moment?",
            "fitness": "How motivating and energizing is this moment? (determination, achievement, inspiration)",
            "music": "How emotionally moving is this musical moment? (chills, joy, intensity)",
            "comedy": "How funny or entertaining is this moment? (laughter, absurdity, wit)",
        }
        return guidance.get(content_type, "How emotionally engaging is this moment?")

    @staticmethod
    def _get_hook_guidance(content_type: str) -> str:
        """Return content-type-specific hook potential guidance."""
        guidance = {
            "gaming": "Would a gamer stop scrolling for this? (epic play, fail, clutch moment)",
            "cooking": "Would a food lover stop scrolling? (sizzle, technique reveal, plating)",
            "educational": "Would a curious person stop scrolling? (bold claim, surprising fact, question)",
            "asmr": "Would this trigger-check grab an ASMR listener immediately?",
            "fitness": "Would someone stop scrolling for this? (impressive feat, transformation, challenge)",
            "music": "Would a music fan stop scrolling? (impressive skill, catchy hook, raw talent)",
            "comedy": "Would this make someone pause to watch? (unexpected setup, absurd premise)",
        }
        return guidance.get(content_type, "Would this grab attention in the first 2 seconds?")

    @staticmethod
    def _get_sensory_guidance(content_type: str) -> str:
        """Return content-type-specific guidance for the asmr_quality field."""
        guidance = {
            "asmr": "ASMR trigger intensity (tingles, relaxation potential)",
            "gaming": "Sensory excitement and immersion (sound design, visual intensity, tactile feel)",
            "cooking": "Sensory appeal (sights, sounds, textures — sizzling, crunching, plating)",
            "educational": "Sensory clarity (visual aids, demonstrations, clear presentation)",
            "fitness": "Physical immersion (movement quality, energy, body awareness)",
            "music": "Auditory richness and sonic immersion (tone, dynamics, production quality)",
            "comedy": "Sensory timing (delivery, physical comedy, visual gags)",
        }
        return guidance.get(content_type, "Sensory engagement and immersive quality")

    def _extract_frames_base64(
        self, video_path: str, duration: float, num_frames: int = 3,
        start_offset: float = 0.0,
    ) -> list[str]:
        """Extract representative JPEG frames from a video.

        Captures frames at evenly-spaced timestamps (25%, 50%, 75% for 3 frames)
        and returns them as base64-encoded JPEG strings suitable for the Ollama
        ``images`` field.

        Args:
            video_path: Path to the video file (source or segment).
            duration: Duration of the segment in seconds.
            num_frames: Number of frames to extract (default 3).
            start_offset: Absolute offset in seconds to add to computed
                timestamps. Use when extracting from the source video rather
                than an already-trimmed segment file.

        Returns:
            List of base64-encoded JPEG strings. Empty on failure.
        """
        try:
            import cv2
        except ImportError:
            logger.error("cv2 is required for frame extraction")
            return []

        frames_b64: list[str] = []
        for i in range(1, num_frames + 1):
            timestamp = start_offset + duration * i / (num_frames + 1)
            frame = get_frame_at_time(video_path, timestamp)
            if frame is None:
                continue
            success, buffer = cv2.imencode(".jpg", frame)
            if success:
                frames_b64.append(
                    base64.b64encode(buffer.tobytes()).decode("utf-8")
                )

        if not frames_b64:
            # Fallback: try a single frame at the midpoint
            frame = get_frame_at_time(video_path, start_offset + duration / 2.0)
            if frame is not None:
                success, buffer = cv2.imencode(".jpg", frame)
                if success:
                    frames_b64.append(
                        base64.b64encode(buffer.tobytes()).decode("utf-8")
                    )

        logger.debug(
            "Extracted %d frames from %s (%.1fs)", len(frames_b64), video_path, duration,
        )
        return frames_b64

    def _query_with_retries(
        self, prompt: str, images: list[str],
    ) -> SemanticFeatures:
        """Send the prompt + frames to Ollama with exponential backoff.

        Args:
            prompt: The analysis prompt.
            images: List of base64-encoded JPEG images.

        Returns:
            Parsed ``SemanticFeatures``. Raises ``RuntimeError`` if all retries fail.
        """
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                logger.info(
                    "Semantic analysis attempt %d/%d", attempt, _MAX_RETRIES
                )
                response = self._session.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "images": images,
                        "stream": False,
                        "options": {
                            # Low temperature (0.3) for reproducible numeric
                            # scores. Semantic ratings must be consistent
                            # across retries for the same input.
                            "temperature": 0.3,
                            "top_p": 0.9,
                        },
                    },
                    timeout=_REQUEST_TIMEOUT,
                )

                if response.status_code != 200:
                    logger.warning(
                        "Ollama returned status %d on attempt %d",
                        response.status_code,
                        attempt,
                    )
                    if attempt < _MAX_RETRIES:
                        self._backoff(attempt)
                    continue

                response_text = response.json().get("response", "")
                features = self._parse_llm_response(response_text)
                if features is not None:
                    logger.info("Semantic analysis succeeded on attempt %d", attempt)
                    return features

                logger.warning(
                    "Failed to parse LLM response on attempt %d", attempt
                )

            except requests.ConnectionError:
                logger.error(
                    "Connection error on attempt %d — is Ollama running?",
                    attempt,
                )
            except requests.Timeout:
                logger.error("Request timed out on attempt %d", attempt)
            except Exception as exc:
                logger.error(
                    "Unexpected error on attempt %d: %s", attempt, exc
                )

            if attempt < _MAX_RETRIES:
                self._backoff(attempt)

        raise RuntimeError(
            f"Semantic analysis failed after {_MAX_RETRIES} attempts. "
            f"Please check that Ollama is running (ollama serve) and "
            f"the model '{self.model}' is available (ollama pull {self.model})."
        )

    def _parse_llm_response(
        self, response_text: str
    ) -> Optional[SemanticFeatures]:
        """Extract ``SemanticFeatures`` from the LLM's JSON response.

        Follows the ``_parse_llm_response`` pattern from
        ``caption_generator.py``: locate the first ``{`` and last ``}``
        in the response, parse the JSON between them, and validate fields.

        Args:
            response_text: Raw text from the LLM.

        Returns:
            A ``SemanticFeatures`` instance, or ``None`` if parsing fails.
        """
        try:
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1

            if start_idx == -1 or end_idx == 0:
                logger.error("No JSON found in LLM response")
                return None

            json_str = response_text[start_idx:end_idx]
            data = json.loads(json_str)

            required = [
                "emotional_intensity",
                "narrative_interest",
                "hook_potential",
                "asmr_quality",
                "visual_appeal",
                "uniqueness",
            ]
            if not all(key in data for key in required):
                missing = [k for k in required if k not in data]
                logger.error("Missing required fields: %s", missing)
                return None

            return SemanticFeatures(
                emotional_intensity=self._clamp(float(data["emotional_intensity"])),
                narrative_interest=self._clamp(float(data["narrative_interest"])),
                hook_potential=self._clamp(float(data["hook_potential"])),
                asmr_quality=self._clamp(float(data["asmr_quality"])),
                visual_appeal=self._clamp(float(data["visual_appeal"])),
                uniqueness=self._clamp(float(data["uniqueness"])),
                description=str(data.get("brief_description", "")),
            )
        except json.JSONDecodeError as exc:
            logger.error("JSON parse error: %s", exc)
            logger.debug("Response text: %s", response_text)
            return None
        except (ValueError, TypeError) as exc:
            logger.error("Value conversion error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp(value: float, lo: float = 0.0, hi: float = 10.0) -> float:
        """Clamp a value to the [lo, hi] range."""
        return max(lo, min(hi, value))

    @staticmethod
    def _backoff(attempt: int) -> None:
        """Exponential backoff: 2s, 4s, 8s, ...

        Sleeps in 1-second increments so KeyboardInterrupt is not
        blocked for the entire delay.
        """
        delay = 2 ** attempt
        logger.info("Backing off for %ds before retry", delay)
        for _ in range(delay):
            time.sleep(1)

