"""
Semantic content analysis module using Qwen2.5-VL via Ollama.

Uses a vision-language model to understand video content meaning, scoring
emotional intensity, narrative interest, hook potential, ASMR quality,
visual appeal, and uniqueness. Adapts the OllamaVideoAnalyzer pattern
from caption_generator.py for virality-focused analysis.
"""

import base64
import json
import logging
import os
import tempfile
import time
from typing import Optional

import requests

from viral_clip_extractor.models import SemanticFeatures
from viral_clip_extractor.utils.video_utils import extract_segment

logger = logging.getLogger(__name__)

# Default scores returned when analysis fails — neutral midpoint values
_DEFAULT_SCORE = 5.0
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
    ) -> None:
        self.model = model
        self.ollama_host = ollama_host.rstrip("/")

    def check_availability(self) -> bool:
        """Check whether the Ollama service is running and the model is loaded.

        Returns:
            True if the service responds and the configured model is listed.
        """
        try:
            response = requests.get(
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
        retries), returns ``SemanticFeatures`` with default midpoint scores
        so the pipeline can continue.

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
            logger.warning(
                "Invalid segment duration (%.2f-%.2f), returning defaults",
                start_time,
                end_time,
            )
            return self._default_features()

        # Extract segment to a temp file
        segment_path: Optional[str] = None
        try:
            fd, segment_path = tempfile.mkstemp(suffix=".mp4", prefix="sem_")
            os.close(fd)
            extract_segment(video_path, start_time, end_time, segment_path)
        except Exception as exc:
            logger.error("Failed to extract segment: %s", exc)
            if segment_path and os.path.exists(segment_path):
                os.unlink(segment_path)
            return self._default_features()

        try:
            # Base64 encode the segment
            video_base64 = self._encode_video(segment_path)
            if video_base64 is None:
                return self._default_features()

            # Build the analysis prompt
            prompt = self._create_virality_prompt(duration, title)

            # Query Ollama with retries
            features = self._query_with_retries(prompt, video_base64)
            return features
        finally:
            # Clean up temp file
            if segment_path and os.path.exists(segment_path):
                os.unlink(segment_path)
                logger.debug("Cleaned up temp segment: %s", segment_path)

    def _create_virality_prompt(self, duration: float, title: str) -> str:
        """Build the virality-analysis prompt for the VLM.

        Args:
            duration: Segment duration in seconds.
            title: Video title for context.

        Returns:
            The formatted prompt string.
        """
        title_line = f' from "{title}"' if title else ""
        return (
            f"Analyze this {duration:.1f}s video segment{title_line}.\n"
            "\n"
            "Rate each factor on a 0-10 scale:\n"
            "1. EMOTIONAL_INTENSITY: How emotionally engaging is this moment?\n"
            "2. NARRATIVE_INTEREST: Does this create curiosity or tell a story?\n"
            "3. HOOK_POTENTIAL: Would this grab attention in the first 2 seconds?\n"
            "4. ASMR_QUALITY: ASMR trigger intensity (tingles, relaxation potential)\n"
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

    def _encode_video(self, video_path: str) -> Optional[str]:
        """Base64-encode a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            Base64-encoded string, or ``None`` on failure.
        """
        try:
            with open(video_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as exc:
            logger.error("Failed to encode video %s: %s", video_path, exc)
            return None

    def _query_with_retries(
        self, prompt: str, video_base64: str
    ) -> SemanticFeatures:
        """Send the prompt + video to Ollama with exponential backoff.

        Args:
            prompt: The analysis prompt.
            video_base64: Base64-encoded video data.

        Returns:
            Parsed ``SemanticFeatures``, or defaults after all retries fail.
        """
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                logger.info(
                    "Semantic analysis attempt %d/%d", attempt, _MAX_RETRIES
                )
                response = requests.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "images": [video_base64],
                        "stream": False,
                        "options": {
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

        logger.warning(
            "All %d attempts failed — returning default features", _MAX_RETRIES
        )
        return self._default_features()

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
        """Exponential backoff: 2s, 4s, 8s, ..."""
        delay = 2 ** attempt
        logger.info("Backing off for %ds before retry", delay)
        time.sleep(delay)

    @staticmethod
    def _default_features() -> SemanticFeatures:
        """Return ``SemanticFeatures`` with neutral default scores."""
        return SemanticFeatures(
            emotional_intensity=_DEFAULT_SCORE,
            narrative_interest=_DEFAULT_SCORE,
            hook_potential=_DEFAULT_SCORE,
            asmr_quality=_DEFAULT_SCORE,
            visual_appeal=_DEFAULT_SCORE,
            uniqueness=_DEFAULT_SCORE,
            description="Analysis unavailable — using default scores",
        )
