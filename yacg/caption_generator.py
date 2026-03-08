"""
Caption generation for extracted clips using Ollama vision models.

Sends a representative frame (not raw video bytes) to Ollama's vision API
and parses the structured caption response. Integrated into the pipeline
as a mandatory post-extraction step — every clip requires a generated caption.
"""

import base64
import json
import logging
import time
from typing import Optional

import requests

from yacg.models import CaptionData

logger = logging.getLogger(__name__)


class OllamaVideoAnalyzer:
    """Interface to Ollama for video analysis using Qwen2.5-VL.

    Extracts a representative frame from the clip and sends it as a
    JPEG image to the Ollama vision API rather than the raw video binary.

    Args:
        model: Ollama model name.
        ollama_host: Ollama API base URL.
    """

    def __init__(
        self,
        model: str = "qwen2.5-vl:7b",
        ollama_host: str = "http://localhost:11434",
        content_type: str = "general",
        channel_description: str = "",
        target_audience: str = "",
        tone: str = "",
        platform: str = "",
        caption_length: str = "",
        hashtag_count: int = 5,
        custom_instructions: str = "",
    ) -> None:
        self.model = model
        self.ollama_host = ollama_host
        self.content_type = content_type
        self.channel_description = channel_description
        self.target_audience = target_audience
        self.tone = tone
        self.platform = platform
        self.caption_length = caption_length
        self.hashtag_count = hashtag_count
        self.custom_instructions = custom_instructions
        # Reuse HTTP connections across Ollama API calls
        self._session = requests.Session()

    def analyze_video(
        self, video_path: str, title: str, max_retries: int = 3,
        transcript_text: str = "",
    ) -> CaptionData:
        """Analyze video and generate a short-form caption.

        Extracts 3 frames (at 25%, 50%, 75% of duration) from the clip,
        encodes them as JPEG, and sends them to Ollama's ``/api/generate``
        endpoint for richer temporal context.

        Args:
            video_path: Path to video file.
            title: User-provided title for the video.
            max_retries: Number of retry attempts for API calls.
            transcript_text: Transcript of what was said in the clip.
                Dramatically improves caption quality vs frames-only.

        Returns:
            CaptionData object. Raises RuntimeError if analysis fails.
        """
        prompt = self._create_caption_prompt(title, transcript_text)

        # Extract representative frames and encode as JPEG
        images_base64 = self._extract_frames_base64(video_path)
        if not images_base64:
            raise RuntimeError(
                f"Could not extract any frames from {video_path} — "
                f"caption generation requires at least one valid video frame"
            )

        for attempt in range(max_retries):
            try:
                logger.info("Analyzing video (attempt %d/%d)...", attempt + 1, max_retries)

                response = self._session.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "images": images_base64,
                        "stream": False,
                        "options": {
                            # Higher temperature (0.7) for creative, varied
                            # captions. Instagram hooks benefit from diversity
                            # and flair — not deterministic repetition.
                            "temperature": 0.7,
                            "top_p": 0.9,
                        },
                    },
                    timeout=120,
                )

                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get("response", "")
                    caption_data = self._parse_llm_response(response_text)
                    if caption_data:
                        logger.info("Successfully generated caption")
                        return caption_data
                    else:
                        logger.warning(
                            "Failed to parse LLM response on attempt %d", attempt + 1
                        )
                else:
                    logger.error("Ollama API error: %d", response.status_code)

            except Exception as exc:
                logger.error("Error on attempt %d: %s", attempt + 1, exc)

            # Backoff on ALL failure types (exception, parse failure, non-200)
            if attempt < max_retries - 1:
                delay = 2 ** (attempt + 1)
                logger.info("Backing off for %ds before retry", delay)
                for _ in range(delay):
                    time.sleep(1)

        raise RuntimeError(
            f"Caption generation failed for {video_path} after {max_retries} attempts"
        )

    def _extract_frames_base64(
        self, video_path: str, num_frames: int = 3,
    ) -> list[str]:
        """Extract frames at evenly-spaced timestamps as base64 JPEGs.

        Uses the same ``duration * i / (num_frames + 1)`` formula as
        ``SemanticAnalyzer._extract_frames_base64`` so both modules
        produce identical timestamps for the same ``num_frames`` value.

        Falls back to fewer frames if some timestamps fail extraction.
        Returns an empty list if all extractions fail.
        """
        try:
            from yacg.utils.video_utils import (
                extract_metadata,
                get_frame_at_time,
            )
            import cv2

            meta = extract_metadata(video_path)
            duration = meta.get("duration", 0)
            if duration <= 0:
                raise RuntimeError(
                    f"Video has zero or negative duration ({duration}s) — "
                    f"cannot extract frames for caption generation"
                )
            else:
                timestamps = [
                    duration * i / (num_frames + 1)
                    for i in range(1, num_frames + 1)
                ]

            frames_b64: list[str] = []
            for ts in timestamps:
                frame = get_frame_at_time(video_path, ts)
                if frame is None:
                    continue
                success, buffer = cv2.imencode(".jpg", frame)
                if success:
                    frames_b64.append(
                        base64.b64encode(buffer.tobytes()).decode("utf-8")
                    )

            return frames_b64

        except RuntimeError:
            raise  # Preserve diagnostic (e.g. zero-duration) for caller
        except Exception as exc:
            logger.error("Frame extraction failed: %s", exc)
            return []

    def _create_caption_prompt(
        self, title: str, transcript_text: str = "",
    ) -> str:
        """Create an optimized prompt for caption generation.

        Adapts persona, tone, platform conventions, and category list
        based on content profile parameters. Injects transcript text so
        the LLM knows what was said, not just what frames look like.
        """
        categories = (
            "ASMR, Satisfying, Tutorial, Story, Transition, Dance, "
            "Comedy, Educational, Cooking, Gaming, Fitness, Music, "
            "Vlog, Beauty, Travel"
        )

        persona = self._get_persona()
        tone_instruction = self._get_tone_instruction()
        platform_instruction = self._get_platform_instruction()

        # Determine hashtag count (clamped to reasonable range)
        htag_count = max(1, min(10, self.hashtag_count))

        # Determine description length guidance
        if self.caption_length:
            length_guidance = f"({self.caption_length})"
        else:
            length_guidance = "(2-3 sentences, 100-150 chars)"

        # Build context sections
        context_parts: list[str] = []
        if self.channel_description:
            context_parts.append(
                f"Channel: {self.channel_description}"
            )
        if self.target_audience:
            context_parts.append(
                f"Target audience: {self.target_audience}"
            )
        context_block = ""
        if context_parts:
            context_block = "\n".join(context_parts) + "\n\n"

        # Transcript injection — the most important improvement
        transcript_block = ""
        if transcript_text:
            transcript_block = (
                f"TRANSCRIPT OF WHAT WAS SAID IN THIS CLIP:\n"
                f"{transcript_text}\n\n"
                f"Use the transcript to write captions that reference what "
                f"was actually said — not just what the frames show.\n\n"
            )

        # Custom instructions
        custom_block = ""
        if self.custom_instructions:
            custom_block = (
                f"CREATOR'S INSTRUCTIONS:\n"
                f"{self.custom_instructions}\n\n"
            )

        return (
            f"You are {persona}.\n"
            f"\n"
            f"{context_block}"
            f'Analyze these frames from a video titled "{title}".\n'
            f"\n"
            f"{transcript_block}"
            f"{custom_block}"
            f"{tone_instruction}"
            f"{platform_instruction}"
            f"Generate short-form optimized content with these requirements:\n"
            f"\n"
            f"1. HOOK (8-12 words): An immediate attention-grabbing opening line.\n"
            f"2. DESCRIPTION {length_guidance}: Keywords, viewer benefit, call-to-action.\n"
            f"3. HASHTAGS ({htag_count}): Mix of niche, discovery, and trending tags.\n"
            f"4. CATEGORY: One of [{categories}]\n"
            f"5. VIRALITY PREDICTION (0-100): Estimated viral potential.\n"
            f"\n"
            f"Output ONLY valid JSON:\n"
            f'{{"hook": "...", "description": "...", "hashtags": ["tag1", "tag2", "tag3"], "category": "...", "virality_score": 75}}'
        )

    def _get_persona(self) -> str:
        """Return a content-type-appropriate persona for the LLM."""
        ct = self.content_type.lower()
        personas = {
            "asmr": "an expert Instagram content strategist specializing in ASMR and short-form viral content",
            "gaming": "an expert gaming content strategist who creates viral clips for Twitch, YouTube, and TikTok",
            "cooking": "an expert food content strategist specializing in recipe videos and cooking content for social media",
            "educational": "an expert educational content strategist who makes complex topics go viral on social media",
            "fitness": "an expert fitness content strategist specializing in workout and transformation content",
            "music": "an expert music content strategist specializing in viral music clips and artist promotion",
            "comedy": "an expert comedy content strategist who knows what makes people laugh and share",
        }
        return personas.get(
            ct,
            "an expert short-form content strategist for social media platforms (TikTok, Instagram Reels, YouTube Shorts)",
        )

    def _get_tone_instruction(self) -> str:
        """Return tone guidance for the caption voice."""
        if not self.tone:
            return ""
        return (
            f"Write in a {self.tone} tone. Match this energy in "
            f"the hook, description, and hashtag choices.\n\n"
        )

    def _get_platform_instruction(self) -> str:
        """Return platform-specific caption guidance."""
        if not self.platform:
            return ""
        p = self.platform.lower()
        instructions = {
            "tiktok": (
                "Optimize for TikTok: punchy hooks, trending sounds "
                "references, TikTok-native hashtags. Keep captions short.\n\n"
            ),
            "reels": (
                "Optimize for Instagram Reels: polished hooks, "
                "engagement-driving questions, mix of niche and broad hashtags.\n\n"
            ),
            "shorts": (
                "Optimize for YouTube Shorts: curiosity-gap hooks, "
                "keyword-rich descriptions, discovery-focused hashtags.\n\n"
            ),
        }
        return instructions.get(p, "")

    def _parse_llm_response(self, response_text: str) -> Optional[CaptionData]:
        """Parse LLM response and extract caption data."""
        try:
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1

            if start_idx == -1 or end_idx == 0:
                logger.error("No JSON found in response")
                return None

            json_str = response_text[start_idx:end_idx]
            data = json.loads(json_str)

            required_fields = ["hook", "description", "hashtags", "category", "virality_score"]
            if not all(field in data for field in required_fields):
                logger.error("Missing required fields in response")
                return None

            hashtags = data["hashtags"]
            if isinstance(hashtags, str):
                hashtags = [tag.strip() for tag in hashtags.split(",")]
            hashtags = [tag if tag.startswith("#") else f"#{tag}" for tag in hashtags]
            hashtags = hashtags[:self.hashtag_count]

            full_caption = self._format_full_caption(
                data["hook"], data["description"], hashtags
            )

            return CaptionData(
                hook=data["hook"],
                description=data["description"],
                hashtags=hashtags,
                category=data["category"],
                virality_score=int(data["virality_score"]),
                full_caption=full_caption,
            )

        except json.JSONDecodeError as exc:
            logger.error("JSON parsing error: %s", exc)
            return None
        except Exception as exc:
            logger.error("Error parsing response: %s", exc)
            return None

    def _format_full_caption(
        self, hook: str, description: str, hashtags: list[str]
    ) -> str:
        """Format complete Instagram caption."""
        return "\n".join([hook, "", description, "", " ".join(hashtags)])
