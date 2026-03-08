"""
Audio analysis module using librosa for ASMR-specific features.

Extracts audio features relevant to virality scoring: peak loudness,
high-frequency content (ASMR triggers), dynamic range, zero-crossing
rate, and overall energy.  Also detects ASMR trigger words via
faster-whisper when available.
"""

import logging
from typing import Optional

import numpy as np

from viral_clip_extractor.models import AudioFeatures, WordTimestamp

logger = logging.getLogger(__name__)

# Default ASMR trigger keywords for speech matching
_DEFAULT_ASMR_KEYWORDS: list[str] = [
    "tingles", "relax", "sleep", "cozy", "gentle",
    "dragon", "scales", "whisper", "magic",
]


class AudioAnalyzer:
    """Analyze audio segments for viral potential signals using librosa.

    Computes spectral and temporal features tuned for ASMR content:
    RMS energy peaks, high-frequency presence, dynamic range,
    zero-crossing rate, and ASMR-specific patterns (tapping, crinkles,
    mouth sounds).  Optionally transcribes speech to detect trigger words.

    Args:
        asmr_keywords: Custom list of trigger words to match against
            transcribed speech.  Falls back to a built-in default list.
    """

    def __init__(self, asmr_keywords: Optional[list[str]] = None) -> None:
        self.asmr_keywords: list[str] = asmr_keywords or list(_DEFAULT_ASMR_KEYWORDS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_segment(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        words: Optional[list[WordTimestamp]] = None,
    ) -> AudioFeatures:
        """Analyze audio features for a time range within a video file.

        Extracts the audio track from *video_path* (or loads it directly
        if it is already a WAV/audio file), then computes spectral and
        energy features over the ``[start_time, end_time)`` window.

        If *words* is provided (a list of WordTimestamp objects from the
        full-video Whisper transcription), trigger word detection reuses
        those words instead of running a redundant Whisper transcription.

        Args:
            video_path: Path to a video or audio file.
            start_time: Segment start in seconds.
            end_time: Segment end in seconds.

        Returns:
            An :class:`AudioFeatures` instance with computed scores.
        """
        try:
            import librosa
        except ImportError:
            raise RuntimeError(
                "librosa is required for audio analysis. "
                "Install it with: pip install librosa"
            )

        duration = end_time - start_time
        if duration <= 0:
            logger.warning("Invalid segment duration (%.2f–%.2f) — returning zeros", start_time, end_time)
            return AudioFeatures(
                audio_peak_score=0.0,
                high_freq_score=0.0,
                dynamic_range=0.0,
                zcr_score=0.0,
                trigger_words=[],
                overall_energy=0.0,
            )

        # Extract audio to WAV first so librosa uses PySoundFile directly,
        # avoiding the deprecated audioread fallback and its warnings.
        try:
            from viral_clip_extractor.utils.video_utils import extract_audio, temp_audio_file

            with temp_audio_file(suffix=".wav") as tmp_wav:
                extract_audio(video_path, tmp_wav, start=start_time, end=end_time)
                y, sr = librosa.load(tmp_wav, sr=22050, mono=True)
        except Exception as exc:
            logger.debug("WAV extraction failed (%s), falling back to direct load", exc)
            # Fallback: let librosa handle it (may emit warnings)
            try:
                y, sr = librosa.load(
                    video_path, sr=22050, mono=True,
                    offset=start_time, duration=duration,
                )
            except Exception as exc2:
                raise RuntimeError(
                    f"Failed to load audio from {video_path} "
                    f"(segment {start_time:.1f}–{end_time:.1f}s): {exc2}"
                ) from exc2

        # Guard against silent / empty audio
        if y is None or len(y) == 0:
            logger.warning("Empty audio data for %s (%.2f–%.2f)", video_path, start_time, end_time)
            return AudioFeatures(
                audio_peak_score=0.0,
                high_freq_score=0.0,
                dynamic_range=0.0,
                zcr_score=0.0,
                trigger_words=[],
                overall_energy=0.0,
            )

        # --- Compute features ------------------------------------------------

        # 1. RMS energy
        rms = librosa.feature.rms(y=y)[0]
        audio_peak_score = float(np.percentile(rms, 90)) if len(rms) > 0 else 0.0
        overall_energy = float(np.mean(rms)) if len(rms) > 0 else 0.0

        # 2. High-frequency score — fraction of spectral centroid frames > 4 kHz
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        high_freq_score = float(np.mean(spectral_centroid > 4000)) if len(spectral_centroid) > 0 else 0.0

        # 3. Dynamic range — std of RMS (interesting vs boring)
        dynamic_range = float(np.std(rms)) if len(rms) > 0 else 0.0

        # 4. Zero-crossing rate — high for whispers and crisp sounds
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_score = float(np.mean(zcr)) if len(zcr) > 0 else 0.0

        # --- ASMR-specific detections -----------------------------------------

        # Tapping detection: sharp onset transients
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            onsets = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env, units="time")
            if len(onsets) > 0:
                # Dense onsets with high strength ⇒ tapping
                onset_density = len(onsets) / max(duration, 0.01)
                tapping_boost = min(onset_density * 0.02, 0.1)
                audio_peak_score += tapping_boost
        except Exception as exc:
            logger.warning("Onset detection failed: %s", exc)

        # Crinkle detection: high-frequency spectral flux
        try:
            spectral_flux = np.diff(spectral_centroid)
            hf_flux = float(np.mean(np.abs(spectral_flux))) if len(spectral_flux) > 0 else 0.0
            if hf_flux > 500:
                high_freq_score = min(high_freq_score + 0.05, 1.0)
        except Exception as exc:
            logger.warning("Spectral flux computation failed: %s", exc)

        # Mouth-sound detection: mid-frequency energy (1–4 kHz band)
        try:
            S = np.abs(librosa.stft(y))
            freqs = librosa.fft_frequencies(sr=sr)
            mid_mask = (freqs >= 1000) & (freqs <= 4000)
            if np.any(mid_mask):
                mid_energy = float(np.mean(S[mid_mask, :]))
                total_energy_spec = float(np.mean(S))
                if total_energy_spec > 0:
                    mid_ratio = mid_energy / total_energy_spec
                    if mid_ratio > 0.5:
                        zcr_score = min(zcr_score + 0.02, 1.0)
        except Exception as exc:
            logger.warning("Mid-frequency analysis failed: %s", exc)

        # --- Trigger word detection -------------------------------------------
        trigger_words = self._detect_trigger_words(
            video_path, start_time, end_time, existing_words=words,
        )

        return AudioFeatures(
            audio_peak_score=float(audio_peak_score),
            high_freq_score=float(high_freq_score),
            dynamic_range=float(dynamic_range),
            zcr_score=float(zcr_score),
            trigger_words=trigger_words,
            overall_energy=float(overall_energy),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_trigger_words(
        self,
        audio_path: str,
        start: float,
        end: float,
        existing_words: Optional[list[WordTimestamp]] = None,
    ) -> list[str]:
        """Detect ASMR trigger words in the audio segment.

        If *existing_words* is provided (WordTimestamp objects from the
        full-video Whisper run), filters them to the segment range and
        matches against the keyword list — avoiding a redundant Whisper
        transcription.  Falls back to a local ``tiny`` Whisper run when
        no pre-existing words are available.

        Args:
            audio_path: Path to the audio/video file.
            start: Segment start in seconds.
            end: Segment end in seconds.
            existing_words: Pre-computed WordTimestamp objects (optional).

        Returns:
            A (possibly empty) list of detected trigger words.
        """
        # Fast path: reuse existing transcription from the pipeline
        if existing_words:
            _BOUNDARY_TOLERANCE = 0.05
            segment_text = " ".join(
                w.word for w in existing_words
                if w.start >= start - _BOUNDARY_TOLERANCE
                and w.end <= end + _BOUNDARY_TOLERANCE
            ).lower()
            found: list[str] = []
            for keyword in self.asmr_keywords:
                if keyword.lower() in segment_text:
                    found.append(keyword)
            if found:
                logger.info("Detected trigger words in %.1f–%.1fs: %s", start, end, found)
            return found

        # Fallback: run Whisper tiny on this segment
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            logger.debug("faster-whisper not installed — skipping trigger word detection")
            return []

        from viral_clip_extractor.utils.video_utils import (
            extract_audio as _extract_audio,
            temp_audio_file,
        )

        try:
            with temp_audio_file(suffix=".wav") as tmp_path:
                _extract_audio(audio_path, tmp_path, start=start, end=end)

                model = WhisperModel("tiny", device="cpu", compute_type="int8")
                segments_gen, _info = model.transcribe(tmp_path, beam_size=1, vad_filter=True)

                transcript_text = " ".join(seg.text for seg in segments_gen).lower()

                found = []
                for keyword in self.asmr_keywords:
                    if keyword.lower() in transcript_text:
                        found.append(keyword)

                if found:
                    logger.info("Detected trigger words in %.1f–%.1fs: %s", start, end, found)
                return found

        except Exception as exc:
            logger.debug("Failed to extract/transcribe audio for trigger words: %s", exc)
            return []
