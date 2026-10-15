"""Tests for the remote Whisper backend (yacg.remote_whisper + TranscriptSegmenter wiring)."""

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from yacg import remote_whisper
from yacg.models import PipelineConfig, WordTimestamp
from yacg.transcript_segmenter import TranscriptSegmenter

SEGMENT_WORDS = {
    "language": "en", "language_probability": 0.97, "duration": 3.0,
    "segments": [
        {"id": 1, "start": 0.0, "end": 1.4, "text": " Hello there.",
         "words": [{"word": " Hello", "start": 0.0, "end": 0.6, "probability": 0.99},
                   {"word": " there.", "start": 0.6, "end": 1.4, "probability": 0.91}]},
        {"id": 2, "start": 1.8, "end": 3.0, "text": " Bye.",
         "words": [{"word": " Bye.", "start": 1.8, "end": 3.0, "probability": 0.88}]},
    ],
}

TOP_LEVEL_WORDS = {
    "language": "english", "duration": 3.0,
    "segments": [{"id": 0, "start": 0.0, "end": 3.0, "text": "Hello there. Bye."}],
    "words": [{"word": "Hello", "start": 0.0, "end": 0.6}, {"word": "there.", "start": 0.6, "end": 1.4},
              {"word": "Bye.", "start": 1.8, "end": 3.0}],
}


class FakeResponse:
    def __init__(self, payload=None, status=200, text=""):
        self._payload, self.status_code, self.text = payload, status, text

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def post(self, url, data=None, files=None, headers=None, timeout=None):
        upload = files["file"]
        self.calls.append({"url": url, "data": data, "headers": headers, "filename": upload[0],
                           "content": upload[1].read(), "content_type": upload[2], "timeout": timeout})
        if self.error:
            raise self.error
        return self.response


@pytest.fixture(autouse=True)
def fake_audio_extract(monkeypatch):
    """Skip ffmpeg: write a stand-in FLAC file."""
    def extract(video_path, out_path):
        Path(out_path).write_bytes(b"fLaC fake audio")
    monkeypatch.setattr(remote_whisper, "_extract_audio", extract)
    monkeypatch.delenv("WHISPER_API_URL", raising=False)
    monkeypatch.delenv("WHISPER_API_KEY", raising=False)


class TestEndpoint:
    def test_base_url_with_or_without_v1(self):
        assert remote_whisper.transcriptions_endpoint("http://gpu:8000") == "http://gpu:8000/v1/audio/transcriptions"
        assert remote_whisper.transcriptions_endpoint("http://gpu:8000/v1/") == "http://gpu:8000/v1/audio/transcriptions"

    def test_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("WHISPER_API_URL", "http://env-host:9000")
        assert remote_whisper.transcriptions_endpoint("") == "http://env-host:9000/v1/audio/transcriptions"

    def test_missing_url_is_a_clear_error(self):
        with pytest.raises(RuntimeError, match="WHISPER_API_URL"):
            remote_whisper.transcriptions_endpoint("")


class TestTranscribe:
    def test_maps_segment_words_and_sends_expected_request(self, monkeypatch):
        monkeypatch.setenv("WHISPER_API_KEY", "k123")
        session = FakeSession(FakeResponse(SEGMENT_WORDS))
        result = remote_whisper.transcribe("video.mp4", url="http://gpu:8000", model="medium",
                                           vad_filter=False, session=session)
        assert result.language == "en" and result.language_probability == 0.97
        assert result.words == [WordTimestamp("Hello", 0.0, 0.6, 0.99), WordTimestamp("there.", 0.6, 1.4, 0.91),
                                WordTimestamp("Bye.", 1.8, 3.0, 0.88)]
        call = session.calls[0]
        assert call["url"] == "http://gpu:8000/v1/audio/transcriptions"
        assert ("model", "medium") in call["data"] and ("vad_filter", "false") in call["data"]
        assert ("timestamp_granularities[]", "word") in call["data"]
        assert call["headers"] == {"Authorization": "Bearer k123"}
        assert call["content_type"] == "audio/flac" and call["content"].startswith(b"fLaC")

    def test_top_level_words_and_missing_probability(self):
        session = FakeSession(FakeResponse(TOP_LEVEL_WORDS))
        result = remote_whisper.transcribe("video.mp4", url="http://gpu:8000", model="small",
                                           vad_filter=True, session=session)
        assert [w.word for w in result.words] == ["Hello", "there.", "Bye."]
        assert all(w.probability is None for w in result.words)
        assert result.language_probability is None

    def test_http_error(self):
        session = FakeSession(FakeResponse(status=500, text="model exploded"))
        with pytest.raises(RuntimeError, match="HTTP 500: model exploded"):
            remote_whisper.transcribe("v.mp4", url="http://gpu:8000", model="small", vad_filter=True, session=session)

    def test_unreachable_server(self):
        session = FakeSession(error=requests.ConnectionError("refused"))
        with pytest.raises(RuntimeError, match="Could not reach.*whisper_backend back to 'local'"):
            remote_whisper.transcribe("v.mp4", url="http://gpu:8000", model="small", vad_filter=True, session=session)


class TestSegmenterRemotePath:
    def make(self):
        return TranscriptSegmenter(whisper_model="small", whisper_backend="remote",
                                   whisper_url="http://gpu:8000", vad_filter=False)

    def test_returns_words_without_loading_faster_whisper(self):
        seg = self.make()
        result = remote_whisper.RemoteTranscript("en", 0.95, [WordTimestamp("Hi", 0.0, 0.4, 0.9)])
        with patch.object(remote_whisper, "transcribe", return_value=result) as fake, \
             patch.dict("sys.modules", {"faster_whisper": None}):
            assert seg.full_transcribe("video.mp4") == result.words
        assert fake.call_args.kwargs == {"url": "http://gpu:8000", "model": "small",
                                         "vad_filter": False, "session": seg._session}

    def test_low_language_confidence_still_gated(self):
        result = remote_whisper.RemoteTranscript("en", 0.3, [WordTimestamp("Hi", 0.0, 0.4, 0.9)])
        with patch.object(remote_whisper, "transcribe", return_value=result):
            with pytest.raises(RuntimeError, match="confidence too low"):
                self.make().full_transcribe("video.mp4")

    def test_missing_language_probability_is_not_gated(self):
        result = remote_whisper.RemoteTranscript("english", None, [WordTimestamp("Hi", 0.0, 0.4, None)])
        with patch.object(remote_whisper, "transcribe", return_value=result):
            assert len(self.make().full_transcribe("video.mp4")) == 1

    def test_no_words_raises(self):
        result = remote_whisper.RemoteTranscript("en", 0.99, [])
        with patch.object(remote_whisper, "transcribe", return_value=result):
            with pytest.raises(RuntimeError, match="No speech detected"):
                self.make().full_transcribe("video.mp4")


class TestConfig:
    def test_default_backend_is_local(self):
        assert PipelineConfig().whisper_backend == "local"

    def test_invalid_backend_rejected(self):
        with pytest.raises(ValueError, match="whisper_backend"):
            PipelineConfig(whisper_backend="cloud")

    def test_ini_round_trip(self, tmp_path):
        from yacg.utils.config import load_config, save_default_config
        path = tmp_path / "yacg.ini"
        save_default_config(str(path))
        text = path.read_text().replace("whisper_backend = local", "whisper_backend = remote")
        text = text.replace("whisper_url = \n", "whisper_url = http://gpu:8000\n")
        path.write_text(text)
        config = load_config(str(path))
        assert (config.whisper_backend, config.whisper_url) == ("remote", "http://gpu:8000")

    def test_cli_flags_override_config(self):
        from yacg.cli import _build_config
        args = argparse.Namespace(whisper_backend="remote", whisper_url="http://gpu:8000", config=None)
        config = _build_config(args)
        assert (config.whisper_backend, config.whisper_url) == ("remote", "http://gpu:8000")
