"""Tests for youtube_downloader.py to boost coverage."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


class TestYouTubeDownloader:
    """Cover YouTubeDownloader methods."""

    def test_extract_video_id_standard(self):
        from yacg.youtube_downloader import YouTubeDownloader

        with tempfile.TemporaryDirectory() as tmpdir:
            dl = YouTubeDownloader(output_dir=tmpdir)
            assert dl.extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_extract_video_id_short_url(self):
        from yacg.youtube_downloader import YouTubeDownloader

        with tempfile.TemporaryDirectory() as tmpdir:
            dl = YouTubeDownloader(output_dir=tmpdir)
            assert dl.extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_extract_video_id_shorts(self):
        from yacg.youtube_downloader import YouTubeDownloader

        with tempfile.TemporaryDirectory() as tmpdir:
            dl = YouTubeDownloader(output_dir=tmpdir)
            assert dl.extract_video_id("https://youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_extract_video_id_embed(self):
        from yacg.youtube_downloader import YouTubeDownloader

        with tempfile.TemporaryDirectory() as tmpdir:
            dl = YouTubeDownloader(output_dir=tmpdir)
            assert dl.extract_video_id("https://youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_extract_video_id_bare(self):
        from yacg.youtube_downloader import YouTubeDownloader

        with tempfile.TemporaryDirectory() as tmpdir:
            dl = YouTubeDownloader(output_dir=tmpdir)
            assert dl.extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_extract_video_id_invalid(self):
        from yacg.youtube_downloader import YouTubeDownloader

        with tempfile.TemporaryDirectory() as tmpdir:
            dl = YouTubeDownloader(output_dir=tmpdir)
            assert dl.extract_video_id("not-a-valid-url") == ""

    def test_download_invalid_url_raises(self):
        from yacg.youtube_downloader import YouTubeDownloader

        with tempfile.TemporaryDirectory() as tmpdir:
            dl = YouTubeDownloader(output_dir=tmpdir)
            with pytest.raises(ValueError, match="Could not parse"):
                dl.download("not-a-youtube-url")

    @patch("yt_dlp.YoutubeDL")
    def test_download_success(self, mock_ydl_cls):
        from yacg.youtube_downloader import YouTubeDownloader

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake downloaded file
            fake_path = os.path.join(tmpdir, "dQw4w9WgXcQ.mp4")
            with open(fake_path, "w") as f:
                f.write("fake video")

            mock_ydl = MagicMock()
            mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
            mock_ydl.__exit__ = MagicMock(return_value=False)
            mock_ydl.extract_info.return_value = {
                "title": "Rick Astley",
                "duration": 212,
                "uploader": "RickAstleyVEVO",
            }
            mock_ydl_cls.return_value = mock_ydl

            dl = YouTubeDownloader(output_dir=tmpdir)
            result = dl.download("https://youtube.com/watch?v=dQw4w9WgXcQ")

            assert result["title"] == "Rick Astley"
            assert result["duration"] == 212.0
            assert result["channel"] == "RickAstleyVEVO"
            assert result["video_path"] == fake_path

    @patch("yt_dlp.YoutubeDL")
    def test_download_no_info_raises(self, mock_ydl_cls):
        from yacg.youtube_downloader import YouTubeDownloader

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_ydl = MagicMock()
            mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
            mock_ydl.__exit__ = MagicMock(return_value=False)
            mock_ydl.extract_info.return_value = None
            mock_ydl_cls.return_value = mock_ydl

            dl = YouTubeDownloader(output_dir=tmpdir)
            with pytest.raises(RuntimeError, match="no info"):
                dl.download("https://youtube.com/watch?v=dQw4w9WgXcQ")

    @patch("yt_dlp.YoutubeDL")
    def test_download_no_file_found_raises(self, mock_ydl_cls):
        from yacg.youtube_downloader import YouTubeDownloader

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_ydl = MagicMock()
            mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
            mock_ydl.__exit__ = MagicMock(return_value=False)
            mock_ydl.extract_info.return_value = {"title": "Test", "duration": 10}
            mock_ydl_cls.return_value = mock_ydl

            dl = YouTubeDownloader(output_dir=tmpdir)
            with pytest.raises(RuntimeError, match="no file found"):
                dl.download("https://youtube.com/watch?v=dQw4w9WgXcQ")

    @patch("yt_dlp.YoutubeDL")
    def test_download_yt_dlp_error_raises(self, mock_ydl_cls):
        import yt_dlp
        from yacg.youtube_downloader import YouTubeDownloader

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_ydl = MagicMock()
            mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
            mock_ydl.__exit__ = MagicMock(return_value=False)
            mock_ydl.extract_info.side_effect = yt_dlp.utils.DownloadError("nope")
            mock_ydl_cls.return_value = mock_ydl

            dl = YouTubeDownloader(output_dir=tmpdir)
            with pytest.raises(RuntimeError, match="Download failed"):
                dl.download("https://youtube.com/watch?v=dQw4w9WgXcQ")

    def test_find_downloaded_file_ignores_part(self):
        from yacg.youtube_downloader import YouTubeDownloader

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a .part file — should be ignored
            part_path = os.path.join(tmpdir, "dQw4w9WgXcQ.mp4.part")
            with open(part_path, "w") as f:
                f.write("partial")

            dl = YouTubeDownloader(output_dir=tmpdir)
            assert dl._find_downloaded_file("dQw4w9WgXcQ") is None

            # Now create the real file
            real_path = os.path.join(tmpdir, "dQw4w9WgXcQ.mp4")
            with open(real_path, "w") as f:
                f.write("video")
            assert dl._find_downloaded_file("dQw4w9WgXcQ") == real_path
