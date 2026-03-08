"""Tests for bootstrap.py to boost coverage."""

from unittest.mock import MagicMock, patch

import pytest


class TestGetOsInfo:
    """Cover get_os_info() OS detection branches."""

    def test_get_os_info_returns_dict(self):
        from yacg.bootstrap import get_os_info

        info = get_os_info()
        assert isinstance(info, dict)
        assert "system" in info
        assert "distro" in info
        assert "pkg_manager" in info
        assert "wsl" in info

    @patch("platform.system", return_value="Linux")
    @patch(
        "builtins.open",
        side_effect=[
            # /etc/os-release
            type("", (), {"__enter__": lambda s: s, "__exit__": lambda *a: None, "read": lambda s: "ID=ubuntu\n"})(),
            # /proc/version
            type("", (), {"__enter__": lambda s: s, "__exit__": lambda *a: None, "read": lambda s: "Microsoft\n"})(),
        ],
    )
    def test_get_os_info_debian_wsl(self, mock_open, mock_system):
        from yacg.bootstrap import get_os_info

        info = get_os_info()
        assert info["distro"] == "debian"
        assert info["pkg_manager"] == "apt"
        assert info["wsl"] is True

    @patch("platform.system", return_value="Darwin")
    def test_get_os_info_macos(self, mock_system):
        from yacg.bootstrap import get_os_info

        info = get_os_info()
        assert info["distro"] == "macos"
        assert info["pkg_manager"] == "brew"


class TestCheckFfmpeg:
    """Cover check_ffmpeg()."""

    def test_check_ffmpeg_present(self):
        from yacg.bootstrap import check_ffmpeg

        # We know ffmpeg is installed in this environment
        assert check_ffmpeg() is True

    @patch("shutil.which", return_value=None)
    def test_check_ffmpeg_missing(self, mock_which):
        from yacg.bootstrap import check_ffmpeg

        assert check_ffmpeg() is False


class TestPrintFfmpegInstructions:
    """Cover print_ffmpeg_instructions() branches."""

    def test_print_for_debian(self, capsys):
        from yacg.bootstrap import print_ffmpeg_instructions

        print_ffmpeg_instructions({"system": "linux", "distro": "debian"})
        captured = capsys.readouterr()
        assert "apt" in captured.out

    def test_print_for_unknown(self, capsys):
        from yacg.bootstrap import print_ffmpeg_instructions

        print_ffmpeg_instructions({"system": "linux", "distro": None})
        captured = capsys.readouterr()
        assert "Install ffmpeg" in captured.out

    def test_print_for_windows(self, capsys):
        from yacg.bootstrap import print_ffmpeg_instructions

        print_ffmpeg_instructions({"system": "windows", "distro": None})
        captured = capsys.readouterr()
        assert "scoop" in captured.out or "choco" in captured.out


class TestPipInstall:
    """Cover _pip_install() fallback strategies."""

    @patch("subprocess.run")
    def test_pip_install_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        from yacg.bootstrap import _pip_install

        assert _pip_install(["fake-package"]) is True

    @patch("subprocess.run", side_effect=Exception("fail"))
    def test_pip_install_all_fail(self, mock_run):
        from yacg.bootstrap import _pip_install

        assert _pip_install(["fake-package"]) is False


class TestEnsurePythonDeps:
    """Cover ensure_python_deps()."""

    def test_all_deps_present(self):
        from yacg.bootstrap import ensure_python_deps

        # All required deps should be present in this environment
        assert ensure_python_deps() is True


class TestCheckOllama:
    """Cover check_ollama()."""

    def test_check_ollama_returns_dict(self):
        from yacg.bootstrap import check_ollama

        result = check_ollama()
        assert isinstance(result, dict)
        assert "available" in result
        assert "models" in result


class TestEnsureReady:
    """Cover ensure_ready() flow."""

    def test_ensure_ready_succeeds(self):
        import yacg.bootstrap as bs

        # Reset the cached flag to exercise the full path
        original = bs._bootstrapped
        bs._bootstrapped = False
        try:
            result = bs.ensure_ready(verbose=True)
            assert result is True
        finally:
            bs._bootstrapped = original

    def test_ensure_ready_cached(self):
        import yacg.bootstrap as bs

        bs._bootstrapped = True
        try:
            assert bs.ensure_ready(verbose=True) is True
        finally:
            bs._bootstrapped = False

    @patch("yacg.bootstrap.check_ffmpeg", return_value=False)
    def test_ensure_ready_missing_ffmpeg(self, mock_ffmpeg):
        import yacg.bootstrap as bs

        bs._bootstrapped = False
        try:
            result = bs.ensure_ready(verbose=True)
            assert result is False
        finally:
            bs._bootstrapped = False

    @patch("yacg.bootstrap.ensure_python_deps", return_value=False)
    def test_ensure_ready_missing_python_deps(self, mock_deps):
        import yacg.bootstrap as bs

        bs._bootstrapped = False
        try:
            result = bs.ensure_ready(verbose=True)
            assert result is False
        finally:
            bs._bootstrapped = False
