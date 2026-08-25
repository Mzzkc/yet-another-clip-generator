"""Tests for the shared extraction and subtitle timing boundary."""

import pytest

from yacg.clip_boundaries import effective_clip_boundaries


def test_effective_context_caps_asmr_padding() -> None:
    boundaries = effective_clip_boundaries(
        10.0,
        20.0,
        120.0,
        context_padding=2.0,
        content_type="asmr",
    )

    assert boundaries.start_time == pytest.approx(9.7)
    assert boundaries.end_time == pytest.approx(20.3)


def test_effective_context_keeps_general_padding() -> None:
    boundaries = effective_clip_boundaries(
        10.0,
        20.0,
        120.0,
        context_padding=2.0,
        content_type="general",
    )

    assert boundaries.start_time == pytest.approx(8.0)
    assert boundaries.end_time == pytest.approx(22.0)


def test_effective_context_clamps_at_source_start_and_end() -> None:
    boundaries = effective_clip_boundaries(
        0.1,
        11.9,
        12.0,
        context_padding=2.0,
        content_type="asmr",
    )

    assert boundaries.start_time == pytest.approx(0.0)
    assert boundaries.end_time == pytest.approx(12.0)
