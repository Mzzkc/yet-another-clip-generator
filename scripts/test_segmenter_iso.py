#!/usr/bin/env python3
"""Tier-1 / Tier-4 isolation harness for yacg's segmenter.

Runs ONLY transcription + segmentation + boundary refinement.
No clip extraction, no subtitle burn, no caption generation.

Use cases:

  * **Tier 1 iteration** — bench segmenter changes (prompt tweaks,
    threshold tuning, model swap) without spinning the full clip-mill
    pipeline.  Same source, different config, see how boundaries
    move.
  * **Tier 4 ground-truth comparison** — when a
    ``<source>.ground-truth.yaml`` exists alongside the source
    video, compute recall / precision / mean boundary error against
    composer-marked good clips.

The harness is deliberately ALL-LOCAL — uses whatever model is
configured via ``--segmentation-model`` (or yacg defaults), no
frontier API calls.  This matches yacg's local-by-design constraint;
ground-truth files themselves can be hand-marked OR frontier-assisted
offline, but THIS harness never makes a cloud call.

Usage:

    # Plain run — print boundaries with context
    python3 scripts/test_segmenter_iso.py \\
        --source "/mnt/e/Finished Vids/furry hypno asmr ai takeover.mp4" \\
        --content-type asmr

    # With ground-truth comparison
    python3 scripts/test_segmenter_iso.py \\
        --source "/mnt/e/Finished Vids/furry hypno asmr ai takeover.mp4" \\
        --content-type asmr \\
        --ground-truth "/mnt/e/Finished Vids/furry hypno asmr ai takeover.ground-truth.yaml"

Ground-truth file shape:

    source: furry hypno asmr ai takeover.mp4
    good_clips:
      - {start: 232.5, end: 261.0, label: "ai-takeover-suggestion"}
      - {start: 380.0, end: 425.0, label: "deepening-induction"}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from yacg.models import PipelineConfig
from yacg.transcript_segmenter import TranscriptSegmenter


# ---------------------------------------------------------------------------
# Ground-truth loading + scoring
# ---------------------------------------------------------------------------


def _load_ground_truth(path: Path) -> list[dict]:
    """Load `<source>.ground-truth.yaml` and return list of good_clips."""
    text = path.read_text()
    data = yaml.safe_load(text) if path.suffix.lower() in (".yaml", ".yml") else json.loads(text)
    clips = data.get("good_clips") if isinstance(data, dict) else data
    if not isinstance(clips, list):
        raise ValueError(
            f"Ground-truth file must contain a list at 'good_clips' or top "
            f"level; got {type(clips).__name__}"
        )
    out: list[dict] = []
    for i, c in enumerate(clips):
        if isinstance(c, dict):
            out.append({
                "start": float(c["start"]),
                "end": float(c["end"]),
                "label": str(c.get("label", f"gt_{i + 1}")),
            })
        elif isinstance(c, (list, tuple)) and len(c) >= 2:
            out.append({
                "start": float(c[0]),
                "end": float(c[1]),
                "label": str(c[2]) if len(c) >= 3 else f"gt_{i + 1}",
            })
        else:
            raise ValueError(f"Unparseable ground-truth entry {i}: {c!r}")
    return out


def _score_against_ground_truth(
    system: list[tuple[float, float]],
    ground_truth: list[dict],
    boundary_tolerance: float = 1.0,
) -> dict:
    """Compute recall / precision / mean boundary error.

    A system clip "matches" a ground-truth clip when both boundaries
    are within ``boundary_tolerance`` seconds of the GT boundaries.
    Greedy matching: each GT clip pairs with at most one system clip.
    """
    matched_gt: set[int] = set()
    matched_sys: set[int] = set()
    boundary_errors: list[float] = []

    for gi, gt in enumerate(ground_truth):
        for si, (s_start, s_end) in enumerate(system):
            if si in matched_sys:
                continue
            start_err = abs(s_start - gt["start"])
            end_err = abs(s_end - gt["end"])
            if start_err <= boundary_tolerance and end_err <= boundary_tolerance:
                matched_gt.add(gi)
                matched_sys.add(si)
                boundary_errors.append((start_err + end_err) / 2)
                break

    recall = len(matched_gt) / len(ground_truth) if ground_truth else 0.0
    precision = len(matched_sys) / len(system) if system else 0.0
    mean_err = sum(boundary_errors) / len(boundary_errors) if boundary_errors else float("nan")
    return {
        "recall": recall,
        "precision": precision,
        "matched": len(matched_gt),
        "total_gt": len(ground_truth),
        "total_sys": len(system),
        "mean_boundary_error_s": mean_err,
        "boundary_tolerance_s": boundary_tolerance,
    }


# ---------------------------------------------------------------------------
# Pause-duration histogram (helps tune the bimodal threshold)
# ---------------------------------------------------------------------------


def _pause_histogram(words) -> dict:
    """Compute pause-duration distribution between consecutive words."""
    pauses = []
    for i in range(len(words) - 1):
        gap = words[i + 1].start - words[i].end
        if gap > 0.05:
            pauses.append(gap)
    if not pauses:
        return {"count": 0}

    pauses_sorted = sorted(pauses)
    bins = [
        ("<0.2s", sum(1 for p in pauses if p < 0.2)),
        ("0.2-0.4s", sum(1 for p in pauses if 0.2 <= p < 0.4)),
        ("0.4-0.6s", sum(1 for p in pauses if 0.4 <= p < 0.6)),
        ("0.6-0.9s", sum(1 for p in pauses if 0.6 <= p < 0.9)),
        ("0.9-1.5s", sum(1 for p in pauses if 0.9 <= p < 1.5)),
        ("1.5-3.0s", sum(1 for p in pauses if 1.5 <= p < 3.0)),
        (">=3.0s", sum(1 for p in pauses if p >= 3.0)),
    ]
    return {
        "count": len(pauses),
        "min": pauses_sorted[0],
        "median": pauses_sorted[len(pauses_sorted) // 2],
        "p90": pauses_sorted[int(len(pauses_sorted) * 0.9)],
        "max": pauses_sorted[-1],
        "bins": bins,
    }


# ---------------------------------------------------------------------------
# Context-text helpers
# ---------------------------------------------------------------------------


def _context_window(words, t: float, span_s: float = 5.0) -> str:
    """Concatenate words whose times fall within [t-span, t+span]."""
    lo = t - span_s
    hi = t + span_s
    chunks = [w.word for w in words if lo <= w.start <= hi]
    return " ".join(chunks).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tier-1 / Tier-4 yacg segmenter isolation harness",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to source video file",
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="Path to <source>.ground-truth.yaml for quantitative scoring "
             "(optional — if omitted, prints boundaries only)",
    )
    parser.add_argument(
        "--content-type",
        default="asmr",
        help="Content type (default: asmr)",
    )
    parser.add_argument(
        "--segmentation-model",
        default=None,
        help="Local Ollama model for segmentation (default: yacg's default)",
    )
    parser.add_argument(
        "--whisper-model",
        default=None,
        help="Local Whisper model size (default: yacg's default)",
    )
    parser.add_argument(
        "--pause-threshold",
        type=float,
        default=None,
        help="Pause threshold in seconds (default: yacg's default)",
    )
    parser.add_argument(
        "--min-segment-duration",
        type=float,
        default=None,
        help="Minimum segment duration in seconds",
    )
    parser.add_argument(
        "--max-segment-duration",
        type=float,
        default=None,
        help="Maximum segment duration in seconds",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=20,
        help="Target number of segments for the LLM to produce (default: 20)",
    )
    parser.add_argument(
        "--no-vad-filter",
        action="store_true",
        help="Disable VAD filter (recommended for ASMR)",
    )
    parser.add_argument(
        "--boundary-tolerance",
        type=float,
        default=1.0,
        help="Tolerance (s) for matching system to GT boundaries (default: 1.0)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("test_segmenter_iso")

    source = Path(args.source)
    if not source.is_file():
        logger.error("Source video does not exist: %s", source)
        return 2

    # Build a PipelineConfig with overrides
    cfg = PipelineConfig()
    cfg.content_profile.content_type = args.content_type
    if args.segmentation_model:
        cfg.segmentation_model = args.segmentation_model
    if args.whisper_model:
        cfg.whisper_model = args.whisper_model
    if args.pause_threshold is not None:
        cfg.pause_threshold = args.pause_threshold
    if args.min_segment_duration is not None:
        cfg.min_segment_duration = args.min_segment_duration
    if args.max_segment_duration is not None:
        cfg.max_segment_duration = args.max_segment_duration
    if args.no_vad_filter:
        cfg.vad_filter = False

    # Construct segmenter from PipelineConfig fields
    seg = TranscriptSegmenter(
        model_name=(cfg.segmentation_model or cfg.model_name),
        ollama_host=cfg.ollama_host,
        whisper_model=cfg.whisper_model,
        whisper_device=cfg.whisper_device,
        whisper_compute_type=cfg.whisper_compute_type,
        pause_threshold=cfg.pause_threshold,
        min_segment_duration=cfg.min_segment_duration,
        max_segment_duration=cfg.max_segment_duration,
        vad_filter=cfg.vad_filter,
        content_type=cfg.content_profile.content_type,
        channel_description=cfg.content_profile.channel_description,
        target_audience=cfg.content_profile.target_audience,
        custom_instructions=cfg.content_profile.custom_instructions,
    )

    logger.info("Step 1: transcribing %s", source)
    words = seg.full_transcribe(str(source))
    logger.info("Transcribed %d words", len(words))

    logger.info("Step 2: identifying segments via LLM (target ~%d)", args.target_count)
    boundaries = seg.segment_by_content(
        words, title=source.stem, target_count=args.target_count,
    )
    logger.info("LLM returned %d candidate segments", len(boundaries))

    logger.info("Step 3: refining boundaries with pause-snap")
    # Pass video_path through so refine_boundaries runs stage B
    # (acoustic RMS local-minimum snap).  See transcript_segmenter.py
    # for details — required for sub-word precision on ASMR/whispered
    # cadence content.
    refined = seg.refine_boundaries(boundaries, words, video_path=str(source))
    logger.info("Refined to %d segments", len(refined))

    # Pause histogram
    print()
    print("=" * 80)
    print("PAUSE-DURATION HISTOGRAM (helps tune bimodal threshold)")
    print("=" * 80)
    h = _pause_histogram(words)
    if h.get("count", 0) > 0:
        print(f"  count: {h['count']}  min: {h['min']:.2f}s  median: {h['median']:.2f}s  "
              f"p90: {h['p90']:.2f}s  max: {h['max']:.2f}s")
        for label, n in h["bins"]:
            bar = "█" * min(60, int(n * 60 / max(1, h["count"])))
            print(f"    {label:>8}: {n:>5}  {bar}")
    else:
        print("  no significant pauses detected")

    # Boundary report with surrounding transcript
    print()
    print("=" * 80)
    print(f"REFINED SEGMENTS ({len(refined)})")
    print("=" * 80)
    system_boundaries = [(s.start_time, s.end_time) for s in refined]
    for i, (s_start, s_end) in enumerate(system_boundaries, start=1):
        duration = s_end - s_start
        before = _context_window(words, s_start, span_s=5.0)
        after = _context_window(words, s_end, span_s=5.0)
        print(f"\n  Clip {i:02d}  [{s_start:.2f}s → {s_end:.2f}s, {duration:.2f}s]")
        print(f"    START context (±5s): {before[:200]}")
        print(f"    END   context (±5s): {after[:200]}")

    # Optional ground-truth comparison
    if args.ground_truth:
        gt_path = Path(args.ground_truth)
        if not gt_path.is_file():
            logger.error("Ground-truth file does not exist: %s", gt_path)
            return 2
        ground_truth = _load_ground_truth(gt_path)
        score = _score_against_ground_truth(
            system_boundaries, ground_truth, args.boundary_tolerance,
        )
        print()
        print("=" * 80)
        print(f"GROUND-TRUTH COMPARISON ({gt_path.name})")
        print("=" * 80)
        print(f"  Recall:    {score['recall'] * 100:5.1f}%  ({score['matched']}/{score['total_gt']} GT clips matched within ±{score['boundary_tolerance_s']}s)")
        print(f"  Precision: {score['precision'] * 100:5.1f}%  ({score['matched']}/{score['total_sys']} system clips near a GT)")
        if score["matched"] > 0:
            print(f"  Mean boundary error: {score['mean_boundary_error_s']:.3f}s")
        else:
            print("  Mean boundary error: n/a (no matches)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
