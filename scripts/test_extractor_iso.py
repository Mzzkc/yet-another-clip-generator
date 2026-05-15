#!/usr/bin/env python3
"""Tier-0 isolation test for yacg's ClipExtractor.

This script answers ONE question: does yacg's clip extractor produce
good output when given known-good boundaries?

If composer hand-picks 3 boundaries that are KNOWN to be coherent
hypnotic suggestions, and yacg extracts them with ``context_padding=0``,
the resulting clips should sound exactly like what composer would
hand-cut.  If they don't, the extractor itself has issues that no
amount of better segmentation can fix — debug those first.

Usage:

    python3 scripts/test_extractor_iso.py \\
        --source "/mnt/e/Finished Vids/furry hypno asmr ai takeover.mp4" \\
        --output-dir /tmp/yacg-tier0 \\
        --boundaries-file /tmp/tier0-boundaries.yaml

Boundaries file (YAML or JSON, suffix-detected) is a list of objects
or arrays:

    # YAML
    - {start: 232.5, end: 261.0, label: "ai-takeover-suggestion"}
    - {start: 380.0, end: 425.0, label: "deepening-induction"}

    // JSON
    [[232.5, 261.0, "ai-takeover-suggestion"], [380.0, 425.0, "deepening"]]

The output directory will contain N MP4 files named
``tier0_NN_<label>_<start>-<end>.mp4`` and a small report.txt
summarizing what was extracted, with what padding, into which file.

This script is read-only against yacg source — no patches, no
changes to PipelineConfig defaults.  It overrides context_padding to
0 explicitly so the boundaries the composer marked are exactly what
plays.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

# yacg imports — assumes pip install -e . done in /home/emzi/Projects/yacg
from yacg.extractors.clip_extractor import ClipExtractor
from yacg.models import PipelineConfig


def _parse_boundaries_file(path: Path) -> list[dict]:
    """Parse boundaries from a YAML or JSON file.

    Each entry is either a dict ``{start, end, label}`` or a list/tuple
    of ``[start, end, label]``.  Returns a normalized list of dicts.
    """
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        # Try YAML first (it accepts JSON syntax too), fall back to JSON
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            data = json.loads(text)

    if not isinstance(data, list):
        raise ValueError(
            f"Boundaries file must contain a list at top level; got {type(data).__name__}"
        )

    boundaries: list[dict] = []
    for i, item in enumerate(data):
        if isinstance(item, dict):
            start = float(item["start"])
            end = float(item["end"])
            label = str(item.get("label", f"clip_{i + 1}"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start = float(item[0])
            end = float(item[1])
            label = str(item[2]) if len(item) >= 3 else f"clip_{i + 1}"
        else:
            raise ValueError(f"Unparseable boundary at index {i}: {item!r}")
        if end <= start:
            raise ValueError(
                f"Boundary {label}: end ({end}) must be > start ({start})"
            )
        boundaries.append({"start": start, "end": end, "label": label})
    return boundaries


def _slugify(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in s).strip("-").lower()[:48]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tier-0 yacg extractor isolation test",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to source video file",
    )
    parser.add_argument(
        "--boundaries-file",
        required=True,
        help="Path to YAML/JSON file with list of {start, end, label} boundaries",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/yacg-tier0",
        help="Where to write extracted clips (default: /tmp/yacg-tier0)",
    )
    parser.add_argument(
        "--context-padding",
        type=float,
        default=0.0,
        help="Padding seconds (default: 0.0 — boundaries play exactly as marked)",
    )
    parser.add_argument(
        "--vlm-crop",
        action="store_true",
        help="Use VLM-based cropping (requires Ollama + qwen3-vl)",
    )
    parser.add_argument(
        "--model",
        default="qwen3-vl:8b-instruct",
        help="VLM model name (only used with --vlm-crop)",
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
    logger = logging.getLogger("test_extractor_iso")

    source = Path(args.source)
    if not source.is_file():
        logger.error("Source video does not exist: %s", source)
        return 2

    bfile = Path(args.boundaries_file)
    if not bfile.is_file():
        logger.error("Boundaries file does not exist: %s", bfile)
        return 2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", out_dir)

    boundaries = _parse_boundaries_file(bfile)
    logger.info("Parsed %d boundaries from %s", len(boundaries), bfile)

    # Build a PipelineConfig with explicit padding override + content_type
    cfg = PipelineConfig()
    cfg.context_padding = float(args.context_padding)
    cfg.vlm_crop = bool(args.vlm_crop)
    if args.vlm_crop:
        cfg.model_name = args.model
    cfg.content_profile.content_type = "asmr"

    # Construct ClipExtractor with padding=0.  The constructor takes the
    # padding from `config` when both are passed, so the override is honored.
    extractor = ClipExtractor(context_padding=cfg.context_padding, config=cfg)

    report_lines: list[str] = []
    report_lines.append("# Tier 0 extractor isolation test")
    report_lines.append("")
    report_lines.append(f"Source: {source}")
    report_lines.append(f"Padding: {cfg.context_padding}s")
    report_lines.append(f"VLM crop: {cfg.vlm_crop}")
    if cfg.vlm_crop:
        report_lines.append(f"VLM model: {cfg.model_name}")
    report_lines.append("")

    successes = 0
    for i, b in enumerate(boundaries, start=1):
        slug = _slugify(b["label"])
        out_path = out_dir / f"tier0_{i:02d}_{slug}_{b['start']:.1f}-{b['end']:.1f}.mp4"
        duration = b["end"] - b["start"]
        logger.info(
            "Extracting clip %d/%d: %s [%.2f → %.2f, %.2fs] → %s",
            i, len(boundaries), b["label"], b["start"], b["end"], duration, out_path.name,
        )
        ok = extractor.extract_clip(
            video_path=str(source),
            start_time=b["start"],
            end_time=b["end"],
            output_path=str(out_path),
        )
        if ok and out_path.exists():
            successes += 1
            size_mb = out_path.stat().st_size / 1024 / 1024
            report_lines.append(
                f"OK   {out_path.name}  ({size_mb:.1f} MB, {duration:.2f}s)"
            )
        else:
            report_lines.append(
                f"FAIL {out_path.name}  (start={b['start']}, end={b['end']})"
            )

    report_lines.append("")
    report_lines.append(f"Result: {successes}/{len(boundaries)} clips extracted")
    report = "\n".join(report_lines)

    report_path = out_dir / "report.txt"
    report_path.write_text(report + "\n")
    print()
    print(report)
    print()
    print(f"Composer ear-check: play each .mp4 in {out_dir}/ and decide:")
    print("  GOOD → extractor is sound; segmentation IS the problem; proceed to tier 1")
    print("  BAD  → extractor has issues; debug those before tier 1")

    return 0 if successes == len(boundaries) else 1


if __name__ == "__main__":
    sys.exit(main())
