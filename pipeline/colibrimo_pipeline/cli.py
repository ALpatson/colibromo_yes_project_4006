"""Command line interface: ``cpipe``.

Examples::

    cpipe run --video data/videos/room1.mp4 --output data/runs/room1 --preset balanced
    cpipe run --video room1.mp4 --output runs/room1 --until poses     # laptop, no GPU
    cpipe run --video room1.mp4 --output runs/room1 --set poses.matcher=exhaustive
    cpipe config --preset fast                                         # show settings
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from colibrimo_pipeline import __version__
from colibrimo_pipeline.config import (
    available_presets,
    deep_merge,
    load_config,
    parse_set_overrides,
)
from colibrimo_pipeline.errors import PipelineError, StepError
from colibrimo_pipeline.pipeline import STEP_NAMES, run_pipeline


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--preset",
        default="balanced",
        help=f"Preset name ({', '.join(available_presets())}) or path to a YAML file. "
        "Default: balanced.",
    )
    parser.add_argument("--frames", type=int, help="Override frames.target_count.")
    parser.add_argument(
        "--steps", type=int, dest="max_steps", help="Override train.max_steps."
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any setting, e.g. --set poses.matcher=exhaustive (repeatable).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cpipe",
        description="Colibrimo: turn a room video into a 3D Gaussian Splat.",
    )
    parser.add_argument("--version", action="version", version=f"cpipe {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the pipeline on a video.")
    run.add_argument("--video", required=True, type=Path, help="Input video file.")
    run.add_argument(
        "--output", required=True, type=Path, help="Run folder (created if needed)."
    )
    _add_config_arguments(run)
    run.add_argument(
        "--force", action="store_true", help="Re-run all steps, even completed ones."
    )
    run.add_argument(
        "--from-step", choices=STEP_NAMES, help="Re-run from this step onwards."
    )
    run.add_argument(
        "--until",
        choices=STEP_NAMES,
        help="Stop after this step (e.g. '--until poses' on a machine without GPU).",
    )
    run.add_argument(
        "-v", "--verbose", action="store_true", help="Show all tool output."
    )

    config = sub.add_parser("config", help="Print the resolved configuration as YAML.")
    _add_config_arguments(config)
    return parser


def _overrides(args: argparse.Namespace) -> dict:
    overrides = parse_set_overrides(args.set)
    if args.frames is not None:
        overrides = deep_merge(overrides, {"frames": {"target_count": args.frames}})
    if args.max_steps is not None:
        overrides = deep_merge(overrides, {"train": {"max_steps": args.max_steps}})
    return overrides


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.preset, _overrides(args))
        if args.command == "config":
            print(
                yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), end=""
            )
            return 0

        manifest = run_pipeline(
            args.video,
            args.output,
            config,
            force=args.force,
            from_step=args.from_step,
            until=args.until,
            verbose=args.verbose,
        )
    except StepError as exc:
        print(
            f"\nRun failed at step '{exc.step}'. Details: {args.output / 'logs' / exc.step}.log",
            file=sys.stderr,
        )
        return 1
    except PipelineError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"\nStatus: {manifest.data['status']}")
    if "scene_ply" in manifest.data.get("outputs", {}):
        print(f"Scene:    {manifest.run_dir / manifest.data['outputs']['scene_ply']}")
    print(f"Manifest: {manifest.path}")
    return 0
