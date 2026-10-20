"""Command line interface: ``cpipe``.

Examples::

    cpipe run --video data/videos/room1.mp4 --output data/runs/room1 --preset balanced
    cpipe run --video bottom.mp4 --video middle.mp4 --video up.mp4 --output runs/room2
    cpipe run --videos data/videos/room2 --output data/runs/room2      # all videos in a folder
    cpipe run --video room1.mp4 --output runs/room1 --until poses     # laptop, no GPU
    cpipe run --video room1.mp4 --output runs/room1 --set poses.matcher=exhaustive
    cpipe run --output runs/room1        # continue a run folder (e.g. on Colab), no videos
    cpipe check --output runs/room1      # scene.ply and cameras.json in the same coordinates?
    cpipe evaluate --output runs/room1   # PSNR/SSIM/LPIPS on held-out frames (GPU)
    cpipe config --preset fast                                         # show settings
    cpipe ui                                                           # local web page
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
from colibrimo_pipeline.utils.videos import collect_videos


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

    run = sub.add_parser("run", help="Run the pipeline on the video(s) of one room.")
    run.add_argument(
        "--video",
        action="append",
        default=[],
        type=Path,
        help="Input video file. Repeat for several videos of the same room "
        "(e.g. filmed at three heights). Without --video/--videos, the --output "
        "run folder is continued with the videos its manifest records.",
    )
    run.add_argument(
        "--videos",
        type=Path,
        metavar="FOLDER",
        help="Use every video file in this folder (sorted by name).",
    )
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

    check = sub.add_parser(
        "check",
        help="Check a finished run: coordinate check and cameras.json (exit 1 if not OK).",
    )
    check.add_argument("--output", required=True, type=Path, help="Run folder.")

    evaluate = sub.add_parser(
        "evaluate",
        help="Score a finished run's scene.ply on its held-out frames (PSNR, SSIM, "
        "LPIPS); needs gsplat + CUDA (e.g. a Brush splat, scored on Colab).",
    )
    evaluate.add_argument("--output", required=True, type=Path, help="Run folder.")

    config = sub.add_parser("config", help="Print the resolved configuration as YAML.")
    _add_config_arguments(config)

    ui = sub.add_parser(
        "ui",
        help="Local web page to upload videos and run the pipeline (this computer only).",
    )
    ui.add_argument("--port", type=int, default=8765, help="Default: 8765.")
    ui.add_argument(
        "--data",
        type=Path,
        default=Path("data"),
        help="Folder for uploads/ and runs/. Default: ./data",
    )
    ui.add_argument(
        "--viewer",
        type=Path,
        help="Viewer project folder (with public/). Default: CPIPE_VIEWER_DIR, "
        "else ~/3DViewing if it exists.",
    )
    ui.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a run setting (default: Brush, max 1,000,000 splats, "
        "max resolution 1280).",
    )
    ui.add_argument(
        "--no-browser", action="store_true", help="Don't open the page automatically."
    )
    return parser


def _ui(args: argparse.Namespace) -> int:
    from colibrimo_pipeline.ui.jobs import default_viewer_dir, ui_overrides
    from colibrimo_pipeline.ui.server import serve

    overrides = ui_overrides(parse_set_overrides(args.set))
    load_config("balanced", overrides)  # fail now (exit 2) if --set is invalid
    serve(
        args.data,
        overrides,
        args.viewer or default_viewer_dir(),
        port=args.port,
        open_browser=not args.no_browser,
    )
    return 0


def _check(run_dir: Path) -> int:
    from colibrimo_pipeline.check import check_run, describe

    result = check_run(run_dir)
    print("\n".join(describe(result)))
    return 0 if result["ok"] else 1


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
        if args.command == "ui":
            return _ui(args)
        if args.command == "check":
            return _check(args.output)
        if args.command == "evaluate":
            from colibrimo_pipeline.evaluate import describe, evaluate_existing_run

            print("Score:", describe(evaluate_existing_run(args.output)))
            return 0
        config = load_config(args.preset, _overrides(args))
        if args.command == "config":
            print(
                yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), end=""
            )
            return 0

        videos = (
            [v.path for v in collect_videos(args.video, args.videos)]
            if args.video or args.videos
            else None  # continue the run folder
        )
        manifest = run_pipeline(
            videos,
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
