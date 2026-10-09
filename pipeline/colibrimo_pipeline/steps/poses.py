"""Step 3: estimate camera poses with COLMAP (features -> matching -> mapper).

COLMAP is called directly rather than through Nerfstudio's ``ns-process-data``
(see docs/decisions.md, D-006). Option names changed between COLMAP 3.x and 4.x
(e.g. ``SiftExtraction.use_gpu`` -> ``FeatureExtraction.use_gpu``), so each
option is looked up in ``colmap <command> --help`` and the supported name is used.

Output layout (what gsplat's trainer expects)::

    colmap/images/        frames (from extract_frames)
    colmap/database.db    COLMAP features and matches
    colmap/sparse/0/      cameras.bin, images.bin, points3D.bin (largest model)
"""

from __future__ import annotations

import logging
import os
import shutil
import sys

from colibrimo_pipeline.errors import StepError
from colibrimo_pipeline.steps.base import CAPTURE_TIPS, StepContext, StepResult
from colibrimo_pipeline.utils.colmap_model import (
    parse_help_options,
    pick_option,
    promote_best_model,
    read_entry_count,
)
from colibrimo_pipeline.utils.proc import find_tool, run_capture, run_tool
from colibrimo_pipeline.utils.versions import colmap_version

NAME = "poses"


class Colmap:
    """Thin wrapper around the ``colmap`` executable."""

    def __init__(self, exe: str, logger: logging.Logger):
        self.exe = exe
        self.logger = logger
        self._options: dict[str, set[str]] = {}
        self.env = dict(os.environ)
        if sys.platform != "win32":
            # COLMAP links Qt; without a display it needs the offscreen platform.
            self.env.setdefault("QT_QPA_PLATFORM", "offscreen")

    def options(self, command: str) -> set[str]:
        if command not in self._options:
            result = run_capture([self.exe, command, "--help"])
            self._options[command] = parse_help_options(result.stdout + result.stderr)
        return self._options[command]

    def add_option(
        self, args: list[str], command: str, candidates: list[str], value
    ) -> None:
        """Append ``--<name> <value>`` using the first option name this COLMAP supports."""
        name = pick_option(self.options(command), candidates)
        if name is None:
            self.logger.debug(
                "COLMAP %s supports none of %s; skipped.", command, candidates
            )
            return
        args += [f"--{name}", str(value)]

    def run(self, command: str, args: list[str]) -> None:
        run_tool(
            [self.exe, command, *args], step=NAME, logger=self.logger, env=self.env
        )


def run(ctx: StepContext) -> StepResult:
    cfg = ctx.config.poses
    colmap = Colmap(find_tool("colmap"), ctx.logger)
    images_dir = ctx.manifest.step_output("extract_frames", "images_dir")
    colmap_dir = ctx.run_dir / "colmap"
    database = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"

    # Start clean: a leftover database or model from an earlier attempt would be reused.
    for leftover in colmap_dir.glob("database.db*"):
        leftover.unlink()
    shutil.rmtree(sparse_dir, ignore_errors=True)
    sparse_dir.mkdir(parents=True)

    num_images = len(list(images_dir.glob("*.jpg")))
    use_gpu = 1 if cfg.use_gpu else 0

    ctx.logger.info("COLMAP 1/3: detecting features in %d frames...", num_images)
    args: list[str] = [
        "--database_path",
        str(database),
        "--image_path",
        str(images_dir),
        "--ImageReader.camera_model",
        cfg.camera_model,
        "--ImageReader.single_camera",
        "1",
    ]
    command = "feature_extractor"
    colmap.add_option(
        args, command, ["FeatureExtraction.use_gpu", "SiftExtraction.use_gpu"], use_gpu
    )
    colmap.add_option(
        args,
        command,
        ["FeatureExtraction.num_threads", "SiftExtraction.num_threads"],
        cfg.num_threads,
    )
    colmap.add_option(
        args, command, ["SiftExtraction.max_num_features"], cfg.max_num_features
    )
    colmap.run(command, args)

    ctx.logger.info("COLMAP 2/3: matching features (%s)...", cfg.matcher)
    command = f"{cfg.matcher}_matcher"
    args = ["--database_path", str(database)]
    colmap.add_option(
        args, command, ["FeatureMatching.use_gpu", "SiftMatching.use_gpu"], use_gpu
    )
    colmap.add_option(
        args,
        command,
        ["FeatureMatching.num_threads", "SiftMatching.num_threads"],
        cfg.num_threads,
    )
    if cfg.matcher == "sequential":
        colmap.add_option(
            args, command, ["SequentialMatching.overlap"], cfg.sequential_overlap
        )
    colmap.run(command, args)

    ctx.logger.info("COLMAP 3/3: reconstructing camera poses (can take a while)...")
    command = "mapper"
    args = [
        "--database_path",
        str(database),
        "--image_path",
        str(images_dir),
        "--output_path",
        str(sparse_dir),
    ]
    colmap.add_option(args, command, ["Mapper.num_threads"], cfg.num_threads)
    colmap.add_option(
        args,
        command,
        ["Mapper.random_seed", "default_random_seed", "random_seed"],
        cfg.random_seed,
    )
    colmap.run(command, args)

    models = promote_best_model(sparse_dir)
    versions = {"colmap": colmap_version(colmap.exe)}
    if not models:
        raise StepError(
            NAME,
            f"COLMAP could not reconstruct the camera positions from any frames. {CAPTURE_TIPS}",
            info={
                "num_images": num_images,
                "num_registered": 0,
                "registered_ratio": 0.0,
            },
        )

    best_dir, registered = models[0]
    ratio = registered / num_images if num_images else 0.0
    info = {
        "num_images": num_images,
        "num_registered": registered,
        "registered_ratio": round(ratio, 4),
        "num_points3d": read_entry_count(best_dir / "points3D.bin"),
        "num_models": len(models),
        "other_model_sizes": [count for _, count in models[1:]],
        "matcher": cfg.matcher,
        "camera_model": cfg.camera_model,
    }
    ctx.logger.info(
        "COLMAP registered %d of %d frames (%.0f%%), %d 3D points, %d model(s).",
        registered,
        num_images,
        ratio * 100,
        info["num_points3d"],
        len(models),
    )
    if ratio < cfg.min_registered_ratio:
        raise StepError(
            NAME,
            f"COLMAP placed only {registered} of {num_images} frames ({ratio:.0%}); at least "
            f"{cfg.min_registered_ratio:.0%} is needed for a good 3D scene. {CAPTURE_TIPS}",
            info=info,
        )
    return StepResult(
        outputs={"sparse_model": best_dir, "database": database},
        info=info,
        versions=versions,
    )
