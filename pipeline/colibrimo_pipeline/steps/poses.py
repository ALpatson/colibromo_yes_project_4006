"""Step 3: estimate camera poses with COLMAP (features -> matching -> mapper).

The matcher is chosen by ``choose_matcher``: sequential for one video, and
sequential with loop detection for several videos (which plain sequential
matching cannot connect).

COLMAP is called directly rather than through Nerfstudio's ``ns-process-data``
(see docs/decisions.md, D-006). Option names changed between COLMAP 3.x and 4.x
(e.g. ``SiftExtraction.use_gpu`` -> ``FeatureExtraction.use_gpu``), so each
option is looked up in ``colmap <command> --help`` and the supported name is used.

Output layout (what gsplat's trainer expects)::

    colmap/images/        frames (from extract_frames)
    colmap/database.db    COLMAP features and matches
    colmap/sparse/0/      cameras.bin, images.bin, points3D.bin (largest model)
    colmap/unused_models/ other, smaller models COLMAP made (kept out of sparse/
                          so trainers only ever see one model)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from colibrimo_pipeline.config import PosesConfig
from colibrimo_pipeline.errors import StepError
from colibrimo_pipeline.steps.base import CAPTURE_TIPS, StepContext, StepResult
from colibrimo_pipeline.utils.colmap_model import (
    UNUSED_MODELS_DIR,
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

    def require_option(self, args: list[str], command: str, name: str, value) -> None:
        """Like add_option, but fail clearly if this COLMAP does not have the option."""
        if name not in self.options(command):
            raise StepError(
                NAME,
                f"This COLMAP version has no '--{name}' option for {command}, which the "
                "chosen matching settings need. Update COLMAP or set "
                "poses.matcher=exhaustive.",
            )
        args += [f"--{name}", str(value)]

    def run(self, command: str, args: list[str]) -> None:
        run_tool(
            [self.exe, command, *args], step=NAME, logger=self.logger, env=self.env
        )


@dataclass
class MatcherChoice:
    """Which COLMAP matcher to run, and why (recorded in the manifest)."""

    matcher: str  # "sequential" or "exhaustive"
    loop_detection: bool
    reason: str


def choose_matcher(cfg: PosesConfig, num_videos: int) -> MatcherChoice:
    """Resolve ``poses.matcher`` (``auto`` picks one from the number of videos).

    Sequential matching only compares each frame with its neighbours in time, so
    separate videos are never connected. With several videos, ``auto`` adds loop
    detection, which also matches similar-looking frames far apart (e.g. from
    another video). On myRoom2 (3 videos, 300 frames) it connected all videos in
    half the time of exhaustive matching (docs/decisions.md, D-011).
    """
    if cfg.matcher != "auto":
        return MatcherChoice(
            cfg.matcher,
            cfg.matcher == "sequential" and cfg.loop_detection,
            "set in config",
        )
    if num_videos <= 1:
        return MatcherChoice(
            "sequential", cfg.loop_detection, "one video: frames are in time order"
        )
    return MatcherChoice(
        "sequential",
        True,
        f"{num_videos} videos: loop detection connects the videos",
    )


def frames_by_video(run_dir: Path, images_dir: Path) -> dict[str, list[str]]:
    """Frame file names per video, in order (frames.json, else the frame name prefix)."""
    groups: dict[str, list[str]] = {}
    frames_file = run_dir / "frames.json"
    if frames_file.is_file():
        for frame in json.loads(frames_file.read_text(encoding="utf-8"))["frames"]:
            groups.setdefault(frame["video"], []).append(frame["file"])
    else:
        for image in sorted(images_dir.glob("*.jpg")):
            groups.setdefault(image.name.rsplit("_", 1)[0], []).append(image.name)
    return groups


def run(ctx: StepContext) -> StepResult:
    cfg = ctx.config.poses
    colmap = Colmap(find_tool("colmap"), ctx.logger)
    images_dir = ctx.manifest.step_output("extract_frames", "images_dir")
    validate_info = ctx.manifest.step_info("validate")
    num_videos = int(validate_info.get("num_videos", 1))
    single_camera = bool(validate_info.get("same_camera", True))
    colmap_dir = ctx.run_dir / "colmap"
    database = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"

    # Start clean: a leftover database or model from an earlier attempt would be reused.
    for leftover in colmap_dir.glob("database.db*"):
        leftover.unlink()
    shutil.rmtree(sparse_dir, ignore_errors=True)
    shutil.rmtree(colmap_dir / UNUSED_MODELS_DIR, ignore_errors=True)
    sparse_dir.mkdir(parents=True)

    num_images = len(list(images_dir.glob("*.jpg")))
    use_gpu = 1 if cfg.use_gpu else 0

    # Camera models (intrinsics: focal length, distortion). All videos with the
    # same resolution/orientation: one shared camera (same phone settings).
    # Otherwise one camera per video, by extracting each video's frames in a
    # separate run with its own image list (D-017). Never one camera per frame:
    # 300 unknown focal lengths made NewRoom fall apart.
    groups = (
        {"all": None} if single_camera else frames_by_video(ctx.run_dir, images_dir)
    )
    ctx.logger.info(
        "COLMAP 1/3: detecting features in %d frames (%s)...",
        num_images,
        (
            "one shared camera"
            if single_camera
            else f"one camera per video: {len(groups)}"
        ),
    )
    command = "feature_extractor"
    lists_dir = colmap_dir / "image_lists"
    shutil.rmtree(lists_dir, ignore_errors=True)
    for video, frames in groups.items():
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
        if frames is not None:
            lists_dir.mkdir(exist_ok=True)
            image_list = lists_dir / f"{video}.txt"
            image_list.write_text("\n".join(frames) + "\n", encoding="utf-8")
            colmap.require_option(args, command, "image_list_path", image_list)
        colmap.add_option(
            args,
            command,
            ["FeatureExtraction.use_gpu", "SiftExtraction.use_gpu"],
            use_gpu,
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

    choice = choose_matcher(cfg, num_videos)
    ctx.logger.info(
        "COLMAP 2/3: matching features (%s%s; %s)...",
        choice.matcher,
        " + loop detection" if choice.loop_detection else "",
        choice.reason,
    )
    command = f"{choice.matcher}_matcher"
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
    if choice.matcher == "sequential":
        colmap.add_option(
            args, command, ["SequentialMatching.overlap"], cfg.sequential_overlap
        )
        if choice.loop_detection:
            colmap.require_option(args, command, "SequentialMatching.loop_detection", 1)
            if cfg.vocab_tree_path:
                colmap.require_option(
                    args,
                    command,
                    "SequentialMatching.vocab_tree_path",
                    cfg.vocab_tree_path,
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
        "matcher": choice.matcher,
        "loop_detection": choice.loop_detection,
        "matcher_reason": choice.reason,
        "num_videos": num_videos,
        "single_camera": single_camera,
        "num_cameras": 1 if single_camera else len(groups),
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
