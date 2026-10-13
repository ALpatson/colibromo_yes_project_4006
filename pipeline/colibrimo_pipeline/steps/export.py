"""Step 5: publish the trained splat as ``scene.ply``, its camera poses as
``cameras.json``, and record the scene transform.

The transform is stored in the manifest because future "after" (renovated)
variants of the room must share the same coordinate frame (PRD section 7).
``cameras.json`` (D-015) gives the viewer every frame's camera pose in
scene.ply's coordinates, to level the room, set the eye height and limit the
walk area.
"""

from __future__ import annotations

import json
import shutil

import numpy as np

from colibrimo_pipeline.steps.base import StepContext, StepResult
from colibrimo_pipeline.utils.cameras import build_cameras
from colibrimo_pipeline.utils.ply import read_vertex_count

NAME = "export"
CAMERAS_FILE = "cameras.json"

TRANSFORM_DESCRIPTION = (
    "4x4 matrix (row-major) mapping COLMAP world coordinates to the coordinates "
    "of scene.ply: x_scene = matrix_4x4 @ [x, y, z, 1]. Applied by the trainer: "
    "gsplat's normalize_world_space (recentre, rotate, rescale), or the identity "
    "for Brush. Needed to align future 'after' variants with this 'before' scene."
)


def _frames(ctx: StepContext) -> list[dict] | None:
    """frames.json from extract_frames (None for run folders made before it existed)."""
    path = ctx.run_dir / "frames.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["frames"]


def run(ctx: StepContext) -> StepResult:
    trained_ply = ctx.manifest.step_output("train", "ply")
    transform = json.loads(
        ctx.manifest.step_output("train", "transform").read_text(encoding="utf-8")
    )

    scene = ctx.run_dir / "scene.ply"
    shutil.copy2(trained_ply, scene)  # copy (not move) so the train step stays complete

    cameras = build_cameras(
        ctx.manifest.step_output("poses", "sparse_model"),
        np.asarray(transform["matrix_4x4"]),
        _frames(ctx),
    )
    cameras_path = ctx.run_dir / CAMERAS_FILE
    cameras_path.write_text(json.dumps(cameras, indent=1), encoding="utf-8")

    ctx.manifest.set_run_info(
        transform={"description": TRANSFORM_DESCRIPTION, **transform},
        outputs={
            "scene_ply": "scene.ply",
            "cameras_json": CAMERAS_FILE,
            "manifest": "manifest.json",
        },
    )
    size = scene.stat().st_size
    num_gaussians = read_vertex_count(scene)
    ctx.logger.info(
        "Wrote %s (%.1f MB, %d Gaussians) and %s (%d cameras: %s).",
        scene.name,
        size / 1e6,
        num_gaussians,
        cameras_path.name,
        cameras["num_cameras"],
        ", ".join(f"{v} {n}" for v, n in cameras["videos"].items()),
    )
    return StepResult(
        outputs={"scene_ply": scene, "cameras_json": cameras_path},
        info={
            "num_gaussians": num_gaussians,
            "file_size_bytes": size,
            "num_cameras": cameras["num_cameras"],
            "cameras_per_video": cameras["videos"],
        },
    )
