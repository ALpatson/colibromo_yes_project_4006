"""Step 5: publish the trained splat as ``scene.ply`` and record the scene transform.

The transform is stored in the manifest because future "after" (renovated)
variants of the room must share the same coordinate frame (PRD section 7).
"""

from __future__ import annotations

import json
import shutil

from colibrimo_pipeline.steps.base import StepContext, StepResult
from colibrimo_pipeline.utils.ply import read_vertex_count

NAME = "export"

TRANSFORM_DESCRIPTION = (
    "4x4 matrix (row-major) mapping COLMAP world coordinates to the coordinates "
    "of scene.ply: x_scene = matrix_4x4 @ [x, y, z, 1]. Applied by gsplat's "
    "normalize_world_space (recentre, rotate, rescale). Needed to align future "
    "'after' variants with this 'before' scene."
)


def run(ctx: StepContext) -> StepResult:
    trained_ply = ctx.manifest.step_output("train", "ply")
    transform = json.loads(
        ctx.manifest.step_output("train", "transform").read_text(encoding="utf-8")
    )

    scene = ctx.run_dir / "scene.ply"
    shutil.copy2(trained_ply, scene)  # copy (not move) so the train step stays complete

    ctx.manifest.set_run_info(
        transform={"description": TRANSFORM_DESCRIPTION, **transform},
        outputs={"scene_ply": "scene.ply", "manifest": "manifest.json"},
    )
    size = scene.stat().st_size
    num_gaussians = read_vertex_count(scene)
    ctx.logger.info(
        "Wrote %s (%.1f MB, %d Gaussians).", scene.name, size / 1e6, num_gaussians
    )
    return StepResult(
        outputs={"scene_ply": scene},
        info={"num_gaussians": num_gaussians, "file_size_bytes": size},
    )
