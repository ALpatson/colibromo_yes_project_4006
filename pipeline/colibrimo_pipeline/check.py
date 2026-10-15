"""Check a finished run: are scene.ply and cameras.json in the same coordinates?

Used by ``cpipe check`` (e.g. at the end of the Colab notebook) and when
``cpipe ui`` imports a result trained elsewhere. Needs the run folder's
``manifest.json``, ``scene.ply``, ``cameras.json`` and COLMAP model
(``colmap/sparse/0``).

- Coordinate check: COLMAP's 3D points, moved by the run's scene transform
  (identity for Brush, gsplat's normalisation for gsplat), must lie right next
  to splats of scene.ply (same test as D-013).
- cameras.json: one camera per frame COLMAP placed, built with the same
  transform as the manifest records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from colibrimo_pipeline.errors import PipelineError
from colibrimo_pipeline.steps._brush import identity_check
from colibrimo_pipeline.utils.colmap_model import read_points3d_xyz
from colibrimo_pipeline.utils.ply import read_vertex_xyz


def transform_points(matrix: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to N x 3 points."""
    matrix = np.asarray(matrix, dtype=float)
    return xyz @ matrix[:3, :3].T + matrix[:3, 3]


def coordinate_check(
    colmap_xyz: np.ndarray, splat_xyz: np.ndarray, matrix: np.ndarray
) -> dict[str, Any]:
    """identity_check after moving COLMAP's points into scene.ply's coordinates."""
    return identity_check(transform_points(matrix, colmap_xyz), splat_xyz)


def check_run(run_dir: Path) -> dict[str, Any]:
    """Check a finished run folder; ``ok`` is True when every check passes."""
    run_dir = Path(run_dir)
    files = {
        name: run_dir / name for name in ("manifest.json", "scene.ply", "cameras.json")
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise PipelineError(f"{run_dir} has no {', '.join(missing)}.")
    manifest = json.loads(files["manifest.json"].read_text(encoding="utf-8"))
    cameras = json.loads(files["cameras.json"].read_text(encoding="utf-8"))
    matrix = (manifest.get("transform") or {}).get("matrix_4x4")
    if matrix is None:
        raise PipelineError("The manifest records no scene transform (export not run).")
    poses = manifest.get("steps", {}).get("poses", {})
    sparse = run_dir / poses.get("outputs", {}).get("sparse_model", "colmap/sparse/0")
    if not (sparse / "points3D.bin").is_file():
        raise PipelineError(f"No COLMAP model at {sparse}.")

    coords = coordinate_check(
        read_points3d_xyz(sparse / "points3D.bin"),
        read_vertex_xyz(files["scene.ply"]),
        np.asarray(matrix),
    )
    placed = poses.get("info", {}).get("num_registered")
    same_transform = bool(
        np.allclose(np.asarray(cameras.get("transform_4x4")), matrix, atol=1e-6)
    )
    return {
        "ok": coords["same_coordinates"]
        and same_transform
        and cameras.get("num_cameras") == placed,
        "coordinate_check": coords,
        "num_cameras": cameras.get("num_cameras"),
        "cameras_per_video": cameras.get("videos"),
        "frames_placed_by_colmap": placed,
        "cameras_use_manifest_transform": same_transform,
        "trainer": manifest.get("steps", {})
        .get("train", {})
        .get("info", {})
        .get("trainer"),
    }


def describe(result: dict[str, Any]) -> list[str]:
    """Plain-language lines for a check_run() result."""
    coords = result["coordinate_check"]
    videos = ", ".join(
        f"{v} {n}" for v, n in (result["cameras_per_video"] or {}).items()
    )
    ok = {True: "OK", False: "FAILED"}
    return [
        f"Coordinate check: median distance from a COLMAP point to the nearest splat = "
        f"{coords['median_nearest_splat_rel']:.4f} x scene size (pass if < "
        f"{coords['threshold_rel']}): {ok[coords['same_coordinates']]}",
        f"Cameras in cameras.json: {result['num_cameras']} ({videos}); frames placed by "
        f"COLMAP: {result['frames_placed_by_colmap']}: "
        f"{ok[result['num_cameras'] == result['frames_placed_by_colmap']]}",
        f"cameras.json uses the manifest's scene transform: "
        f"{ok[result['cameras_use_manifest_transform']]}",
        f"Overall: {'PASS' if result['ok'] else 'FAIL'} (trainer: {result['trainer']})",
    ]
