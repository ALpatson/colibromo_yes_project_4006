"""``cameras.json``: where every frame was filmed from, in scene.ply's coordinates.

The viewer uses it to level the room (cameras were held upright), set the eye
height (middle_* cameras) and limit the walk area (where the cameras were).

Each camera has its centre (``position``) and orientation. Camera axes follow
COLMAP/OpenCV: x right, y down, z forward (viewing direction). ``rotation_wxyz``
is the world-from-camera rotation as a quaternion; ``up`` (= -y) and
``forward`` (= +z) are the same axes as unit vectors in scene coordinates.

COLMAP poses are camera-from-world: x_cam = R x_world + t, so the centre is
C = -R^T t and the camera axes in world coordinates are the rows of R. A scene
transform M = [s Q | m] (gsplat's normalisation: uniform scale s, rotation Q,
translation m) maps them to C' = s Q C + m and axes' = Q axes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from colibrimo_pipeline.utils.colmap_model import (
    qvec_to_rotation,
    read_cameras,
    read_images,
    rotation_to_qvec,
)

SCHEMA_VERSION = 1
CONVENTION = (
    "Coordinates are those of scene.ply. position: camera centre. rotation_wxyz: "
    "world-from-camera rotation (w, x, y, z). Camera axes (COLMAP/OpenCV): x right, "
    "y down, z forward; 'up' = -y and 'forward' = +z as unit vectors in scene "
    "coordinates. intrinsics: COLMAP camera model and parameters in pixels of the "
    "frames in colmap/images."
)


def split_similarity(matrix: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Split a 4x4 similarity transform into (scale, rotation 3x3, translation)."""
    linear = matrix[:3, :3]
    scale = float(np.cbrt(np.linalg.det(linear)))
    if scale <= 0:
        raise ValueError("Scene transform must be a similarity with positive scale.")
    rotation = linear / scale
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4):
        raise ValueError("Scene transform is not a rotation + uniform scale.")
    return scale, rotation, matrix[:3, 3]


def scene_camtoworld(camtoworld: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """A COLMAP camera-to-world 4x4 moved into scene.ply's coordinates.

    Rotation becomes Q R, the centre s Q C + m: the camera sees the moved scene
    exactly as it saw the original (only distances are scaled by s).
    """
    matrix = np.asarray(matrix, dtype=float)
    _, rotation, _ = split_similarity(matrix)
    moved = np.eye(4)
    moved[:3, :3] = rotation @ camtoworld[:3, :3]
    moved[:3, 3] = matrix[:3, :3] @ camtoworld[:3, 3] + matrix[:3, 3]
    return moved


def _round(values, digits: int = 6) -> list[float]:
    return [round(float(v), digits) for v in values]


def build_cameras(
    sparse_model: Path,
    matrix: np.ndarray,
    frames: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Camera poses of a COLMAP model (``sparse/0``) in scene coordinates.

    ``matrix``: the 4x4 transform from COLMAP to scene.ply coordinates.
    ``frames``: entries of frames.json (file, video, index, time_s); if missing
    (old run folders), the video is taken from the frame name ``<video>_NNNNN.jpg``.
    """
    scale, rot, shift = split_similarity(np.asarray(matrix, dtype=float))
    by_file = {f["file"]: f for f in frames or []}

    cameras = []
    for image in sorted(
        read_images(sparse_model / "images.bin"), key=lambda i: i["name"]
    ):
        r = qvec_to_rotation(image["qvec"])  # camera-from-world
        centre = -r.T @ np.asarray(image["tvec"])
        world_from_cam = rot @ r.T
        frame = by_file.get(image["name"], {})
        cameras.append(
            {
                "name": image["name"],
                "video": frame.get("video") or image["name"].rsplit("_", 1)[0],
                "frame_index": frame.get("index"),
                "time_s": frame.get("time_s"),
                "position": _round(scale * rot @ centre + shift),
                "rotation_wxyz": _round(rotation_to_qvec(world_from_cam), 8),
                "up": _round(-world_from_cam[:, 1]),
                "forward": _round(world_from_cam[:, 2]),
                "camera_id": image["camera_id"],
            }
        )

    intrinsics = {
        str(cid): cam for cid, cam in read_cameras(sparse_model / "cameras.bin").items()
    }
    videos: dict[str, int] = {}
    for cam in cameras:
        videos[cam["video"]] = videos.get(cam["video"], 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "convention": CONVENTION,
        "transform_4x4": [_round(row, 9) for row in np.asarray(matrix, dtype=float)],
        "num_cameras": len(cameras),
        "videos": videos,
        "intrinsics": intrinsics,
        "cameras": cameras,
    }
