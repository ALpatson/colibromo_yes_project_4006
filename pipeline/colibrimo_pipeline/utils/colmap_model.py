"""Read what we need from COLMAP binary models, and keep one model per run.

COLMAP's ``images.bin`` and ``points3D.bin`` both start with a little-endian
uint64 holding the number of entries. Format reference:
https://colmap.github.io/format.html#binary-file-format
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np


def read_entry_count(path: Path) -> int:
    """Number of entries in a COLMAP ``images.bin`` / ``points3D.bin`` / ``cameras.bin``."""
    with open(path, "rb") as f:
        header = f.read(8)
    if len(header) != 8:
        raise ValueError(f"{path} is too short to be a COLMAP binary file.")
    return struct.unpack("<Q", header)[0]


def find_models(sparse_dir: Path) -> list[tuple[Path, int]]:
    """All models under ``sparse_dir`` as (folder, registered image count), largest first."""
    models = []
    if sparse_dir.is_dir():
        for child in sorted(sparse_dir.iterdir()):
            images_bin = child / "images.bin"
            if child.is_dir() and images_bin.is_file():
                models.append((child, read_entry_count(images_bin)))
    return sorted(models, key=lambda m: m[1], reverse=True)


# Leftover models go to <run>/colmap_unused/, NEXT TO colmap/, not inside it:
# Brush scans the whole colmap/ folder and mixed files from leftover models
# (D-018). Older layouts: colmap/sparse/unused_N (D-010), colmap/unused_models/ (D-014).
UNUSED_MODELS_DIR = "colmap_unused"
OLD_UNUSED_MODELS_DIR = "unused_models"


def unused_models_dir(sparse_dir: Path) -> Path:
    """``<run>/colmap_unused`` for ``<run>/colmap/sparse``."""
    return sparse_dir.parent.parent / UNUSED_MODELS_DIR


def find_model_dirs(colmap_dir: Path) -> list[Path]:
    """Every COLMAP model (folder with an images.bin) anywhere under ``colmap_dir``."""
    return sorted(p.parent for p in Path(colmap_dir).rglob("images.bin"))


def _move_unique(source: Path, target_dir: Path, name: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / name
    n = 1
    while target.exists():
        target = target_dir / f"{name}_{n}"
        n += 1
    source.rename(target)
    return target


def promote_best_model(sparse_dir: Path) -> list[tuple[Path, int]]:
    """Keep only the largest model, as ``sparse/0``; move the others out of ``colmap/``.

    COLMAP's mapper may split a capture into several models. The others go to
    ``<run>/colmap_unused/model_N`` (kept for debugging): trainers must only
    see one model (Brush mixed files of leftover models, D-018).
    Returns the models after moving, largest first (empty list if there are none).
    """
    models = find_models(sparse_dir)
    if not models:
        return models

    # Move to temporary names first so "0" is free for the best model.
    temp = []
    for i, (path, count) in enumerate(models):
        tmp_path = path.with_name(f"_tmp_{i}")
        path.rename(tmp_path)
        temp.append((tmp_path, count))
    best_tmp, best_count = temp[0]
    best = best_tmp.with_name("0")
    best_tmp.rename(best)
    result = [(best, best_count)]

    unused_dir = unused_models_dir(sparse_dir)
    for i, (tmp_path, count) in enumerate(temp[1:], start=1):
        result.append((_move_unique(tmp_path, unused_dir, f"model_{i}"), count))
    return result


def move_unused_models(sparse_dir: Path) -> list[str]:
    """Move every leftover model out of ``colmap/`` into ``<run>/colmap_unused/``.

    Leftovers are any folder in ``sparse/`` other than ``0`` (layout before
    D-014) and anything in ``colmap/unused_models/`` (layout of D-014, still
    inside colmap/). For run folders made before D-018. Idempotent, never
    overwrites. Returns what was moved, e.g. ``["unused_models/model_1"]``.
    """
    moved: list[str] = []
    colmap_dir = sparse_dir.parent
    target_dir = unused_models_dir(sparse_dir)
    if sparse_dir.is_dir():
        for child in sorted(sparse_dir.iterdir()):
            if child.is_dir() and child.name != "0":
                _move_unique(child, target_dir, child.name)
                moved.append(f"sparse/{child.name}")
    old_dir = colmap_dir / OLD_UNUSED_MODELS_DIR
    if old_dir.is_dir():
        for child in sorted(old_dir.iterdir()):
            _move_unique(child, target_dir, child.name)
            moved.append(f"{OLD_UNUSED_MODELS_DIR}/{child.name}")
        old_dir.rmdir()
    return moved


# COLMAP camera model id -> (name, number of parameters). From COLMAP's
# src/colmap/sensor/models.h (same ids in COLMAP 3.x and 4.x).
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def read_cameras(path: Path) -> dict[int, dict]:
    """Intrinsics from a COLMAP ``cameras.bin``: camera_id -> model, size, params.

    Per camera: camera_id (int32), model_id (int32), width (uint64),
    height (uint64), then the model's parameters (float64).
    """
    data = Path(path).read_bytes()
    (count,) = struct.unpack_from("<Q", data, 0)
    offset, cameras = 8, {}
    for _ in range(count):
        camera_id, model_id, width, height = struct.unpack_from("<iiQQ", data, offset)
        offset += 24
        if model_id not in CAMERA_MODELS:
            raise ValueError(f"{path}: unknown COLMAP camera model id {model_id}.")
        model, num_params = CAMERA_MODELS[model_id]
        params = struct.unpack_from(f"<{num_params}d", data, offset)
        offset += 8 * num_params
        cameras[camera_id] = {
            "model": model,
            "width": width,
            "height": height,
            "params": list(params),
        }
    return cameras


def read_images(path: Path) -> list[dict]:
    """Registered images from a COLMAP ``images.bin``.

    Per image: image_id (uint32), qvec (w, x, y, z) and tvec (float64) of the
    camera-from-world transform, camera_id (uint32), name (null-terminated),
    then its 2D points (count uint64, each x, y float64 + point3D_id int64),
    which are skipped.
    """
    images = []
    for record in _image_records(Path(path).read_bytes()):
        images.append(
            {k: record[k] for k in ("image_id", "qvec", "tvec", "camera_id", "name")}
        )
    return images


def _image_records(data: bytes):
    """Each image of an ``images.bin`` with the byte range of its record
    (``start``/``end``) and of its 2D points (``points_at``, ``num_points2d``)."""
    (count,) = struct.unpack_from("<Q", data, 0)
    offset = 8
    for _ in range(count):
        start = offset
        image_id, *pose = struct.unpack_from("<I7d", data, offset)
        offset += 60
        (camera_id,) = struct.unpack_from("<I", data, offset)
        offset += 4
        end = data.index(b"\0", offset)
        name = data[offset:end].decode("utf-8")
        offset = end + 1
        (num_points2d,) = struct.unpack_from("<Q", data, offset)
        points_at = offset + 8
        offset = points_at + 24 * num_points2d
        yield {
            "image_id": image_id,
            "qvec": pose[:4],
            "tvec": pose[4:],
            "camera_id": camera_id,
            "name": name,
            "start": start,
            "end": offset,
            "points_at": points_at,
            "num_points2d": num_points2d,
        }


def write_images_subset(src: Path, dst: Path, keep_names: set[str]) -> int:
    """Copy ``images.bin`` keeping only the images named in ``keep_names``
    (records copied byte for byte); returns how many were kept."""
    data = Path(src).read_bytes()
    kept = [r for r in _image_records(data) if r["name"] in keep_names]
    out = struct.pack("<Q", len(kept)) + b"".join(
        data[r["start"] : r["end"]] for r in kept
    )
    Path(dst).write_bytes(out)
    return len(kept)


# Leading parameters in pixels (focal lengths, principal point) per camera model;
# the rest (distortion) has no unit.
_PIXEL_PARAMS = {
    "SIMPLE_PINHOLE": 3,
    "SIMPLE_RADIAL": 3,
    "RADIAL": 3,
    "PINHOLE": 4,
    "OPENCV": 4,
}


def write_scaled_model(src: Path, dst: Path, scale: float) -> None:
    """Copy a COLMAP model (``sparse/0``) for images ``scale`` times larger.

    Camera poses and 3D points are unchanged; image sizes, focal lengths,
    principal points and the 2D feature positions are multiplied by ``scale``
    (COLMAP's pixel coordinates start at the image corner, so this is exact).
    """
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    out = struct.pack("<Q", len(cameras := read_cameras(src / "cameras.bin")))
    ids = {name: i for i, (name, _) in CAMERA_MODELS.items()}
    for camera_id, cam in cameras.items():
        if cam["model"] not in _PIXEL_PARAMS:
            raise ValueError(f"Cannot rescale camera model {cam['model']}.")
        n = _PIXEL_PARAMS[cam["model"]]
        params = [p * scale for p in cam["params"][:n]] + cam["params"][n:]
        out += struct.pack(
            "<iiQQ",
            camera_id,
            ids[cam["model"]],
            round(cam["width"] * scale),
            round(cam["height"] * scale),
        )
        out += struct.pack(f"<{len(params)}d", *params)
    (dst / "cameras.bin").write_bytes(out)

    data = bytearray((src / "images.bin").read_bytes())
    for record in _image_records(bytes(data)):
        for i in range(record["num_points2d"]):
            at = record["points_at"] + 24 * i
            x, y = struct.unpack_from("<dd", data, at)
            struct.pack_into("<dd", data, at, x * scale, y * scale)
    (dst / "images.bin").write_bytes(bytes(data))
    for other in src.iterdir():  # points3D.bin, frames.bin, rigs.bin, ...
        if other.is_file() and other.name not in ("cameras.bin", "images.bin"):
            (dst / other.name).write_bytes(other.read_bytes())


def qvec_to_rotation(qvec) -> np.ndarray:
    """3x3 rotation matrix from a COLMAP quaternion (w, x, y, z)."""
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotation_to_qvec(r: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z), w >= 0, from a 3x3 rotation matrix."""
    # Robust conversion via the largest diagonal term (Shepperd's method).
    m = r
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = [
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        ]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        ]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        ]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [
            (m[1, 0] - m[0, 1]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
        ]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return -q if q[0] < 0 else q


def read_points3d_xyz(path: Path) -> np.ndarray:
    """Positions (N x 3, float64) of the 3D points in a COLMAP ``points3D.bin``.

    Per point: id (uint64), xyz (3 x float64), rgb (3 x uint8), error
    (float64), track length (uint64), then (image_id, point2D_idx) int32 pairs.
    """
    data = Path(path).read_bytes()
    (count,) = struct.unpack_from("<Q", data, 0)
    xyz = np.empty((count, 3))
    offset = 8
    for i in range(count):
        xyz[i] = struct.unpack_from("<3d", data, offset + 8)
        (track_length,) = struct.unpack_from("<Q", data, offset + 43)
        offset += 51 + 8 * track_length
    return xyz


def parse_help_options(help_text: str) -> set[str]:
    """Option names (without ``--``) listed in a ``colmap <command> --help`` output."""
    return set(re.findall(r"--([A-Za-z0-9_.]+)", help_text))


def pick_option(available: set[str], candidates: list[str]) -> str | None:
    """First candidate option name supported by this COLMAP version (names differ 3.x vs 4.x)."""
    for name in candidates:
        if name in available:
            return name
    return None
