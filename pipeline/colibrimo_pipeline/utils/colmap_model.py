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


UNUSED_MODELS_DIR = "unused_models"


def promote_best_model(sparse_dir: Path) -> list[tuple[Path, int]]:
    """Keep only the largest model, as ``sparse/0``; move the others out of ``sparse/``.

    COLMAP's mapper may split a capture into several models. The others go to
    ``<colmap>/unused_models/model_N`` (kept for debugging): trainers must only
    see one model (Brush loaded a tiny leftover model instead of ``0``).
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

    unused_dir = sparse_dir.parent / UNUSED_MODELS_DIR
    for i, (tmp_path, count) in enumerate(temp[1:], start=1):
        unused_dir.mkdir(exist_ok=True)
        final = unused_dir / f"model_{i}"
        tmp_path.rename(final)
        result.append((final, count))
    return result


def move_unused_models(sparse_dir: Path) -> list[str]:
    """Move every folder in ``sparse/`` except ``0`` to ``unused_models/``.

    For run folders made before ``promote_best_model`` moved them itself
    (they had ``sparse/unused_N``). Idempotent. Returns the moved folder names.
    """
    moved = []
    if not sparse_dir.is_dir():
        return moved
    unused_dir = sparse_dir.parent / UNUSED_MODELS_DIR
    for child in sorted(sparse_dir.iterdir()):
        if child.is_dir() and child.name != "0":
            unused_dir.mkdir(exist_ok=True)
            target = unused_dir / child.name
            n = 1
            while target.exists():
                target = unused_dir / f"{child.name}_{n}"
                n += 1
            child.rename(target)
            moved.append(child.name)
    return moved


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
