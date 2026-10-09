"""Read just enough of a COLMAP binary model to judge a reconstruction.

COLMAP's ``images.bin`` and ``points3D.bin`` both start with a little-endian
uint64 holding the number of entries, which is all we need here.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path


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


def promote_best_model(sparse_dir: Path) -> list[tuple[Path, int]]:
    """Make the largest model ``sparse/0`` (what the trainer reads) and rename the others.

    COLMAP's mapper may split a capture into several models. Returns the models
    after renaming, largest first (empty list if there are none).
    """
    models = find_models(sparse_dir)
    if not models or models[0][0].name == "0":
        return models

    # Two passes so names never collide: first to temporary names, then final ones.
    temp = []
    for i, (path, count) in enumerate(models):
        tmp_path = path.with_name(f"_tmp_{i}")
        path.rename(tmp_path)
        temp.append((tmp_path, count))
    result = []
    for i, (tmp_path, count) in enumerate(temp):
        final = tmp_path.with_name("0" if i == 0 else f"unused_{i}")
        tmp_path.rename(final)
        result.append((final, count))
    return result


def parse_help_options(help_text: str) -> set[str]:
    """Option names (without ``--``) listed in a ``colmap <command> --help`` output."""
    return set(re.findall(r"--([A-Za-z0-9_.]+)", help_text))


def pick_option(available: set[str], candidates: list[str]) -> str | None:
    """First candidate option name supported by this COLMAP version (names differ 3.x vs 4.x)."""
    for name in candidates:
        if name in available:
            return name
    return None
