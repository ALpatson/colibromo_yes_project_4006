"""Input videos of one room: collect them and give each a short, unique name.

A room can be filmed as several videos (e.g. ``bottom.mp4``, ``middle.mp4``,
``up.mp4`` at three heights). Each video's name is its file name without the
extension, reduced to letters, digits, ``-`` and ``_``; frames are named after
it (``bottom_00000.jpg``), so names must be unique.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from colibrimo_pipeline.errors import ConfigError

# Extensions picked up by ``--videos <folder>`` (case-insensitive).
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}


@dataclass(frozen=True)
class VideoInput:
    """One input video: ``name`` is used for frame names and in the manifest."""

    name: str
    path: Path


def video_name(path: Path) -> str:
    """'Living Room (2).MOV' -> 'Living_Room_2'."""
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(path).stem).strip("_")
    return name or "video"


def videos_in_folder(folder: Path) -> list[Path]:
    """Video files directly inside ``folder``, sorted by name."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ConfigError(f"Video folder not found: {folder}")
    found = sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not found:
        raise ConfigError(
            f"No video files ({', '.join(sorted(VIDEO_EXTENSIONS))}) in {folder}"
        )
    return found


def collect_videos(
    videos: str | Path | Sequence[str | Path] | None = None,
    folder: str | Path | None = None,
) -> list[VideoInput]:
    """Videos from explicit paths and/or a folder, named and checked for duplicates.

    Paths are resolved but not checked for existence here: the ``validate``
    step reports a missing file with a clear message.
    """
    paths: list[Path] = []
    if isinstance(videos, str | Path):
        paths.append(Path(videos))
    elif videos:
        paths.extend(Path(v) for v in videos)
    if folder is not None:
        paths.extend(videos_in_folder(Path(folder)))
    if not paths:
        raise ConfigError("No input video: give --video (repeatable) or --videos.")
    return name_videos(Path(p).resolve() for p in paths)


def name_videos(paths: Iterable[Path]) -> list[VideoInput]:
    """Pair each path with its name; reject the same file twice or clashing names."""
    result: list[VideoInput] = []
    seen_paths: set[Path] = set()
    seen_names: dict[str, Path] = {}
    for path in paths:
        if path in seen_paths:
            continue  # same file given twice (e.g. --video and --videos)
        seen_paths.add(path)
        name = video_name(path)
        if name.lower() in seen_names:
            raise ConfigError(
                f"Two videos would get the same name '{name}': "
                f"{seen_names[name.lower()].name} and {path.name}. Rename one of them."
            )
        seen_names[name.lower()] = path
        result.append(VideoInput(name, path))
    return result
