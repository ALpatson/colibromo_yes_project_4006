"""What ``cpipe ui`` does, without the HTTP layer (so it can be tested on its own).

- uploads: room/file name checks and streaming a video to ``data/uploads/<room>/``;
- progress: read from the run folder only (``manifest.json`` + ``logs/train.log``);
- runs: one ``run_pipeline()`` at a time in a background thread, resumable;
- viewer: copy ``scene.ply`` + ``cameras.json`` into the viewer's ``public/runs/<room>/``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

from colibrimo_pipeline.config import deep_merge, load_config
from colibrimo_pipeline.errors import PipelineError, StepError, ToolNotFoundError
from colibrimo_pipeline.pipeline import STEP_NAMES, run_pipeline
from colibrimo_pipeline.steps._brush import parse_progress
from colibrimo_pipeline.utils.proc import find_tool, run_capture
from colibrimo_pipeline.utils.videos import VIDEO_EXTENSIONS, video_name

PRESETS = ["fast", "balanced", "quality"]
VIEWER_URL = "http://localhost:5173/"

# Settings for the UI's runs: Brush on the laptop, sized for a 4 GB GPU (the
# values that worked on the team's NVIDIA T500). `cpipe ui --set ...` overrides.
UI_OVERRIDES: dict[str, Any] = {
    "train": {"trainer": "brush", "max_splats": 1_000_000, "max_resolution": 1280}
}

_ROOM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class UserError(PipelineError):
    """Something the person using the page should fix (shown as is)."""


class Busy(UserError):
    """A run is already going."""


# ---- names and uploads --------------------------------------------------------------


def clean_room_name(name: str) -> str:
    """'Living room 2' -> 'Living_room_2'; rejects empty or odd names."""
    cleaned = re.sub(r"\s+", "_", (name or "").strip())
    if not _ROOM.match(cleaned):
        raise UserError(
            "Room name: 1-64 characters, letters, digits, '-' or '_' "
            "(spaces become '_'), starting with a letter or digit."
        )
    return cleaned


def clean_video_filename(name: str) -> str:
    """Safe file name for an uploaded video: no folders, a video extension."""
    base = Path((name or "").replace("\\", "/")).name
    suffix = Path(base).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise UserError(
            f"'{base}' is not a video ({', '.join(sorted(VIDEO_EXTENSIONS))})."
        )
    return video_name(Path(base)) + suffix


def save_upload(
    uploads_dir: Path,
    room: str,
    filename: str,
    stream: BinaryIO,
    length: int,
    chunk_size: int = 1 << 20,
) -> Path:
    """Stream ``length`` bytes to ``uploads/<room>/<filename>``.

    Written to a ``.part`` file first and renamed at the end, so an interrupted
    upload is never taken for a video.
    """
    target_dir = Path(uploads_dir) / clean_room_name(room)
    target = target_dir / clean_video_filename(filename)
    target_dir.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    remaining = length
    with open(partial, "wb") as out:
        while remaining > 0:
            data = stream.read(min(chunk_size, remaining))
            if not data:
                break
            out.write(data)
            remaining -= len(data)
    if remaining:
        partial.unlink(missing_ok=True)
        raise UserError(f"Upload of {target.name} was interrupted; try again.")
    partial.replace(target)
    return target


def list_videos(uploads_dir: Path, room: str) -> list[Path]:
    """Uploaded videos of a room, sorted by name."""
    folder = Path(uploads_dir) / room
    if not folder.is_dir():
        return []
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def list_rooms(data_dir: Path) -> list[str]:
    """Rooms that have uploads or a run folder."""
    names: set[str] = set()
    for sub in ("uploads", "runs"):
        folder = Path(data_dir) / sub
        if folder.is_dir():
            names.update(p.name for p in folder.iterdir() if p.is_dir())
    return sorted(n for n in names if _ROOM.match(n))


# ---- progress ------------------------------------------------------------------------


def _tail(path: Path, size: int = 65536) -> str:
    """Last ``size`` bytes of a (possibly growing) text file."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - size))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def read_manifest(run_dir: Path) -> dict[str, Any] | None:
    path = Path(run_dir) / "manifest.json"
    for _ in range(3):  # it may be replaced while we read it
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            continue
    return None


def read_progress(run_dir: Path) -> dict[str, Any] | None:
    """Status of a run folder for the page (None if it has no manifest yet)."""
    manifest = read_manifest(run_dir)
    if manifest is None:
        return None
    recorded = manifest.get("steps", {})
    steps = []
    for name in STEP_NAMES:
        step = recorded.get(name, {})
        steps.append(
            {
                "name": name,
                "status": step.get("status", "pending"),
                "duration_s": step.get("duration_s"),
                "error": step.get("error"),
            }
        )

    brush = None
    if recorded.get("train", {}).get("status") == "running":
        lines = [
            p
            for p in map(
                parse_progress, _tail(Path(run_dir) / "logs" / "train.log").splitlines()
            )
            if p
        ]
        total = manifest.get("config", {}).get("train", {}).get("max_steps")
        if lines:
            brush = {"step": lines[-1][0], "total": total, "splats": lines[-1][1]}
        else:
            brush = {"step": 0, "total": total, "splats": None}

    result = None
    train = recorded.get("train", {})
    export = recorded.get("export", {})
    if export.get("status") == "succeeded":
        train_info, export_info = train.get("info", {}), export.get("info", {})
        result = {
            "train_seconds": train_info.get("train_seconds", train.get("duration_s")),
            "num_gaussians": export_info.get("num_gaussians"),
            "file_size_bytes": export_info.get("file_size_bytes"),
            "num_cameras": export_info.get("num_cameras"),
            "gpu": (train_info.get("gpu") or {}).get("name"),
        }
    return {
        "status": manifest.get("status"),
        "error": manifest.get("error"),
        "preset": manifest.get("config", {}).get("preset"),
        "steps": steps,
        "brush": brush,
        "result": result,
    }


# ---- runs ----------------------------------------------------------------------------


def colmap_has_cuda() -> bool:
    """True if the COLMAP found by the pipeline is a CUDA build (then run it on the GPU)."""
    try:
        exe = find_tool("colmap")
    except ToolNotFoundError:
        return False
    output = run_capture([exe, "-h"], timeout=60)
    return "with CUDA" in (output.stdout + output.stderr)


class Runner:
    """Runs the pipeline for one room at a time in a background thread."""

    def __init__(
        self,
        data_dir: Path,
        overrides: dict[str, Any] | None = None,
        run_fn: Callable[..., Any] = run_pipeline,
    ):
        self.data_dir = Path(data_dir)
        self.overrides = overrides or {}
        self.run_fn = run_fn
        self.room: str | None = None
        self.errors: dict[str, str] = {}  # room -> error not recorded in a manifest
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    def run_dir(self, room: str) -> Path:
        return self.data_dir / "runs" / room

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, room: str, preset: str) -> None:
        room = clean_room_name(room)
        if preset not in PRESETS:
            raise UserError(f"Unknown preset '{preset}': use one of {PRESETS}.")
        videos = list_videos(self.uploads_dir, room)
        if not videos:
            raise UserError(f"No videos uploaded for '{room}' yet.")
        config = load_config(preset, self.overrides)  # ConfigError if --set is wrong

        with self._lock:
            if self.running:
                raise Busy(
                    f"'{self.room}' is still running; wait until it has finished."
                )
            self.room = room
            self.errors.pop(room, None)
            self._thread = threading.Thread(
                target=self._run, args=(room, videos, config), daemon=True
            )
            self._thread.start()

    def resume(self, room: str) -> None:
        """Run again on the same folder: finished steps are skipped (resumability)."""
        room = clean_room_name(room)
        manifest = read_manifest(self.run_dir(room)) or {}
        preset = manifest.get("config", {}).get("preset") or "balanced"
        self.start(room, preset if preset in PRESETS else "balanced")

    def _run(self, room: str, videos: list[Path], config) -> None:
        try:
            self.run_fn(videos, self.run_dir(room), config)
        except StepError:
            pass  # recorded in manifest.json (status, failing step, message)
        except Exception as exc:  # e.g. a tool not found before any step started
            self.errors[room] = str(exc)

    def wait(self, timeout: float | None = None) -> None:
        """Wait for the current run (tests)."""
        if self._thread is not None:
            self._thread.join(timeout)


def ui_overrides(
    extra: dict[str, Any] | None = None, gpu_colmap: bool | None = None
) -> dict:
    """The UI's run settings: UI_OVERRIDES, COLMAP on the GPU if it is a CUDA build, then ``extra``."""
    overrides = deep_merge(UI_OVERRIDES, {})
    if gpu_colmap if gpu_colmap is not None else colmap_has_cuda():
        overrides = deep_merge(overrides, {"poses": {"use_gpu": True}})
    return deep_merge(overrides, extra or {})


# ---- viewer --------------------------------------------------------------------------


def default_viewer_dir() -> Path | None:
    """CPIPE_VIEWER_DIR, else ~/3DViewing if it exists."""
    configured = os.environ.get("CPIPE_VIEWER_DIR")
    if configured:
        return Path(configured)
    guess = Path.home() / "3DViewing"
    return guess if guess.is_dir() else None


def copy_to_viewer(run_dir: Path, viewer_dir: Path, room: str) -> str:
    """Copy scene.ply + cameras.json to <viewer>/public/runs/<room>/; return the viewer URL."""
    room = clean_room_name(room)
    files = [Path(run_dir) / "scene.ply", Path(run_dir) / "cameras.json"]
    missing = [f.name for f in files if not f.is_file()]
    if missing:
        raise UserError(f"The run has no {' or '.join(missing)} yet.")
    public = Path(viewer_dir) / "public"
    if not public.is_dir():
        raise UserError(f"No viewer at {viewer_dir} (no public/ folder).")
    target = public / "runs" / room
    target.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, target / f.name)
    return f"{VIEWER_URL}?run=runs/{room}"


def viewer_running(url: str = VIEWER_URL) -> bool:
    """Is the viewer's dev server (npm run dev) answering?"""
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            return response.status == 200
    except OSError:
        return False
