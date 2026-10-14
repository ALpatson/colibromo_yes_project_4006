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
from colibrimo_pipeline.steps.validate import parse_video_info, probe_video
from colibrimo_pipeline.utils.proc import find_tool, run_capture
from colibrimo_pipeline.utils.videos import (
    VIDEO_EXTENSIONS,
    VideoInput,
    collect_videos,
    video_name,
)

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


# ---- labels (bottom / middle / up) and video details ---------------------------------

LABELS = ("bottom", "middle", "up")
LABELS_FILE = "labels.json"


def default_label(filename: str) -> str:
    """'middle_take2.mp4' -> 'middle'; '' when the name says nothing."""
    words = re.split(r"[^a-z]+", Path(filename).stem.lower())
    return next((label for label in LABELS if label in words), "")


def read_labels(uploads_dir: Path, room: str) -> dict[str, str]:
    """Label of each uploaded video: saved in labels.json, else from the file name.

    A room with a single video defaults to "middle" (its cameras then give the
    viewer's eye height); a label saved explicitly, even "none", wins.
    """
    saved: dict[str, str] = {}
    path = Path(uploads_dir) / room / LABELS_FILE
    if path.is_file():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            saved = {}
    videos = list_videos(uploads_dir, room)
    single = len(videos) == 1
    return {
        v.name: saved.get(v.name, default_label(v.name) or ("middle" if single else ""))
        for v in videos
    }


def set_label(
    uploads_dir: Path, room: str, filename: str, label: str
) -> dict[str, str]:
    """Label one video as bottom / middle / up ('' = none); each label at most once."""
    room = clean_room_name(room)
    labels = read_labels(uploads_dir, room)
    if filename not in labels:
        raise UserError(f"No video '{filename}' in room '{room}'.")
    if label not in ("", *LABELS):
        raise UserError(f"Label must be one of {', '.join(LABELS)} (or none).")
    if label and any(f != filename and lab == label for f, lab in labels.items()):
        raise UserError(f"Another video is already labelled '{label}'.")
    labels[filename] = label
    path = Path(uploads_dir) / room / LABELS_FILE
    path.write_text(json.dumps(labels, indent=1), encoding="utf-8")
    return labels


def named_videos(uploads_dir: Path, room: str) -> list[VideoInput]:
    """The room's videos with their run names: the label, else the file name."""
    labels = read_labels(uploads_dir, room)
    return collect_videos(
        [
            VideoInput(labels[v.name] or video_name(v), v)
            for v in list_videos(uploads_dir, room)
        ]
    )


_DETAILS_CACHE: dict[tuple[str, float, int], dict[str, Any]] = {}


def video_details(path: Path) -> dict[str, Any] | None:
    """Resolution, orientation, duration, fps, codec, HDR (ffprobe; cached per file)."""
    stat = path.stat()
    key = (str(path), stat.st_mtime, stat.st_size)
    if key not in _DETAILS_CACHE:
        try:
            info = parse_video_info(probe_video(find_tool("ffprobe"), path), path)
        except PipelineError:
            return None
        _DETAILS_CACHE[key] = {
            "width": info.width,
            "height": info.height,
            "orientation": "portrait" if info.height > info.width else "landscape",
            "duration_s": info.duration_s,
            "fps": info.fps,
            "codec": info.codec,
            "hdr": info.hdr,
        }
    return _DETAILS_CACHE[key]


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
    # "Continue anyway (partial room)" is offered when poses failed only because
    # too few frames were placed, and enough were placed to build something.
    poses_cfg = manifest.get("config", {}).get("poses", {})
    poses_step = recorded.get("poses", {})
    poses_info = poses_step.get("info", {})
    can_continue_partial = (
        poses_step.get("status") == "failed"
        and poses_info.get("registered_ratio", 1.0)
        < poses_cfg.get("min_registered_ratio", 0.7)
        and poses_info.get("num_registered", 0)
        >= poses_cfg.get("min_registered_frames", 20)
    )
    return {
        "status": manifest.get("status"),
        "error": manifest.get("error"),
        "partial_room": manifest.get("partial_room"),
        "can_continue_partial": can_continue_partial,
        "preset": manifest.get("config", {}).get("preset"),
        "warnings": recorded.get("validate", {}).get("info", {}).get("warnings", []),
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

    def inputs_changed(self, room: str, videos: list[VideoInput]) -> bool:
        """Do the room's videos or labels differ from its last run?"""
        previous = (read_manifest(self.run_dir(room)) or {}).get("input", {})
        if not previous:
            return False
        same_paths = [str(Path(p).resolve()) for p in previous.get("videos", [])] == [
            str(v.path) for v in videos
        ]
        same_names = previous.get("names") in (None, [v.name for v in videos])
        return not (same_paths and same_names)

    def start(
        self,
        room: str,
        preset: str,
        resume: bool = False,
        allow_partial: bool = False,
    ) -> None:
        """Run the room. A new Start with other videos/labels than the last run
        starts from scratch (old frames carry the old names); Resume reuses steps.
        ``allow_partial``: continue with a partial room (poses.allow_partial)."""
        room = clean_room_name(room)
        if preset not in PRESETS:
            raise UserError(f"Unknown preset '{preset}': use one of {PRESETS}.")
        if not list_videos(self.uploads_dir, room):
            raise UserError(f"No videos uploaded for '{room}' yet.")
        videos = named_videos(self.uploads_dir, room)  # ConfigError on clashing names
        overrides = self.overrides
        if allow_partial:
            overrides = deep_merge(overrides, {"poses": {"allow_partial": True}})
        config = load_config(preset, overrides)  # ConfigError if --set is wrong
        force = not resume and self.inputs_changed(room, videos)

        with self._lock:
            if self.running:
                raise Busy(
                    f"'{self.room}' is still running; wait until it has finished."
                )
            self.room = room
            self.errors.pop(room, None)
            self._thread = threading.Thread(
                target=self._run, args=(room, videos, config, force), daemon=True
            )
            self._thread.start()

    def resume(self, room: str, allow_partial: bool | None = None) -> None:
        """Run again on the same folder: finished steps are skipped (resumability).

        A run that already continued as a partial room keeps doing so.
        """
        room = clean_room_name(room)
        config = (read_manifest(self.run_dir(room)) or {}).get("config", {})
        preset = config.get("preset") or "balanced"
        if allow_partial is None:
            allow_partial = bool(config.get("poses", {}).get("allow_partial"))
        self.start(
            room,
            preset if preset in PRESETS else "balanced",
            resume=True,
            allow_partial=allow_partial,
        )

    def continue_partial(self, room: str) -> None:
        """'Continue anyway': resume with a partial room (COLMAP's result is reused)."""
        progress = read_progress(self.run_dir(clean_room_name(room)))
        if not progress or not progress["can_continue_partial"]:
            raise UserError(
                "Nothing to continue: this only applies when COLMAP placed too few "
                "frames (but at least the minimum)."
            )
        self.resume(room, allow_partial=True)

    def _run(self, room: str, videos: list[VideoInput], config, force: bool) -> None:
        try:
            self.run_fn(videos, self.run_dir(room), config, force=force)
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
