"""Step 2: extract frames from the videos with FFmpeg, keeping the sharpest ones.

``target_count`` frames are shared between the videos in proportion to their
duration (each keeps at least ``min_per_video``). For each video, with
``blur_filter`` on, we extract ``oversample x`` its share as evenly spaced
candidate frames, split them into consecutive groups and keep the sharpest
frame of each group (sharpness = variance of the Laplacian). This drops
motion-blurred frames while keeping even coverage of the whole video.

Frames are named after their video and numbered in video order
(``colmap/images/bottom_00000.jpg`` ...), so frames of one video stay together
and in order (COLMAP's sequential matcher relies on that). ``frames.json``
lists every frame with its source video and timestamp.
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from colibrimo_pipeline.config import FramesConfig
from colibrimo_pipeline.errors import StepError
from colibrimo_pipeline.steps.base import CAPTURE_TIPS, StepContext, StepResult
from colibrimo_pipeline.utils.proc import find_tool, run_tool
from colibrimo_pipeline.utils.versions import ffmpeg_version
from colibrimo_pipeline.utils.videos import video_name

NAME = "extract_frames"
FRAMES_FILE = "frames.json"


def split_frame_counts(
    durations: Sequence[float], total: int, minimum: int = 1
) -> list[int]:
    """Share ``total`` frames between videos in proportion to their durations.

    Every video gets at least ``minimum`` frames (lowered if ``total`` is too
    small for that); the counts always add up to ``total`` exactly (largest
    remainder rounding).
    """
    n = len(durations)
    if n == 0:
        return []
    minimum = max(0, min(minimum, total // n))
    weights = np.asarray(durations, dtype=float)
    weights = weights / weights.sum() if weights.sum() > 0 else np.full(n, 1.0 / n)

    ideal = weights * total
    counts = np.floor(ideal).astype(int)
    for i in np.argsort(-(ideal - counts), kind="stable")[: total - counts.sum()]:
        counts[i] += 1

    # Raise short videos to the minimum, taking frames from the largest shares.
    for i in range(n):
        while counts[i] < minimum:
            donor = int(np.argmax(counts))
            counts[donor] -= 1
            counts[i] += 1
    return [int(c) for c in counts]


def laplacian_variance(gray: np.ndarray) -> float:
    """Sharpness score of a grayscale image: higher = sharper."""
    g = gray.astype(np.float32)
    lap = g[1:-1, 2:] + g[1:-1, :-2] + g[2:, 1:-1] + g[:-2, 1:-1] - 4.0 * g[1:-1, 1:-1]
    return float(lap.var())


def score_frame(
    path, blur_size: int = 640, thumb_size: int = 64
) -> tuple[float, np.ndarray]:
    """Return (sharpness, small grayscale thumbnail) for one image file."""
    with Image.open(path) as image:
        gray = image.convert("L")
    small = gray.copy()
    small.thumbnail((blur_size, blur_size))
    thumb = gray.resize((thumb_size, thumb_size), Image.Resampling.BILINEAR)
    return laplacian_variance(np.asarray(small)), np.asarray(thumb, dtype=np.float32)


def select_sharpest(scores: Sequence[float], target: int) -> list[int]:
    """Indices of the sharpest frame in each of ``target`` consecutive groups."""
    n = len(scores)
    if n <= target:
        return list(range(n))
    values = np.asarray(scores)
    groups = np.array_split(np.arange(n), target)
    return [int(group[np.argmax(values[group])]) for group in groups]


def motion_score(thumbnails: Sequence[np.ndarray]) -> float:
    """Largest mean absolute difference (0-255) between any frame and the first one.

    Close to 0 for a camera that never moves; large when walking around a room.
    """
    if len(thumbnails) < 2:
        return 0.0
    first = thumbnails[0]
    return float(max(np.abs(t - first).mean() for t in thumbnails[1:]))


def scale_filter(max_long_side: int) -> str:
    """FFmpeg filter: downscale so the long side is at most ``max_long_side`` (never upscale)."""
    side = max_long_side
    return (
        f"scale=w='if(gte(iw,ih),min({side},iw),-2)'"
        f":h='if(gte(iw,ih),-2,min({side},ih))'"
    )


def videos_from_manifest(ctx: StepContext) -> list[dict[str, Any]]:
    """Video metadata recorded by ``validate`` (one dict per video, with ``name``)."""
    info = ctx.manifest.step_info("validate")
    if info.get("videos"):
        return info["videos"]
    if info.get("video"):  # run folder from before multi-video support
        old = info["video"]
        return [{"name": video_name(Path(old["path"])), **old}]
    raise StepError(NAME, "Video metadata missing: run the 'validate' step first.")


def extract_candidates(
    ffmpeg: str,
    video: Path,
    out_dir: Path,
    fps: float,
    cfg: FramesConfig,
    ctx: StepContext,
) -> list[Path]:
    """Extract evenly spaced frames (``fps`` per second) as candidate JPEGs."""
    run_tool(
        [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            video,
            "-vf",
            f"fps={fps:.6f},{scale_filter(cfg.max_long_side)}",
            "-q:v",
            str(cfg.jpeg_quality),
            "-start_number",
            "0",
            out_dir / "%05d.jpg",
        ],
        step=NAME,
        logger=ctx.logger,
    )
    return sorted(out_dir.glob("*.jpg"))


def run(ctx: StepContext) -> StepResult:
    cfg = ctx.config.frames
    videos = videos_from_manifest(ctx)
    ffmpeg = find_tool("ffmpeg")

    candidates_root = ctx.run_dir / "frames_candidates"
    images_dir = ctx.run_dir / "colmap" / "images"
    for folder in (candidates_root, images_dir):
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)

    counts = split_frame_counts(
        [float(v["duration_s"]) for v in videos], cfg.target_count, cfg.min_per_video
    )
    frames: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    all_kept: list[float] = []
    all_dropped: list[float] = []
    for video, count in zip(videos, counts, strict=True):
        name, duration = video["name"], float(video["duration_s"])
        n_wanted = math.ceil(count * cfg.oversample) if cfg.blur_filter else count
        fps = n_wanted / duration
        label = f"'{name}'" if len(videos) > 1 else "the video"
        ctx.logger.info(
            "Extracting about %d candidate frames from %s (%.3f per second) to keep %d...",
            n_wanted,
            label,
            fps,
            count,
        )
        candidates_dir = candidates_root / name
        candidates_dir.mkdir()
        candidates = extract_candidates(
            ffmpeg, Path(video["path"]), candidates_dir, fps, cfg, ctx
        )
        if not candidates:
            raise StepError(NAME, f"FFmpeg produced no frames from {label}.")

        sharpness, thumbnails = zip(*(score_frame(p) for p in candidates), strict=True)
        motion = motion_score(thumbnails)
        if motion < cfg.min_motion:
            raise StepError(
                NAME,
                f"{label[0].upper() + label[1:]} looks static (motion score {motion:.1f}, "
                f"minimum {cfg.min_motion}). The camera must move through the room. "
                f"{CAPTURE_TIPS}",
                info={"video": name, "motion_score": round(motion, 2)},
            )

        selected = (
            select_sharpest(sharpness, count)
            if cfg.blur_filter
            else list(range(min(count, len(candidates))))
        )
        for new_index, old_index in enumerate(selected):
            file_name = f"{name}_{new_index:05d}.jpg"
            shutil.copy2(candidates[old_index], images_dir / file_name)
            frames.append(
                {
                    "file": file_name,
                    "video": name,
                    "index": new_index,
                    "time_s": round(old_index / fps, 3),
                    "sharpness": round(sharpness[old_index], 2),
                }
            )
        selected_set = set(selected)
        all_kept += [s for i, s in enumerate(sharpness) if i in selected_set]
        all_dropped += [s for i, s in enumerate(sharpness) if i not in selected_set]
        summaries.append(
            {
                "name": name,
                "num_candidates": len(candidates),
                "num_frames": len(selected),
                "first_frame": f"{name}_00000.jpg",
                "last_frame": f"{name}_{len(selected) - 1:05d}.jpg",
                "extraction_fps": round(fps, 4),
                "motion_score": round(motion, 2),
            }
        )
        ctx.logger.info(
            "%s: kept %d of %d frames, motion score %.1f.",
            name,
            len(selected),
            len(candidates),
            motion,
        )

    if not cfg.keep_candidates:
        shutil.rmtree(candidates_root)
    frames_file = ctx.run_dir / FRAMES_FILE
    frames_file.write_text(json.dumps({"frames": frames}, indent=1), encoding="utf-8")

    with Image.open(images_dir / frames[0]["file"]) as first:
        frame_size = first.size
    ctx.logger.info(
        "Kept %d frames (%dx%d) from %d video(s).",
        len(frames),
        *frame_size,
        len(videos),
    )
    return StepResult(
        outputs={"images_dir": images_dir, "frames": frames_file},
        info={
            "num_videos": len(videos),
            "num_candidates": sum(s["num_candidates"] for s in summaries),
            "num_frames": len(frames),
            "frame_width": frame_size[0],
            "frame_height": frame_size[1],
            "blur_filter": cfg.blur_filter,
            "sharpness_min": round(min(all_kept), 2),
            "sharpness_median": round(float(np.median(all_kept)), 2),
            "sharpness_dropped_median": (
                round(float(np.median(all_dropped)), 2) if all_dropped else None
            ),
            # Lowest of all videos: the least moving one decides.
            "motion_score": min(s["motion_score"] for s in summaries),
            "videos": summaries,
        },
        versions={"ffmpeg": ffmpeg_version(ffmpeg)},
    )
