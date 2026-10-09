"""Step 2: extract frames from the video with FFmpeg, keeping the sharpest ones.

With ``blur_filter`` on, we extract ``oversample x target_count`` evenly spaced
candidate frames, split them into ``target_count`` consecutive groups and keep
the sharpest frame of each group (sharpness = variance of the Laplacian). This
drops motion-blurred frames while keeping even coverage of the whole video.

Selected frames are written to ``colmap/images/frame_00000.jpg`` ... in video
order (COLMAP's sequential matcher relies on that order).
"""

from __future__ import annotations

import math
import shutil
from collections.abc import Sequence

import numpy as np
from PIL import Image

from colibrimo_pipeline.errors import StepError
from colibrimo_pipeline.steps.base import CAPTURE_TIPS, StepContext, StepResult
from colibrimo_pipeline.utils.proc import find_tool, run_tool
from colibrimo_pipeline.utils.versions import ffmpeg_version

NAME = "extract_frames"


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


def run(ctx: StepContext) -> StepResult:
    cfg = ctx.config.frames
    video_info = ctx.manifest.step_info("validate").get("video")
    if not video_info:
        raise StepError(NAME, "Video metadata missing: run the 'validate' step first.")
    duration = float(video_info["duration_s"])
    ffmpeg = find_tool("ffmpeg")

    candidates_dir = ctx.run_dir / "frames_candidates"
    images_dir = ctx.run_dir / "colmap" / "images"
    for folder in (candidates_dir, images_dir):
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)

    n_wanted = (
        math.ceil(cfg.target_count * cfg.oversample)
        if cfg.blur_filter
        else cfg.target_count
    )
    fps = n_wanted / duration
    ctx.logger.info("Extracting about %d frames (%.3f per second)...", n_wanted, fps)
    run_tool(
        [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            ctx.video,
            "-vf",
            f"fps={fps:.6f},{scale_filter(cfg.max_long_side)}",
            "-q:v",
            str(cfg.jpeg_quality),
            "-start_number",
            "0",
            candidates_dir / "%05d.jpg",
        ],
        step=NAME,
        logger=ctx.logger,
    )
    candidates = sorted(candidates_dir.glob("*.jpg"))
    if not candidates:
        raise StepError(NAME, "FFmpeg produced no frames from this video.")

    ctx.logger.info("Scoring %d candidate frames for sharpness...", len(candidates))
    sharpness, thumbnails = zip(*(score_frame(p) for p in candidates), strict=True)
    motion = motion_score(thumbnails)
    if motion < cfg.min_motion:
        raise StepError(
            NAME,
            f"The video looks static (motion score {motion:.1f}, minimum {cfg.min_motion}). "
            f"The camera must move through the room. {CAPTURE_TIPS}",
            info={"motion_score": round(motion, 2), "num_candidates": len(candidates)},
        )

    selected = (
        select_sharpest(sharpness, cfg.target_count)
        if cfg.blur_filter
        else list(range(min(cfg.target_count, len(candidates))))
    )
    for new_index, old_index in enumerate(selected):
        shutil.copy2(candidates[old_index], images_dir / f"frame_{new_index:05d}.jpg")
    if not cfg.keep_candidates:
        shutil.rmtree(candidates_dir)

    with Image.open(images_dir / "frame_00000.jpg") as first:
        frame_size = first.size
    selected_set = set(selected)
    kept = [s for i, s in enumerate(sharpness) if i in selected_set]
    dropped = [s for i, s in enumerate(sharpness) if i not in selected_set]
    ctx.logger.info(
        "Kept %d of %d frames (%dx%d), motion score %.1f.",
        len(selected),
        len(candidates),
        *frame_size,
        motion,
    )
    return StepResult(
        outputs={"images_dir": images_dir},
        info={
            "num_candidates": len(candidates),
            "num_frames": len(selected),
            "frame_width": frame_size[0],
            "frame_height": frame_size[1],
            "extraction_fps": round(fps, 4),
            "blur_filter": cfg.blur_filter,
            "sharpness_min": round(min(kept), 2),
            "sharpness_median": round(float(np.median(kept)), 2),
            "sharpness_dropped_median": (
                round(float(np.median(dropped)), 2) if dropped else None
            ),
            "motion_score": round(motion, 2),
        },
        versions={"ffmpeg": ffmpeg_version(ffmpeg)},
    )
