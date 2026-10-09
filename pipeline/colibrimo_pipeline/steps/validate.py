"""Step 1: check the input video and record its metadata (ffprobe)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from colibrimo_pipeline.config import ValidationConfig
from colibrimo_pipeline.errors import StepError
from colibrimo_pipeline.steps.base import StepContext, StepResult
from colibrimo_pipeline.utils.proc import find_tool, run_capture
from colibrimo_pipeline.utils.versions import ffmpeg_version

NAME = "validate"


@dataclass
class VideoInfo:
    """Video metadata. ``width``/``height`` are as displayed (after phone rotation)."""

    path: str
    duration_s: float
    width: int
    height: int
    fps: float | None
    codec: str | None
    rotation: int
    size_bytes: int | None
    container: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe_video(ffprobe: str, video: Path) -> dict[str, Any]:
    """Run ffprobe and return its JSON output."""
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
    ]
    result = run_capture([*cmd, str(video)])
    if result.returncode != 0:
        reason = (result.stderr.strip().splitlines() or ["unknown error"])[-1]
        raise StepError(
            NAME, f"FFmpeg cannot read '{video.name}' ({reason}). Is it a video file?"
        )
    return json.loads(result.stdout)


def _fraction(text: str | None) -> float | None:
    """'30000/1001' -> 29.97; None for missing or '0/0'."""
    if not text:
        return None
    num, _, den = text.partition("/")
    try:
        value = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return None
    return round(value, 3) if value > 0 else None


def _rotation(stream: dict[str, Any]) -> int:
    """Phone videos store their orientation as a rotation (side data or 'rotate' tag)."""
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            return int(round(float(side_data["rotation"])))
    rotate_tag = stream.get("tags", {}).get("rotate")
    return int(rotate_tag) if rotate_tag else 0


def parse_video_info(probe: dict[str, Any], video: Path) -> VideoInfo:
    """Extract the fields we care about from ffprobe's JSON."""
    streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        raise StepError(NAME, f"'{video.name}' contains no video stream.")
    stream = streams[0]
    fmt = probe.get("format", {})

    width, height = int(stream.get("width", 0)), int(stream.get("height", 0))
    rotation = _rotation(stream)
    if abs(rotation) % 180 == 90:  # portrait video stored as landscape + rotation
        width, height = height, width

    duration = float(fmt.get("duration") or stream.get("duration") or 0)
    size = fmt.get("size")
    return VideoInfo(
        path=str(video),
        duration_s=round(duration, 3),
        width=width,
        height=height,
        fps=_fraction(stream.get("avg_frame_rate"))
        or _fraction(stream.get("r_frame_rate")),
        codec=stream.get("codec_name"),
        rotation=rotation,
        size_bytes=int(size) if size else None,
        container=fmt.get("format_name"),
    )


def check_video(info: VideoInfo, cfg: ValidationConfig) -> list[str]:
    """Raise StepError if the video is unusable; return warnings for minor issues."""
    if info.duration_s <= 0:
        raise StepError(
            NAME, "Could not determine the video duration. Is the file complete?"
        )
    if info.duration_s < cfg.min_duration_s:
        raise StepError(
            NAME,
            f"The video is {info.duration_s:.1f} s long, but at least "
            f"{cfg.min_duration_s:.0f} s is needed. Film the room for 1 to 3 minutes, "
            "walking slowly around it (see docs/capture-guide.md).",
        )
    if info.duration_s > cfg.max_duration_s:
        raise StepError(
            NAME,
            f"The video is {info.duration_s / 60:.1f} min long, but at most "
            f"{cfg.max_duration_s / 60:.0f} min is allowed. Trim it, or film one room per video.",
        )

    warnings = []
    short_side = min(info.width, info.height)
    if short_side < cfg.min_resolution:
        warnings.append(
            f"Resolution {info.width}x{info.height} is below {cfg.min_resolution}p: "
            "the 3D result may be blurry. Film in 1080p or 4K if possible."
        )
    if info.fps is None:
        warnings.append("Could not read the frame rate; continuing anyway.")
    return warnings


def run(ctx: StepContext) -> StepResult:
    if not ctx.video.is_file():
        raise StepError(NAME, f"Video file not found: {ctx.video}")
    ffprobe = find_tool("ffprobe")

    info = parse_video_info(probe_video(ffprobe, ctx.video), ctx.video)
    warnings = check_video(info, ctx.config.validation)
    for warning in warnings:
        ctx.logger.warning("Warning: %s", warning)
    ctx.logger.info(
        "Video: %.1f s, %dx%d, %s fps, %s",
        info.duration_s,
        info.width,
        info.height,
        info.fps,
        info.codec,
    )

    out = ctx.run_dir / "input" / "video_info.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info.to_dict(), indent=2), encoding="utf-8")
    return StepResult(
        outputs={"video_info": out},
        info={"video": info.to_dict(), "warnings": warnings},
        versions={"ffprobe": ffmpeg_version(ffprobe)},
    )
