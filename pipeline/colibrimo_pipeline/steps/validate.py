"""Step 1: check the input videos and record their metadata (ffprobe).

A room can be filmed as several videos (e.g. at three heights). Each video is
checked on its own (readable, at most ``max_duration_s``); the minimum duration
applies to all videos together. HDR video (iPhone HLG, or PQ) is detected from
the main video stream's ``color_transfer`` and recorded, so ``extract_frames``
can tone-map it to SDR.
"""

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


# Transfer characteristics (ffprobe ``color_transfer``) of HDR video: HLG, used
# by iPhones, and PQ (HDR10 / Dolby Vision profile 8.1 base layers).
HDR_TRANSFERS = {"arib-std-b67": "HLG", "smpte2084": "PQ"}


@dataclass
class VideoInfo:
    """Video metadata of the main video stream (the first one, ``0:v:0``).

    ``width``/``height`` are as displayed (after phone rotation). ``hdr`` is
    True for HLG or PQ video, which is tone-mapped to SDR when extracting frames.
    """

    path: str
    duration_s: float
    width: int
    height: int
    fps: float | None
    codec: str | None
    rotation: int
    size_bytes: int | None
    container: str | None
    pix_fmt: str | None = None
    color_transfer: str | None = None
    color_primaries: str | None = None
    color_space: str | None = None
    hdr: bool = False
    hdr_format: str | None = None  # "HLG" or "PQ"
    num_streams: int = 1

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
    transfer = stream.get("color_transfer")
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
        pix_fmt=stream.get("pix_fmt"),
        color_transfer=transfer,
        color_primaries=stream.get("color_primaries"),
        color_space=stream.get("color_space"),
        hdr=transfer in HDR_TRANSFERS,
        hdr_format=HDR_TRANSFERS.get(transfer),
        num_streams=len(probe.get("streams", [])),
    )


def check_video(
    info: VideoInfo, cfg: ValidationConfig, label: str = "The video"
) -> list[str]:
    """Checks for one video: raise StepError if unusable, return warnings.

    The minimum duration applies to all videos together (see check_videos).
    """
    if info.duration_s <= 0:
        raise StepError(
            NAME,
            f"Could not determine the duration of {Path(info.path).name}. "
            "Is the file complete?",
        )
    if info.duration_s > cfg.max_duration_s:
        raise StepError(
            NAME,
            f"{label} is {info.duration_s / 60:.1f} min long, but at most "
            f"{cfg.max_duration_s / 60:.0f} min per video is allowed. Trim it, or "
            "film one room per video.",
        )

    warnings = []
    short_side = min(info.width, info.height)
    if short_side < cfg.min_resolution:
        warnings.append(
            f"{label}: resolution {info.width}x{info.height} is below "
            f"{cfg.min_resolution}p: the 3D result may be blurry. Film in 1080p or 4K "
            "if possible."
        )
    if info.fps is None:
        warnings.append(f"{label}: could not read the frame rate; continuing anyway.")
    return warnings


def check_videos(infos: list[VideoInfo], cfg: ValidationConfig) -> list[str]:
    """Check every video, then the total duration. Returns all warnings."""
    several = len(infos) > 1
    warnings: list[str] = []
    for info in infos:
        label = f"Video '{Path(info.path).name}'" if several else "The video"
        warnings += check_video(info, cfg, label)

    total = sum(info.duration_s for info in infos)
    if total < cfg.min_duration_s:
        what = (
            f"The videos are {total:.1f} s long in total"
            if several
            else (f"The video is {total:.1f} s long")
        )
        raise StepError(
            NAME,
            f"{what}, but at least {cfg.min_duration_s:.0f} s is needed. Film the room "
            "for 1 to 3 minutes, walking slowly around it (see docs/capture-guide.md).",
        )
    if several and len({(i.width, i.height) for i in infos}) > 1:
        warnings.append(
            "The videos have different resolutions or orientations; each video gets "
            "its own camera model in COLMAP. Film all passes with the same settings "
            "if possible."
        )
    return warnings


def same_camera(infos: list[VideoInfo]) -> bool:
    """True if all videos have the same displayed resolution (assume the same lens)."""
    return len({(info.width, info.height) for info in infos}) == 1


def run(ctx: StepContext) -> StepResult:
    for video in ctx.videos:
        if not video.path.is_file():
            raise StepError(NAME, f"Video file not found: {video.path}")
    ffprobe = find_tool("ffprobe")

    infos = [
        parse_video_info(probe_video(ffprobe, video.path), video.path)
        for video in ctx.videos
    ]
    warnings = check_videos(infos, ctx.config.validation)
    for warning in warnings:
        ctx.logger.warning("Warning: %s", warning)
    for video, info in zip(ctx.videos, infos, strict=True):
        ctx.logger.info(
            "Video '%s': %.1f s, %dx%d, %s fps, %s%s",
            video.name,
            info.duration_s,
            info.width,
            info.height,
            info.fps,
            info.codec,
            (
                f", HDR ({info.hdr_format}, {info.color_transfer}): frames will be "
                "tone-mapped to SDR"
                if info.hdr
                else ""
            ),
        )

    videos = [
        {"name": video.name, **info.to_dict()}
        for video, info in zip(ctx.videos, infos, strict=True)
    ]
    out = ctx.run_dir / "input" / "videos.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(videos, indent=2), encoding="utf-8")
    return StepResult(
        outputs={"videos_info": out},
        info={
            "videos": videos,
            "num_videos": len(videos),
            "total_duration_s": round(sum(i.duration_s for i in infos), 3),
            "same_camera": same_camera(infos),
            "hdr_videos": [
                v.name for v, i in zip(ctx.videos, infos, strict=True) if i.hdr
            ],
            "warnings": warnings,
        },
        versions={"ffprobe": ffmpeg_version(ffprobe)},
    )
