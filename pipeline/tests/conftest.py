"""Shared fixtures. Tests needing FFmpeg/COLMAP or a GPU are skipped when unavailable."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from colibrimo_pipeline.errors import ToolNotFoundError
from colibrimo_pipeline.utils.proc import find_tool


def _tool_available(name: str) -> bool:
    try:
        find_tool(name)
        return True
    except ToolNotFoundError:
        return False


def _cuda_available() -> bool:
    try:
        import torch  # noqa: PLC0415 (optional dependency)

        return torch.cuda.is_available()
    except Exception:
        return False


HAS_FFMPEG = _tool_available("ffmpeg") and _tool_available("ffprobe")
HAS_COLMAP = _tool_available("colmap")


def pytest_collection_modifyitems(config, items):
    skip_tools = pytest.mark.skip(
        reason="FFmpeg/ffprobe not found (see docs/pipeline.md)"
    )
    skip_gpu = pytest.mark.skip(reason="needs an NVIDIA GPU with CUDA")
    has_cuda = None
    for item in items:
        if "tools" in item.keywords and not HAS_FFMPEG:
            item.add_marker(skip_tools)
        if "gpu" in item.keywords:
            if has_cuda is None:
                has_cuda = _cuda_available()
            if not has_cuda:
                item.add_marker(skip_gpu)


def make_video(
    path: Path,
    duration: float,
    size: str = "1920x1080",
    moving: bool = True,
    rate: int = 10,
) -> Path:
    """Generate a synthetic test video with FFmpeg.

    moving=True: rotating test pattern (lots of motion).
    moving=False: flat grey image with per-frame sensor-like noise (a "static shot").
    """
    ffmpeg = find_tool("ffmpeg")
    if moving:
        source = f"testsrc2=size={size}:rate={rate}"
        video_filter = "rotate=a=0.4*t"
    else:
        source = f"color=c=gray:size={size}:rate={rate}"
        video_filter = "noise=alls=10:allf=t+u"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            source,
            "-t",
            str(duration),
            "-vf",
            video_filter,
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",
            str(path),
        ],
        check=True,
    )
    return path


@pytest.fixture(scope="session")
def video_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("videos")


@pytest.fixture(scope="session")
def good_video(video_dir) -> Path:
    """25 s, 1080p, lots of motion: passes validation and frame extraction."""
    return make_video(video_dir / "good.mp4", duration=25)


@pytest.fixture(scope="session")
def short_video(video_dir) -> Path:
    """5 s: too short, must fail validation."""
    return make_video(video_dir / "short.mp4", duration=5)


@pytest.fixture(scope="session")
def static_video(video_dir) -> Path:
    """25 s of a camera that never moves: must fail frame extraction."""
    return make_video(video_dir / "static.mp4", duration=25, moving=False)
