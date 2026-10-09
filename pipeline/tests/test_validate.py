from pathlib import Path

import pytest

from colibrimo_pipeline.config import ValidationConfig
from colibrimo_pipeline.errors import StepError
from colibrimo_pipeline.steps.validate import (
    VideoInfo,
    check_video,
    parse_video_info,
    probe_video,
)
from colibrimo_pipeline.utils.proc import find_tool

# Shape of real ffprobe output for an iPhone portrait video (trimmed).
IPHONE_PORTRAIT_PROBE = {
    "streams": [
        {"codec_type": "audio", "codec_name": "aac"},
        {
            "codec_type": "video",
            "codec_name": "hevc",
            "width": 1920,
            "height": 1080,
            "avg_frame_rate": "30000/1001",
            "r_frame_rate": "30/1",
            "side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}],
        },
    ],
    "format": {"duration": "95.5", "size": "123456789", "format_name": "mov,mp4,m4a"},
}


def info(duration=60.0, width=1920, height=1080, fps=30.0):
    return VideoInfo("v.mp4", duration, width, height, fps, "h264", 0, 1, "mp4")


def test_parse_portrait_phone_video_swaps_dimensions():
    result = parse_video_info(IPHONE_PORTRAIT_PROBE, Path("room.mov"))
    assert (result.width, result.height) == (1080, 1920)
    assert result.rotation == -90
    assert result.fps == 29.97
    assert result.duration_s == 95.5
    assert result.codec == "hevc"
    assert result.size_bytes == 123456789


def test_parse_rejects_file_without_video_stream():
    with pytest.raises(StepError, match="no video stream"):
        parse_video_info(
            {"streams": [{"codec_type": "audio"}], "format": {}}, Path("a.m4a")
        )


def test_check_accepts_normal_video():
    assert check_video(info(), ValidationConfig()) == []


@pytest.mark.parametrize("duration", [5.0, 19.9])
def test_check_rejects_too_short(duration):
    with pytest.raises(StepError, match="at least 20 s"):
        check_video(info(duration=duration), ValidationConfig())


def test_check_rejects_too_long():
    with pytest.raises(StepError, match="at most 5 min"):
        check_video(info(duration=301), ValidationConfig())


def test_check_warns_but_accepts_low_resolution():
    warnings = check_video(info(width=1280, height=720), ValidationConfig())
    assert len(warnings) == 1 and "below 1080p" in warnings[0]


def test_check_portrait_1080p_is_not_low_resolution():
    assert check_video(info(width=1080, height=1920), ValidationConfig()) == []


@pytest.mark.tools
def test_probe_real_video(good_video):
    result = parse_video_info(probe_video(find_tool("ffprobe"), good_video), good_video)
    assert (result.width, result.height) == (1920, 1080)
    assert result.duration_s == pytest.approx(25, abs=0.5)


@pytest.mark.tools
def test_probe_rejects_non_video_file(tmp_path):
    fake = tmp_path / "notes.mp4"
    fake.write_text("this is not a video")
    with pytest.raises(StepError, match="cannot read"):
        probe_video(find_tool("ffprobe"), fake)
