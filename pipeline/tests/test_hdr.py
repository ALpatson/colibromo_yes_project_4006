"""HDR (iPhone HLG / PQ) detection and tone-mapping to SDR."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from colibrimo_pipeline.config import FramesConfig, load_config
from colibrimo_pipeline.errors import StepError
from colibrimo_pipeline.pipeline import run_pipeline
from colibrimo_pipeline.steps import extract_frames
from colibrimo_pipeline.steps.extract_frames import (
    missing_filters,
    needs_tonemap,
    video_filter,
)
from colibrimo_pipeline.steps.validate import parse_video_info
from colibrimo_pipeline.utils.proc import find_tool, run_capture


def probe(transfer=None, primaries=None, space=None, extra_streams=0):
    """Shape of ffprobe output for a phone video with the given colour tags."""
    stream = {
        "codec_type": "video",
        "codec_name": "hevc",
        "width": 1920,
        "height": 1080,
        "pix_fmt": "yuv420p10le",
        "avg_frame_rate": "30/1",
        "side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}],
    }
    for key, value in (
        ("color_transfer", transfer),
        ("color_primaries", primaries),
        ("color_space", space),
    ):
        if value:
            stream[key] = value
    others = [{"codec_type": "data"} for _ in range(extra_streams)]
    return {
        "streams": [stream, {"codec_type": "audio"}, *others],
        "format": {"duration": "80.3", "format_name": "mov,mp4,m4a"},
    }


# ---- detection (validate) -----------------------------------------------------


def test_iphone_hlg_is_hdr():
    info = parse_video_info(
        probe("arib-std-b67", "bt2020", "bt2020nc", extra_streams=6), Path("room1.MOV")
    )
    assert info.hdr is True
    assert info.hdr_format == "HLG"
    assert (info.color_primaries, info.color_space) == ("bt2020", "bt2020nc")
    assert info.pix_fmt == "yuv420p10le"
    assert info.num_streams == 8


def test_pq_is_hdr():
    info = parse_video_info(probe("smpte2084", "bt2020", "bt2020nc"), Path("a.mp4"))
    assert (info.hdr, info.hdr_format) == (True, "PQ")


@pytest.mark.parametrize("transfer", ["bt709", None, "unknown"])
def test_sdr_is_not_hdr(transfer):
    info = parse_video_info(probe(transfer, "bt709", "bt709"), Path("a.mp4"))
    assert (info.hdr, info.hdr_format) == (False, None)


# ---- filters (extract_frames) ---------------------------------------------------

HLG = {
    "hdr": True,
    "hdr_format": "HLG",
    "color_transfer": "arib-std-b67",
    "color_primaries": "bt2020",
    "color_space": "bt2020nc",
}


def test_hdr_filter_chain_matches_the_manual_recipe():
    chain = video_filter(HLG, 2.0, FramesConfig(max_long_side=1600))
    assert chain.startswith(
        "fps=2.000000,zscale=tin=arib-std-b67:min=bt2020nc:pin=bt2020"
    )
    assert ":t=linear:npl=100,format=gbrpf32le,zscale=p=bt709," in chain
    assert (
        "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p,"
        in chain
    )
    assert chain.endswith("min(1600,ih))'")  # our usual downscale comes last


def test_pq_uses_its_own_transfer():
    pq = {**HLG, "hdr_format": "PQ", "color_transfer": "smpte2084"}
    assert "zscale=tin=smpte2084:" in video_filter(pq, 1.0, FramesConfig())


def test_missing_matrix_and_primaries_default_to_bt2020():
    chain = video_filter(
        {"hdr": True, "color_transfer": "arib-std-b67"}, 1.0, FramesConfig()
    )
    assert "min=bt2020nc:pin=bt2020:" in chain


def test_sdr_and_disabled_tonemap_use_the_plain_chain():
    sdr = {"hdr": False, "color_transfer": "bt709"}
    assert "zscale" not in video_filter(sdr, 1.0, FramesConfig())
    off = FramesConfig(hdr_tonemap=False)
    assert not needs_tonemap(HLG, off)
    assert "zscale" not in video_filter(HLG, 1.0, off)


def test_tonemap_operator_is_configurable():
    chain = video_filter(HLG, 1.0, FramesConfig(tonemap="mobius", hdr_peak_nits=203))
    assert "tonemap=tonemap=mobius" in chain and "npl=203" in chain


# ---- real FFmpeg ------------------------------------------------------------------


def _can_make_hdr_clip() -> bool:
    try:
        ffmpeg = find_tool("ffmpeg")
    except Exception:
        return False
    encoders = run_capture([ffmpeg, "-hide_banner", "-encoders"]).stdout
    return " libx264 " in encoders and not missing_filters(
        ffmpeg, ["zscale", "tonemap"]
    )


needs_hdr_tools = pytest.mark.skipif(
    not _can_make_hdr_clip(), reason="FFmpeg without libx264 or zscale/tonemap"
)


def make_hlg_video(path: Path, duration: float = 6) -> Path:
    """Moving test pattern tagged as HLG / BT.2020 (like an iPhone video)."""
    subprocess.run(
        [
            find_tool("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1280x720:rate=10",
            "-t",
            str(duration),
            "-vf",
            "rotate=a=0.4*t",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-x264-params",
            "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc",
            str(path),
        ],
        check=True,
    )
    return path


def mean_pixel_difference(folder_a: Path, folder_b: Path) -> float:
    """Average absolute RGB difference (0-255) between same-named frames."""
    diffs = [
        np.abs(
            np.asarray(Image.open(a), dtype=float)
            - np.asarray(Image.open(folder_b / a.name), dtype=float)
        ).mean()
        for a in sorted(folder_a.glob("*.jpg"))
    ]
    return float(np.mean(diffs))


@pytest.mark.tools
@needs_hdr_tools
def test_hlg_video_is_detected_and_tonemapped(tmp_path):
    video = make_hlg_video(tmp_path / "iphone.mp4")
    overrides = {
        "validation": {"min_duration_s": 2},
        "frames": {"target_count": 10, "max_long_side": 320},
    }
    run = tmp_path / "on"
    manifest = run_pipeline(
        video, run, load_config("fast", overrides), until="extract_frames"
    )

    validate = manifest.step_info("validate")
    assert validate["videos"][0]["hdr"] is True
    assert validate["videos"][0]["hdr_format"] == "HLG"
    assert validate["hdr_videos"] == ["iphone"]
    summary = manifest.step_info("extract_frames")["videos"][0]
    assert summary["tonemapped"] is True
    assert len(list((run / "colmap" / "images").glob("iphone_*.jpg"))) == 10
    assert (
        "zscale=tin=arib-std-b67" in (run / "logs" / "extract_frames.log").read_text()
    )

    # Without tone-mapping the same frames come out with different colours.
    # (The test pattern is already saturated, so compare pixels, not saturation;
    # about 9 on average with FFmpeg 9.0.2. Real iPhone footage: docs D-012.)
    off = tmp_path / "off"
    overrides["frames"]["hdr_tonemap"] = False
    run_pipeline(video, off, load_config("fast", overrides), until="extract_frames")
    data = json.loads((off / "manifest.json").read_text())
    assert data["steps"]["extract_frames"]["info"]["videos"][0]["tonemapped"] is False
    assert (
        mean_pixel_difference(run / "colmap" / "images", off / "colmap" / "images") > 4
    )


def test_missing_zscale_fails_clearly(monkeypatch, tmp_path):
    """An FFmpeg without zscale must not silently produce washed-out frames."""
    monkeypatch.setattr(extract_frames, "find_tool", lambda name: "ffmpeg")
    monkeypatch.setattr(
        extract_frames, "missing_filters", lambda ffmpeg, names: ["zscale"]
    )

    class Ctx:
        config = load_config("fast")
        run_dir = tmp_path

        class manifest:  # noqa: N801 (minimal stand-in)
            @staticmethod
            def step_info(name):
                return {
                    "videos": [
                        {"name": "room1", "path": "room1.MOV", "duration_s": 80, **HLG}
                    ]
                }

    with pytest.raises(StepError, match="zscale filter"):
        extract_frames.run(Ctx())
