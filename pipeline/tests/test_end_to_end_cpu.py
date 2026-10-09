"""Real FFmpeg runs on synthetic videos (no GPU, no COLMAP needed).

Covers the acceptance criterion "a deliberately bad video (5 s, or a static
shot) fails early with a clear, human-readable error".
"""

import json

import pytest
from PIL import Image

from colibrimo_pipeline.cli import main
from colibrimo_pipeline.config import load_config
from colibrimo_pipeline.errors import StepError
from colibrimo_pipeline.pipeline import run_pipeline

pytestmark = pytest.mark.tools


def test_good_video_validate_and_extract(good_video, tmp_path):
    config = load_config("fast", {"frames": {"target_count": 20, "max_long_side": 640}})
    manifest = run_pipeline(
        good_video, tmp_path / "run", config, until="extract_frames"
    )

    images = sorted((tmp_path / "run" / "colmap" / "images").glob("*.jpg"))
    assert len(images) == 20
    assert images[0].name == "frame_00000.jpg"
    with Image.open(images[0]) as first:
        assert max(first.size) == 640  # downscaled 1920x1080 -> 640x360
    assert not (tmp_path / "run" / "frames_candidates").exists()  # cleaned up

    info = manifest.step_info("extract_frames")
    assert info["num_candidates"] >= 38  # oversample 2x of 20 (FFmpeg may round)
    assert info["motion_score"] > 5
    assert manifest.data["status"] == "stopped after 'extract_frames'"
    assert manifest.data["tool_versions"]["ffmpeg"].startswith("ffmpeg version")


def test_short_video_fails_at_validate(short_video, tmp_path):
    with pytest.raises(StepError) as excinfo:
        run_pipeline(short_video, tmp_path / "run", load_config("fast"))
    assert excinfo.value.step == "validate"
    assert "at least 20 s" in excinfo.value.message
    assert not (tmp_path / "run" / "colmap").exists()  # nothing after validate ran


def test_static_video_fails_at_extract_frames(static_video, tmp_path):
    config = load_config("fast", {"frames": {"target_count": 20, "max_long_side": 640}})
    with pytest.raises(StepError) as excinfo:
        run_pipeline(static_video, tmp_path / "run", config)
    assert excinfo.value.step == "extract_frames"
    assert "looks static" in excinfo.value.message
    data = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert data["steps"]["extract_frames"]["info"]["motion_score"] < 5


def test_cli_reports_failure_with_exit_code(short_video, tmp_path, capsys):
    code = main(["run", "--video", str(short_video), "--output", str(tmp_path / "run")])
    assert code == 1
    assert "Run failed at step 'validate'" in capsys.readouterr().err


def test_cli_missing_video_file(tmp_path, capsys):
    code = main(
        ["run", "--video", str(tmp_path / "nope.mp4"), "--output", str(tmp_path / "r")]
    )
    assert code == 1
    assert "Video file not found" in capsys.readouterr().out
