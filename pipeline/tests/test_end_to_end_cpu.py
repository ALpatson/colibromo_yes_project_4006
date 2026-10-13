"""Real FFmpeg runs on synthetic videos (no GPU, no COLMAP needed).

Covers the acceptance criterion "a deliberately bad video (5 s, or a static
shot) fails early with a clear, human-readable error".
"""

import json

import pytest
from conftest import make_video
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
    assert images[0].name == "good_00000.jpg"  # named after the video
    with Image.open(images[0]) as first:
        assert max(first.size) == 640  # downscaled 1920x1080 -> 640x360
    assert not (tmp_path / "run" / "frames_candidates").exists()  # cleaned up

    info = manifest.step_info("extract_frames")
    assert info["num_candidates"] >= 38  # oversample 2x of 20 (FFmpeg may round)
    assert info["motion_score"] > 5
    assert manifest.data["status"] == "stopped after 'extract_frames'"
    assert manifest.data["tool_versions"]["ffmpeg"].startswith("ffmpeg version")


def test_several_videos_share_frames_by_duration(video_dir, tmp_path):
    """Three videos of one room: frames split by duration, named per video, recorded."""
    long_video = make_video(video_dir / "bottom.mp4", duration=20)
    mid_video = make_video(video_dir / "middle.mp4", duration=10)
    short_video = make_video(video_dir / "up.mp4", duration=10, size="1280x720")
    config = load_config(
        "fast",
        {"frames": {"target_count": 40, "min_per_video": 5, "max_long_side": 320}},
    )
    run = tmp_path / "run"
    manifest = run_pipeline(
        [long_video, mid_video, short_video], run, config, until="extract_frames"
    )

    names = sorted(p.name for p in (run / "colmap" / "images").glob("*.jpg"))
    assert len(names) == 40
    by_video = {
        v: [n for n in names if n.startswith(f"{v}_")]
        for v in ("bottom", "middle", "up")
    }
    assert [len(by_video[v]) for v in ("bottom", "middle", "up")] == [20, 10, 10]
    assert by_video["bottom"][0] == "bottom_00000.jpg"
    assert by_video["bottom"][-1] == "bottom_00019.jpg"

    validate = manifest.step_info("validate")
    assert [v["name"] for v in validate["videos"]] == ["bottom", "middle", "up"]
    assert validate["total_duration_s"] == pytest.approx(40, abs=0.5)
    assert validate["same_camera"] is False  # up.mp4 is 720p
    assert any("different resolutions" in w for w in validate["warnings"])
    assert manifest.data["input"]["videos"][2].endswith("up.mp4")

    extract = manifest.step_info("extract_frames")
    assert [s["num_frames"] for s in extract["videos"]] == [20, 10, 10]
    assert extract["videos"][2]["last_frame"] == "up_00009.jpg"

    frames = json.loads((run / "frames.json").read_text())["frames"]
    assert len(frames) == 40
    up_frames = [f for f in frames if f["video"] == "up"]
    assert [f["index"] for f in up_frames] == list(range(10))
    times = [f["time_s"] for f in up_frames]
    assert times == sorted(times) and times[-1] < 10.5


def test_total_duration_counts_for_the_minimum(video_dir, tmp_path):
    """Two 12 s videos (24 s together) pass the 20 s minimum."""
    a = make_video(video_dir / "a.mp4", duration=12)
    b = make_video(video_dir / "b.mp4", duration=12)
    manifest = run_pipeline(
        [a, b], tmp_path / "run", load_config("fast"), until="validate"
    )
    assert manifest.step_info("validate")["total_duration_s"] == pytest.approx(
        24, abs=0.5
    )


def test_cli_videos_folder(video_dir, tmp_path):
    folder = tmp_path / "room"
    folder.mkdir()
    for name in ("b_low.mp4", "a_high.mp4"):
        make_video(folder / name, duration=12)
    (folder / "notes.txt").write_text("not a video")
    code = main(
        [
            "run",
            "--videos",
            str(folder),
            "--output",
            str(tmp_path / "run"),
            "--until",
            "validate",
        ]
    )
    assert code == 0
    data = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert [v["name"] for v in data["steps"]["validate"]["info"]["videos"]] == [
        "a_high",
        "b_low",
    ]


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
