"""Several videos per room: naming, frame split, duration rules, matcher choice."""

from pathlib import Path

import pytest

from colibrimo_pipeline.config import PosesConfig, ValidationConfig
from colibrimo_pipeline.errors import ConfigError, StepError
from colibrimo_pipeline.steps.extract_frames import split_frame_counts
from colibrimo_pipeline.steps.poses import choose_matcher
from colibrimo_pipeline.steps.validate import VideoInfo, check_videos
from colibrimo_pipeline.utils.videos import collect_videos, video_name


def info(path="v.mp4", duration=60.0, width=1920, height=1080):
    return VideoInfo(path, duration, width, height, 30.0, "h264", 0, 1, "mp4")


# ---- names ------------------------------------------------------------------


@pytest.mark.parametrize(
    "file_name, expected",
    [
        ("bottom.mp4", "bottom"),
        ("Living Room (2).MOV", "Living_Room_2"),
        ("up-pass_1.m4v", "up-pass_1"),
        ("###.mp4", "video"),
    ],
)
def test_video_name(file_name, expected):
    assert video_name(Path(file_name)) == expected


def test_collect_keeps_order_and_names(tmp_path):
    videos = collect_videos([tmp_path / "bottom.mp4", tmp_path / "middle.mp4"])
    assert [v.name for v in videos] == ["bottom", "middle"]
    assert videos[0].path == (tmp_path / "bottom.mp4").resolve()


def test_collect_single_path_still_works(tmp_path):
    assert [v.name for v in collect_videos(tmp_path / "room1.mp4")] == ["room1"]


def test_collect_folder_sorted_and_filtered(tmp_path):
    for name in ("up.MOV", "bottom.mp4", "readme.txt", "middle.mp4"):
        (tmp_path / name).write_bytes(b"")
    videos = collect_videos(folder=tmp_path)
    assert [v.name for v in videos] == ["bottom", "middle", "up"]


def test_collect_same_file_twice_is_used_once(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"")
    videos = collect_videos([tmp_path / "a.mp4"], folder=tmp_path)
    assert [v.name for v in videos] == ["a"]


def test_collect_rejects_clashing_names(tmp_path):
    with pytest.raises(ConfigError, match="same name 'room'"):
        collect_videos([tmp_path / "x" / "room.mp4", tmp_path / "y" / "room.MOV"])


def test_collect_needs_at_least_one_video():
    with pytest.raises(ConfigError, match="No input video"):
        collect_videos([])


def test_collect_folder_without_videos(tmp_path):
    with pytest.raises(ConfigError, match="No video files"):
        collect_videos(folder=tmp_path)


# ---- frame split --------------------------------------------------------------


def test_split_in_proportion_to_duration():
    # myRoom2: bottom 93.5 s, middle 81.9 s, up 56.9 s
    counts = split_frame_counts([93.5, 81.9, 56.9], 300, 20)
    assert sum(counts) == 300
    assert counts == [121, 106, 73]


def test_split_total_is_always_exact():
    for total in (10, 99, 300, 501):
        assert sum(split_frame_counts([1.0, 2.0, 3.3, 0.7], total, 5)) == total


def test_split_raises_short_videos_to_the_minimum():
    counts = split_frame_counts([100.0, 2.0], 100, 20)
    assert counts == [80, 20]


def test_split_minimum_lowered_when_target_too_small():
    assert split_frame_counts([10.0, 10.0, 10.0], 10, 20) == [4, 3, 3]


def test_split_single_video_gets_everything():
    assert split_frame_counts([42.0], 300, 20) == [300]


# ---- duration rules -------------------------------------------------------------


def test_minimum_applies_to_the_total():
    assert (
        check_videos([info("a.mp4", 12), info("b.mp4", 12)], ValidationConfig()) == []
    )


def test_total_below_minimum_fails():
    with pytest.raises(StepError, match="15.0 s long in total"):
        check_videos([info("a.mp4", 10), info("b.mp4", 5)], ValidationConfig())


def test_maximum_applies_per_video():
    # 3 x 4 min = 12 min in total is fine; one 6 min video is not.
    ok = [info(f"{n}.mp4", 240) for n in "abc"]
    assert check_videos(ok, ValidationConfig()) == []
    with pytest.raises(StepError, match=r"Video 'b.mp4' is 6.0 min long"):
        check_videos([info("a.mp4", 60), info("b.mp4", 360)], ValidationConfig())


def test_different_resolutions_warn():
    warnings = check_videos(
        [info("a.mp4"), info("b.mp4", width=1080, height=1920)], ValidationConfig()
    )
    assert any("different orientations" in w for w in warnings)
    assert any("b.mp4 portrait 1080x1920" in w for w in warnings)
    assert any("landscape recommended" in w for w in warnings)


# ---- matcher choice -------------------------------------------------------------


def test_auto_one_video_is_sequential():
    choice = choose_matcher(PosesConfig(), num_videos=1)
    assert (choice.matcher, choice.loop_detection) == ("sequential", False)


def test_auto_several_videos_uses_loop_detection():
    choice = choose_matcher(PosesConfig(), num_videos=3)
    assert (choice.matcher, choice.loop_detection) == ("sequential", True)
    assert "3 videos" in choice.reason


def test_explicit_matcher_is_respected():
    cfg = PosesConfig(matcher="exhaustive")
    assert choose_matcher(cfg, num_videos=3).matcher == "exhaustive"
    cfg = PosesConfig(matcher="sequential")
    assert choose_matcher(cfg, num_videos=3).loop_detection is False
    cfg = PosesConfig(matcher="sequential", loop_detection=True)
    assert choose_matcher(cfg, 1).loop_detection is True
    cfg = PosesConfig(matcher="exhaustive", loop_detection=True)
    assert choose_matcher(cfg, 1).loop_detection is False
