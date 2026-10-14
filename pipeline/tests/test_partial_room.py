"""Continue with a partial room when COLMAP places too few frames (D-019)."""

import struct
from pathlib import Path

import pytest
from test_poses_cameras import FakeColmap, make_run
from test_ui import FakePipeline, write_manifest

from colibrimo_pipeline.config import load_config
from colibrimo_pipeline.errors import StepError
from colibrimo_pipeline.steps import poses
from colibrimo_pipeline.ui.jobs import Runner, UserError, read_progress


class FewFramesColmap(FakeColmap):
    """The mapper places only ``placed`` of the 7 frames (like a broken capture)."""

    def __init__(self, monkeypatch, placed):
        super().__init__(monkeypatch)
        self.placed = placed

    def run(self, command, args):
        self.calls.append((command, list(args)))
        if command == "mapper":
            model = Path(args[args.index("--output_path") + 1]) / "0"
            model.mkdir(parents=True)
            (model / "images.bin").write_bytes(struct.pack("<Q", self.placed))
            (model / "points3D.bin").write_bytes(struct.pack("<Q", 100))
            (model.parent.parent / "database.db").write_bytes(b"db")


def settings(ctx, **poses_settings):
    ctx.config = load_config(
        "fast", {"poses": {"min_registered_frames": 2, **poses_settings}}
    )
    return ctx


def test_default_still_fails_but_suggests_continuing(tmp_path, monkeypatch):
    FewFramesColmap(monkeypatch, placed=3)  # 3 of 7 = 43% < 70%
    ctx = settings(make_run(tmp_path, same_camera=True))
    with pytest.raises(StepError) as excinfo:
        poses.run(ctx)
    assert "placed only 3 of 7 frames" in excinfo.value.message
    assert "poses.allow_partial=true" in excinfo.value.message
    assert excinfo.value.info["partial"] is True
    assert (tmp_path / poses.ATTEMPT_FILE).is_file()  # kept for "continue anyway"


def test_continue_reuses_colmap_and_marks_the_room_partial(tmp_path, monkeypatch):
    colmap = FewFramesColmap(monkeypatch, placed=3)
    ctx = settings(make_run(tmp_path, same_camera=True))
    with pytest.raises(StepError):
        poses.run(ctx)
    calls_before = len(colmap.calls)

    result = poses.run(settings(ctx, allow_partial=True))
    assert len(colmap.calls) == calls_before  # COLMAP not run again
    assert result.info["partial"] is True
    assert result.info["reused_previous_attempt"] is True
    assert "PARTIAL ROOM" in result.info["warning"]
    assert ctx.manifest.data["partial_room"] == {
        "frames_placed": 3,
        "frames_total": 7,
        "ratio": 0.4286,
        "min_registered_ratio": 0.7,
    }


def test_partial_room_still_needs_the_minimum_frames(tmp_path, monkeypatch):
    FewFramesColmap(monkeypatch, placed=3)
    ctx = settings(make_run(tmp_path, same_camera=True), allow_partial=True)
    ctx.config.poses.min_registered_frames = 5
    with pytest.raises(StepError, match="needs at least 5 placed frames"):
        poses.run(ctx)


def test_changed_settings_run_colmap_again(tmp_path, monkeypatch):
    colmap = FewFramesColmap(monkeypatch, placed=3)
    ctx = settings(make_run(tmp_path, same_camera=True))
    with pytest.raises(StepError):
        poses.run(ctx)
    calls_before = len(colmap.calls)
    poses.run(settings(ctx, allow_partial=True, max_num_features=4096))
    assert len(colmap.calls) > calls_before  # different settings: recomputed


def test_full_room_is_not_partial(tmp_path, monkeypatch):
    FewFramesColmap(monkeypatch, placed=7)
    ctx = settings(make_run(tmp_path, same_camera=True), allow_partial=True)
    result = poses.run(ctx)
    assert result.info["partial"] is False
    assert "partial_room" not in ctx.manifest.data


# ---- cpipe ui ----------------------------------------------------------------------


def failed_for_too_few_frames(run_dir, placed):
    write_manifest(
        run_dir,
        {
            "poses": {
                "status": "failed",
                "error": "COLMAP placed only ...",
                "info": {
                    "num_registered": placed,
                    "num_images": 300,
                    "registered_ratio": placed / 300,
                },
            }
        },
        status="failed",
        input={"videos": [], "names": []},
    )


def test_ui_offers_continue_only_when_it_applies(tmp_path):
    run = tmp_path / "runs" / "room"
    failed_for_too_few_frames(run, placed=68)
    assert read_progress(run)["can_continue_partial"] is True
    failed_for_too_few_frames(run, placed=12)  # below the 20-frame floor
    assert read_progress(run)["can_continue_partial"] is False


def test_ui_continue_resumes_with_allow_partial(tmp_path):
    (tmp_path / "uploads" / "room").mkdir(parents=True)
    (tmp_path / "uploads" / "room" / "middle.mp4").write_bytes(b"v")
    fake = FakePipeline()
    runner = Runner(tmp_path, run_fn=fake)
    with pytest.raises(UserError, match="Nothing to continue"):
        runner.continue_partial("room")
    failed_for_too_few_frames(tmp_path / "runs" / "room", placed=68)

    runner.continue_partial("room")
    runner.wait(30)
    videos, run_dir, preset = fake.calls[0]
    assert fake.forces == [False]  # resume: finished steps reused
    assert fake.configs[0].poses.allow_partial is True


def test_ui_shows_partial_room(tmp_path):
    run = tmp_path / "runs" / "room"
    write_manifest(
        run,
        {"export": {"status": "succeeded"}},
        status="succeeded",
        partial_room={"frames_placed": 68, "frames_total": 300, "ratio": 0.2267},
    )
    assert read_progress(run)["partial_room"]["frames_placed"] == 68
